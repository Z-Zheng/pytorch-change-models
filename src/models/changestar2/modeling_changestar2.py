# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import functools
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from transformers import PreTrainedModel
from timm.models.layers import DropPath

import ever as er
import ever.module as M
import ever.module.loss as L
from ever.core.registry import Registry
from ever.core.dist import get_rank, get_world_size

from .configuration_changestar2 import ChangeStar2Config

# Constants for output keys
CHANGE = 'change_prediction'
T1SEM = 't1_semantic_prediction'
T2SEM = 't2_semantic_prediction'


class Field(Enum):
    MASK1 = 1
    MASK2 = 2
    XIMG1 = 3
    XMASK1 = 4


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather_tensor(data, group=None):
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(output, data, group=group)
    return output


def generate_target_v3(x1, y, shuffle_prob=0.5):
    N = x1.size(0)
    x_x1 = y[Field.XIMG1]
    x_mask1 = y[Field.XMASK1]

    shuffle_inds = torch.randperm(N, device=x1.device)
    regular_inds = torch.arange(N, device=x1.device)
    prob = torch.rand(N, device=x1.device)
    shuffle_inds = torch.where(prob < shuffle_prob, shuffle_inds, regular_inds)

    pseudo_x2 = x_x1[shuffle_inds, ...]
    x = torch.cat([x1, pseudo_x2], dim=1)

    pseudo_mask2 = x_mask1[shuffle_inds, ...]
    y[Field.MASK2] = pseudo_mask2

    return x, y


def sync_generate_target_v3(x1, y, shuffle_prob=0.5):
    rank = get_rank()
    world_size = get_world_size()

    # fallback to single gpu generate_target when not necessary
    if world_size <= 1:
        return generate_target_v3(x1, y, shuffle_prob)

    bs = x1.size(0)
    N = bs * world_size

    shuffle_inds = torch.randperm(N, device=x1.device)
    regular_inds = torch.arange(N, device=x1.device)
    prob = torch.rand(N, device=x1.device)
    shuffle_inds = torch.where(prob < shuffle_prob, shuffle_inds, regular_inds)

    dist.broadcast(shuffle_inds, src=0)
    shuffle_inds = shuffle_inds.tolist()

    x1 = all_gather_tensor(x1)
    x1 = torch.cat(x1, dim=0)

    x_x1 = y[Field.XIMG1]
    x_x1 = all_gather_tensor(x_x1)
    x_x1 = torch.cat(x_x1, dim=0)
    pseudo_x2 = x_x1[shuffle_inds, ...]
    x = torch.cat([x1, pseudo_x2], dim=1)

    x = torch.split(x, bs, dim=0)[rank]

    x_mask1 = y[Field.XMASK1]
    x_mask1 = all_gather_tensor(x_mask1)
    x_mask1 = torch.cat(x_mask1, dim=0)
    pseudo_mask2 = x_mask1[shuffle_inds, ...]
    pseudo_mask2 = torch.split(pseudo_mask2, bs, dim=0)[rank]
    y[Field.MASK2] = pseudo_mask2

    return x, y


class TargetGenerator:
    def __init__(self, name, **kwargs):
        self.name = name
        self.params = kwargs

    def __call__(self, x, y):
        if self.name == 'sync_generate_target_v3':
            return sync_generate_target_v3(x, y, **self.params)
        else:
            raise NotImplementedError(f'{self.name}')


def pcm_m2m_inference(s1, s2, c):
    classes = s1.size(1)

    s1 = s1.unsqueeze_(dim=2)  # N, C, 1, H, W
    s2 = s2.unsqueeze_(dim=1)  # N, 1, C, H, W
    cs = s1 * s2  # N, C, C, H, W

    c = c.unsqueeze_(dim=2)
    w = torch.eye(classes, device=c.device).reshape(1, classes, classes, 1, 1)
    w = w * (1 - c) + (1 - w) * c
    cs = cs.mul_(w)

    cs = torch.flatten(cs, 1, 2)

    cs = cs.argmax(dim=1)  # N H W

    # decoding
    s1 = torch.div(cs, classes, rounding_mode='floor')
    s2 = cs % classes

    return s1, s2, (s1 != s2).to(torch.uint8)


# Segmentation model registry
Models = {
    'deeplabv3': {
        'backbone': M.ResNetEncoder,
        'neck': M.ListIndex(-1),
        'head': M.AtrousSpatialPyramidPool,
    },
    'deeplabv3p': {
        'backbone': M.ResNetEncoder,
        'neck': M.ListIndex(0, -1),
        'head': M.Deeplabv3pDecoder,
    },
    'pspnet': {
        'backbone': M.ResNetEncoder,
        'neck': M.ListIndex(-1),
        'head': M.PyramidPoolModule,
    },
    'semantic_fpn': {
        'backbone': M.ResNetEncoder,
        'neck': M.FPN,
        'head': M.AssymetricDecoder
    },
    'farseg': {
        'backbone': M.ResNetEncoder,
        'neck': M.ListIndex(0, 1, 2, 3),
        'head': M.FarSegHead,
    },
}


class Segmentation(nn.Module):
    def __init__(self, config):
        super(Segmentation, self).__init__()
        
        model_type = config.get('model_type', 'farseg')
        if model_type not in Models:
            raise ValueError(f"Model type {model_type} not supported")
            
        odict = Models[model_type]
        modules = []
        
        for k, v in odict.items():
            if isinstance(v, nn.Module):
                modules.append((k, v))
            elif hasattr(v, '__call__'):
                if k in config:
                    modules.append((k, v(**config[k])))
                else:
                    modules.append((k, v()))
            else:
                raise ValueError(f"Invalid module type for {k}")
        
        self.features = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self.features(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim,
                 kernel_size=7,
                 drop_path=0.,
                 mlp_ratio=4,
                 layer_scale_init_value=1e-6,
                 ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
                                groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class TSM(nn.Module):
    def __init__(self, in_channels, dim, num_convs, drop_prob, bn):
        super(TSM, self).__init__()
        self.agg = M.ConvBlock(in_channels * 2, dim, 1, bias=False, bn=True)
        layers = []
        for _ in range(num_convs - 1):
            layers.append(M.ConvBlock(dim, dim, 3, 1, 1, bias=False, bn=bn))
        layers.append(DropPath(drop_prob))
        self.convs = nn.Sequential(*layers)

    def forward(self, t1, t2):
        x = self.agg(torch.cat([t1, t2], dim=1))
        residual = x
        x = self.convs(x)
        x = x + residual
        return x


class Repeat(nn.Module):
    def __init__(self, module, n: int):
        super().__init__()
        assert n >= 1
        if n == 1:
            self.ops = module
        else:
            layers = [module]
            for _ in range(n - 1):
                layers.append(copy.deepcopy(module))
            self.ops = nn.Sequential(*layers)

    def forward(self, x):
        return self.ops(x)


class TDM(nn.Module):
    def __init__(self, in_channels, dim, **kwargs):
        super().__init__()
        if kwargs.get('PreNorm', 'BN') == 'BN':
            prenorm = nn.BatchNorm2d(in_channels, affine=False)
        elif kwargs.get('PreNorm', 'BN') == 'LN':
            prenorm = nn.GroupNorm(1, in_channels, affine=False)
        else:
            raise NotImplementedError

        self.nproj = nn.Sequential(
            prenorm,
            nn.Conv2d(in_channels, dim, 1),
            nn.ReLU(True),
            Repeat(ConvNeXtBlock(dim, kwargs.get('k', 7), drop_path=0.0), kwargs.get('NConvNeXtBlock', 1))
        )
        self.diff_op_name = kwargs.get('op_name', 'abs')

    def forward(self, t1, t2):
        diff = self.diff_op(self.diff_op_name)(t1, t2)
        diff = self.nproj(diff)
        return diff

    def diff_op(self, op_name):
        if 'abs' == op_name:
            return lambda a, b: (a - b).abs()
        elif 'mul' == op_name:
            return lambda a, b: a * b
        elif 'square' == op_name:
            return lambda a, b: (a - b).square()

        raise NotImplementedError(op_name)


DETECTOR = Registry()


def get_detector(name, **kwargs):
    if name in DETECTOR:
        return DETECTOR[name](**kwargs)
    raise ValueError(f'{name} is not supported.')


@DETECTOR.register()
class TSMTDM(nn.Module):
    def __init__(
            self,
            in_channels,
            scale,
            tsm_cfg,
            tdm_cfg,
            pre_scale=1
    ):
        super().__init__()
        tsm_dim = tsm_cfg['dim']
        dp_prob = tsm_cfg['drop_path_prob']
        num_convs = tsm_cfg['num_convs']
        use_bn = tsm_cfg.get('bn', True)

        self.tsm = TSM(in_channels, tsm_dim, num_convs, dp_prob, use_bn)
        if tdm_cfg is not None:
            self.tdm = TDM(in_channels, tsm_dim, **tdm_cfg)
        else:
            self.tdm = None

        self.proj = M.ConvBlock(tsm_dim, tsm_dim, 1)
        self.conv_cls = M.ConvUpsampling(tsm_dim, 1, scale, 1)

        if pre_scale > 1:
            self.pre_transform = nn.UpsamplingBilinear2d(scale_factor=pre_scale)
        else:
            self.pre_transform = nn.Identity()

    def forward(self, x):
        t1 = self.pre_transform(x[0])
        t2 = self.pre_transform(x[1])
        # TSM
        features = self.tsm(t1, t2)
        # TDM
        if self.tdm is not None:
            diff_t = self.tdm(t1, t2)
            features = features + diff_t

        t = self.proj(F.relu(features, inplace=True))
        return self.conv_cls(t)


def classification_loss(y_true: torch.Tensor, y_pred: torch.Tensor, loss_config, prefix=''):
    loss_dict = dict()
    weight = loss_config.get('weight', 1.0)
    if loss_config.get('dice', False):
        loss_dict[f'{prefix}dice_loss'] = weight * M.loss.dice_loss_with_logits(y_pred, y_true,
                                                                                ignore_index=loss_config.ignore_index)

    if 'tver' in loss_config:
        alpha = loss_config.tver.alpha
        beta = round(1. - alpha, 2)
        _weight = loss_config.tver.get('weight', 1.0)
        smooth_value = loss_config.tver.get('smooth_value', 1.0)
        loss_dict[f'{prefix}tver[{alpha},{beta}]@w{_weight}_loss'] = _weight * M.loss.tversky_loss_with_logits(
            y_pred, y_true.float(),
            alpha, beta,
            smooth_value=smooth_value,
            ignore_index=loss_config.ignore_index)

    if loss_config.get('bce', False):
        losses = weight * M.loss.binary_cross_entropy_with_logits(y_pred, y_true,
                                                                  ignore_index=loss_config.ignore_index,
                                                                  reduction='none')
        loss_dict[f'{prefix}bce_loss'] = losses.mean()

        if loss_config.get('log_bce_pos_neg_stat', False):
            with torch.no_grad():
                output, target = M.loss._masked_ignore(y_pred, y_true, loss_config.ignore_index)

                pos_mask = target > 0
                pos_losses = losses[pos_mask]
                neg_losses = losses[~pos_mask]
                loss_dict[f'{prefix}bce_pos_mean'] = pos_losses.mean() if pos_losses.numel() > 0 else y_pred.new_tensor(
                    0.)
                loss_dict[f'{prefix}bce_neg_mean'] = neg_losses.mean() if neg_losses.numel() > 0 else y_pred.new_tensor(
                    0.)

                loss_dict[f'{prefix}bce_pos'] = pos_losses.sum() / target.size(0)
                loss_dict[f'{prefix}bce_neg'] = neg_losses.sum() / target.size(0)

                loss_dict[f'{prefix}PN_ratio'] = y_pred.new_tensor(pos_losses.numel() / (neg_losses.numel() + 1e-7))

    if 'ce' in loss_config:
        ls = loss_config.ce.get('label_smooth', 0.0)
        loss_dict[f'{prefix}ce_loss'] = weight * F.cross_entropy(y_pred, y_true.long(),
                                                                 ignore_index=loss_config.ignore_index,
                                                                 label_smoothing=ls)

    return loss_dict


def semantic_and_symmetry_loss(
        y1_true,
        vy2_true,
        y1_logit,
        change_y1vy2_logit,
        change_y2vy1_logit,
        loss_config
):
    total_loss = dict()
    change_type = getattr(loss_config, 'change_type', 'binary')
    if change_type == 'binary':
        y1_true = y1_true > 0
        vy2_true = vy2_true > 0
        positive_mask = torch.logical_xor(y1_true, vy2_true)
        num_pp = (y1_true & vy2_true).float().sum()
        num_nn = (~(y1_true | vy2_true)).float().sum()

        total_loss['num_pp'] = num_pp
        total_loss['num_nn'] = num_nn
        total_loss['num_pn'] = positive_mask.float().sum()

    elif change_type == 'multi_class':
        positive_mask = (y1_true != vy2_true).float()
        ignore_mask = (y1_true == -1) | (vy2_true == -1)
        positive_mask = torch.where(ignore_mask, -1 * torch.ones_like(positive_mask), positive_mask)

    if 'semantic' in loss_config and getattr(loss_config.semantic, 'on', True):
        total_loss.update(classification_loss(y1_true, y1_logit, loss_config.semantic, 's'))
    if change_y1vy2_logit is not None:
        total_loss.update(
            classification_loss(positive_mask, change_y1vy2_logit, loss_config.change, 'c12'))
    if change_y2vy1_logit is not None:
        total_loss.update(
            classification_loss(positive_mask, change_y2vy1_logit, loss_config.change, 'c21'))

    return total_loss


class ChangeMixin2(nn.Module):
    def __init__(
            self,
            feature_extractor,
            classifier,
            detector,
            target_generator,
            loss_config,
    ):
        super().__init__()

        self.features = feature_extractor
        self.classifier = classifier
        self.change_detector = detector
        self.target_generator = target_generator
        self.loss_config = er.config.from_dict(loss_config)

    def forward(self, x, y=None):
        if self.training:
            x, y = self.target_generator(x, y)

        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)

        features = self.features(x)
        features = rearrange(features, '(b t) c h w -> t b c h w', t=2)

        seg_logit1 = self.classifier(features[0])

        change_logit = self.change_detector(features)

        if self.training:
            y1_true = y[Field.MASK1]
            vy2_true = y[Field.MASK2]

            loss_dict = dict()

            if self.loss_config.change.get('symmetry_loss', False):
                features = torch.flip(features, dims=(0,))
                change21_logit = self.change_detector(features)
            else:
                change21_logit = None

            loss_dict.update(semantic_and_symmetry_loss(
                y1_true,
                vy2_true,
                seg_logit1,
                change_logit,
                change21_logit,
                self.loss_config
            ))
            return loss_dict

        seg_logit2 = self.classifier(features[1])

        if getattr(self.loss_config, 'change_type', 'binary') == 'binary':
            return {
                'type': 'bcd',
                't1_semantic_prediction': seg_logit1.sigmoid(),
                't2_semantic_prediction': seg_logit2.sigmoid(),
                'change_prediction': change_logit.sigmoid(),
            }
        else:
            return {
                'type': 'scd',
                't1_semantic_prediction': seg_logit1.softmax(dim=1),
                't2_semantic_prediction': seg_logit2.softmax(dim=1),
                'change_prediction': change_logit.sigmoid(),
            }


class ChangeStar2Model(PreTrainedModel):
    """
    ChangeStar2 model for change detection with semantic segmentation.
    
    This model can be used for change detection tasks with semantic segmentation capabilities.
    """
    
    config_class = ChangeStar2Config
    base_model_prefix = "changestar2"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: ChangeStar2Config):
        super().__init__(config)
        
        # Initialize segmentation model
        self.segmentation = Segmentation(config.segmentation_config)
        
        # Initialize semantic classifier
        classifier_config = config.semantic_classifier_config
        self.classifier = M.ConvUpsampling(
            classifier_config['in_channels'],
            classifier_config['out_channels'],
            classifier_config['scale'],
            3, 1, 1
        )
        
        # Initialize change detector
        self.detector = get_detector(**config.change_detector_config)
        
        # Initialize target generator
        target_gen_config = config.target_generator_config
        name = target_gen_config.pop('name')
        self.target_generator = TargetGenerator(name, **target_gen_config)
        
        # Initialize change mixin
        self.changemixin = ChangeMixin2(
            self.segmentation,
            self.classifier,
            self.detector,
            self.target_generator,
            config.loss_config
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        Forward pass of the model.
        
        Args:
            pixel_values: Input tensor of shape (batch_size, 2*channels, height, width)
            labels: Optional labels for training
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        predictions = self.changemixin(pixel_values, labels)
        
        if self.training:
            if not return_dict:
                return predictions
            return {"loss": predictions}
        
        # Post-processing for inference
        if isinstance(predictions, dict) and predictions['type'] == 'scd':
            if self.config.pcm_m2m_inference:
                p1 = predictions['t1_semantic_prediction']
                p2 = predictions['t2_semantic_prediction']
                c = predictions['change_prediction']
                s1, s2, c = pcm_m2m_inference(p1, p2, c)
                predictions.update({
                    't1_semantic_prediction': s1,
                    't2_semantic_prediction': s2,
                })
            else:
                p1 = predictions['t1_semantic_prediction']
                p2 = predictions['t2_semantic_prediction']
                predictions.update({
                    't1_semantic_prediction': p1.argmax(dim=1),
                    't2_semantic_prediction': p2.argmax(dim=1),
                })
        
        if not return_dict:
            return predictions
        
        return {"logits": predictions}


class ChangeStar2ForChangeDetection(ChangeStar2Model):
    """
    ChangeStar2 model for change detection with a specific head for change detection.
    """
    
    def __init__(self, config: ChangeStar2Config):
        super().__init__(config)
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        Forward pass for change detection.
        
        Args:
            pixel_values: Input tensor of shape (batch_size, 2*channels, height, width)
            labels: Optional labels for training
            return_dict: Whether to return a dictionary
            
        Returns:
            Change detection outputs
        """
        outputs = super().forward(pixel_values, labels, return_dict)
        
        if not self.training:
            # For inference, return change predictions
            if isinstance(outputs, dict):
                return {"change_logits": outputs["logits"][CHANGE]}
            else:
                return outputs[CHANGE]
        
        return outputs 