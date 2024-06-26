# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from enum import Enum
import functools
import ever as er
import ever.module as M
from einops import rearrange
from ever.core.registry import Registry
from ever.core.dist import get_rank, get_world_size

from collections import OrderedDict
from inspect import isfunction
from timm.models.layers import DropPath

import copy


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


Models = dict(
    deeplabv3=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.ListIndex(-1)),
        ('head', M.AtrousSpatialPyramidPool),
    ]),
    deeplabv3p=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.ListIndex(0, -1)),
        ('head', M.Deeplabv3pDecoder),
    ]),
    pspnet=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.ListIndex(-1)),
        ('head', M.PyramidPoolModule),
    ]),
    semantic_fpn=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.FPN),
        ('head', M.AssymetricDecoder)
    ]),
    farseg=OrderedDict([
        ('backbone', M.ResNetEncoder),
        ('neck', M.ListIndex(0, 1, 2, 3)),
        ('head', M.FarSegHead),
    ]),
)


@er.registry.MODEL.register()
class Segmentation(er.ERModule):
    def __init__(self, config):
        super(Segmentation, self).__init__(config)

        odict = Models[self.config.model_type]
        for k, v in odict.items():
            if isinstance(v, nn.Module):
                odict[k] = v
            elif isfunction(v):
                odict[k] = v(**self.config[k])
            elif issubclass(v, er.ERModule):
                odict[k] = v(self.config[k])
            elif issubclass(v, nn.Module):
                odict[k] = v(**self.config[k])

        self.features = nn.Sequential(odict)

    def forward(self, x, y=None):
        logit = self.features(x)
        return logit

    def set_default_config(self):
        self.config.update(dict())


class DropConnect(nn.Module):
    def __init__(self, drop_rate):
        super(DropConnect, self).__init__()
        self.p = drop_rate

    def forward(self, inputs):
        """Drop connect.
            Args:
                input (tensor: BCWH): Input of this structure.
                p (float: 0.0~1.0): Probability of drop connection.
                training (bool): The running mode.
            Returns:
                output: Output after drop connection.
        """
        p = self.p
        assert 0 <= p <= 1, 'p must be in range of [0,1]'

        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        keep_prob = 1 - p

        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)

        output = inputs / keep_prob * binary_tensor
        return output


DETECTOR = Registry()


def get_detector(name, **kwargs):
    if name in DETECTOR:
        return DETECTOR[name](**kwargs)

    raise ValueError(f'{name} is not supported.')


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


@er.registry.MODEL.register()
class ChangeStar2(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        segmentation = Segmentation(self.config.segmentation)

        classifier = M.ConvUpsampling(
            self.config.semantic_classifier.in_channels,
            self.config.semantic_classifier.out_channels,
            self.config.semantic_classifier.scale,
            3, 1, 1
        )

        detector = get_detector(**self.config.change_detector)

        name = self.config.target_generator.pop('name')
        target_generator = TargetGenerator(name, **self.config.target_generator)

        self.changemixin = ChangeMixin2(
            segmentation,
            classifier,
            detector,
            target_generator,
            self.config.loss
        )

    def forward(self, x, y=None):
        predictions = self.changemixin(x, y)
        if not self.training:
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

        return predictions

    def set_default_config(self):
        # cstar2_r50_farseg
        self.config.update(dict(
            segmentation=dict(
                model_type='farseg',
                backbone=dict(
                    resnet_type='resnet50',
                    pretrained=True,
                    freeze_at=0,
                    output_stride=32,
                ),
                head=dict(
                    fpn=dict(
                        in_channels_list=(256, 512, 1024, 2048),
                        out_channels=256,
                    ),
                    fs_relation=dict(
                        scene_embedding_channels=2048,
                        in_channels_list=(256, 256, 256, 256),
                        out_channels=256,
                        scale_aware_proj=True
                    ),
                    fpn_decoder=dict(
                        in_channels=256,
                        out_channels=256,
                        in_feat_output_strides=(4, 8, 16, 32),
                        out_feat_output_stride=4,
                        classifier_config=None
                    )
                ),
            ),
            semantic_classifier=dict(
                in_channels=256,
                out_channels=1,
                scale=4.0
            ),
            change_detector=dict(
                name='TSMTDM',
                in_channels=256,
                scale=4.0,
                tsm_cfg=dict(
                    dim=16,
                    drop_path_prob=0.2,
                    num_convs=4,
                ),
                tdm_cfg=dict(
                    NConvNeXtBlock=9,
                    PreNorm='LN'
                ),
            ),
            target_generator=dict(
                name='sync_generate_target_v3',
                shuffle_prob=1.0
            ),
            loss=dict(
                change=dict(
                    symmetry_loss=True,
                    bce=True,
                    dice=False,
                    weight=0.5,
                    ignore_index=-1,
                    log_bce_pos_neg_stat=True,
                ),
                semantic=dict(
                    on=True,
                    bce=True,
                    dice=True,
                    ignore_index=-1,
                ),
            ),
            pcm_m2m_inference=False,
        ))

    def log_info(self):
        return dict(
            cfg=self.config,
            arch=self
        )

    def custom_param_groups(self):
        if self.cfg.segmentation.model_type.startswith('swin'):
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for i, p in self.named_parameters():
                if 'norm' in i:
                    param_groups[0]['params'].append(p)
                elif 'relative_position_bias_table' in i:
                    param_groups[0]['params'].append(p)
                elif 'absolute_pos_embed' in i:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        else:
            return self.parameters()
