# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel
from torch.cuda.amp import autocast
from timm.models.layers import DropPath
from timm.models.swin_transformer import window_partition, window_reverse, to_2tuple, WindowAttention
import math
import numpy as np

import ever as er
import ever.module as M
import ever.module.loss as L

from .configuration_changesparse import ChangeSparseConfig

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import get_encoder
except ImportError:
    print(f"segmentation_models_pytorch not found. please `pip install segmentation_models_pytorch`")

CHANGE = 'change_prediction'
T1SEM = 't1_semantic_prediction'
T2SEM = 't2_semantic_prediction'


def bitemporal_forward(module, x):
    """Forward pass for bitemporal data processing."""
    x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
    features = module(x)
    if isinstance(features, list) or isinstance(features, tuple):
        t1_features, t2_features = [], []
        for feat in features:
            t1_feat, t2_feat = rearrange(feat, '(b t) c h w -> t b c h w', t=2)
            t1_features.append(t1_feat)
            t2_features.append(t2_feat)
    else:
        t1_features, t2_features = rearrange(features, '(b t) c h w -> t b c h w', t=2)
    return t1_features, t2_features


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SMPEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[2:]


def get_backbone(name, pretrained=False, **kwargs):
    if name == 'er.R50c':
        return M.ResNetEncoder(dict(
            resnet_type='resnet50_v1c',
            pretrained=pretrained,
        )), (256, 512, 1024, 2048)
    elif name == 'er.R18':
        return M.ResNetEncoder(dict(
            resnet_type='resnet18',
            pretrained=pretrained,
        )), (64, 128, 256, 512)
    elif name == 'er.R101c':
        return M.ResNetEncoder(dict(
            resnet_type='resnet101_v1c',
            pretrained=pretrained,
        )), (256, 512, 1024, 2048)
    elif name.startswith('efficientnet'):
        in_channels = kwargs.get('in_channels', 3)
        model = get_encoder(name=name, weights='imagenet' if pretrained else None, in_channels=in_channels)
        out_channels = model.out_channels[2:]
        model = SMPEncoder(model)
        return model, out_channels
    else:
        raise NotImplementedError(f'{name} is not supported now.')


class ADBN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        x = rearrange(x, 'b (t c) h w ->t b c h w', t=2)
        x = torch.abs(x[0] - x[1])
        x = self.bn(x)
        return x


class TemporalReduction(nn.Module):
    def __init__(self, single_temporal_in_channels, reduce_type='conv'):
        super().__init__()
        self.channels = single_temporal_in_channels
        if reduce_type == 'conv':
            op = M.ConvBlock
        elif reduce_type == 'ADBN':
            op = ADBN
        else:
            raise NotImplementedError

        self.temporal_convs = nn.ModuleList()
        for c in self.channels:
            self.temporal_convs.append(op(2 * c, c, 1, bias=False))

    def forward(self, features):
        return [tc(rearrange(f, '(b t) c h w -> b (t c) h w ', t=2)) for f, tc in zip(features, self.temporal_convs)]


class ConvMlp(Mlp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dwconv = nn.Conv2d(self.fc1.out_features,
                                self.fc1.out_features, 3, 1, 1, bias=False,
                                groups=self.fc1.out_features)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c ->b c h w', h=h, w=w).contiguous()
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w ->b (h w) c').contiguous()
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @autocast(dtype=torch.float32)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SparseAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def masked_attn(self, x, indices):
        device = x.device
        B, N, C = x.shape

        batch_range = torch.arange(B, device=device)[:, None]

        selected_x = x[batch_range, indices]

        selected_x = self.attn(selected_x)

        x[batch_range, indices] = selected_x

        return x

    def forward(self, x, indices):
        h, w = x.size(2), x.size(3)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = x + self.drop_path(self.masked_attn(self.norm1(x), indices))
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x, h, w))
        x = rearrange(x, 'b (h w) c ->b c h w', h=h, w=w).contiguous()
        return x


class SwinAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        if min([H, W]) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min([H, W])
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1), device=x.device)  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        x = rearrange(x, 'b c h w -> b (h w) c')
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class SimpleFusion(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_ratio=0.1):
        super(SimpleFusion, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.dropout = nn.Dropout2d(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, feat_list):
        x0 = feat_list[0]
        x0_h, x0_w = x0.size(2), x0.size(3)
        feats = [x0]
        for feat in feat_list[1:]:
            xi = F.interpolate(feat, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            feats.append(xi)

        x = torch.cat(feats, dim=1)
        x = self.fuse_conv(x)
        x = self.dropout(x)
        return x


class SparseChangeTransformer(nn.Module):
    def __init__(self,
                 in_channels_list,
                 inner_channels=192,
                 num_heads=(3, 3, 3, 3),
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 change_threshold=0.5,
                 min_keep_ratio=0.1,
                 max_keep_ratio=0.5,
                 train_max_keep=2000,
                 num_blocks=(2, 2, 2, 2),
                 disable_attn_refine=False,
                 output_type='single_scale',
                 pc_upsample='nearest',
                 ):
        super().__init__()
        self.pc_upsample = pc_upsample
        self.disable_attn_refine = disable_attn_refine
        self.train_max_keep = train_max_keep
        top_layers = [M.ConvBlock(in_channels_list[-1], inner_channels, 1, bias=False)]
        win_size = 8
        top_layers += [
            SwinAttentionBlock(inner_channels, num_heads[0],
                               window_size=win_size,
                               shift_size=0 if (i % 2 == 0) else win_size // 2,
                               mlp_ratio=4.,
                               qkv_bias=qkv_bias,
                               drop=drop,
                               attn_drop=attn_drop,
                               drop_path=drop_path)
            for i in range(num_blocks[0])
        ]
        self.top_attn = nn.Sequential(*top_layers)

        self.num_stages = len(in_channels_list) - 1
        self.region_predictor = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(inner_channels, 1, 1)
        )

        self.refine_stages = nn.ModuleList()

        for i in range(self.num_stages):
            stage = nn.ModuleList([
                SparseAttentionBlock(inner_channels, num_heads[i + 1], 4.0, qkv_bias, drop, attn_drop, drop_path)
                for _ in range(num_blocks[i + 1])])
            self.refine_stages.append(stage)

        self.conv1x1s = nn.ModuleList(
            [M.ConvBlock(in_channels_list[i], inner_channels, 1, bias=False) for i in range(self.num_stages)])
        self.reduce_convs = nn.ModuleList(
            [M.ConvBlock(inner_channels * 2, inner_channels, 1, bias=False) for _ in range(self.num_stages)])

        self.change_threshold = change_threshold
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio

        self.output_type = output_type
        if output_type == 'multi_scale':
            self.simple_fuse = SimpleFusion(inner_channels * 4, inner_channels)

        self.init_weight()

    def init_weight(self):
        prior_prob = 0.001
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.region_predictor[-1].bias, bias_value)

    def forward(self, features):
        outputs = [self.top_attn(features.pop(-1))]

        intermediate_logits = []
        estimated_change_ratios = []
        prob = None
        for i in range(len(features)):
            top = outputs[-(i + 1)]
            if i == 0:
                indices, logit, top_2x, ecr = self.change_region_predict(self.region_predictor, top)
                intermediate_logits.append(logit)
                estimated_change_ratios.append(ecr)
                prob = logit.sigmoid()
            else:
                top_2x = F.interpolate(top, scale_factor=2., mode='nearest')

                if self.pc_upsample == 'nearest':
                    prob = F.interpolate(prob, scale_factor=2., mode='nearest')
                elif self.pc_upsample == 'bilinear':
                    prob = F.interpolate(prob, scale_factor=2., mode='bilinear', align_corners=True)
                elif self.pc_upsample == 'bicubic':
                    prob = F.interpolate(prob, scale_factor=2., mode='bicubic', align_corners=True)
                else:
                    raise ValueError('unknown upsampling method.')

                indices, _ = self.prob2indices(prob)

            down = features.pop(-1)
            down = self.conv1x1s[-(i + 1)](down)
            down = self.reduce_convs[i](torch.cat([down, top_2x], dim=1))

            if not self.disable_attn_refine:
                down = self.attention_refine(self.refine_stages[i], down, indices)

            outputs.insert(0, down)
        if self.output_type == 'single_scale':
            output = outputs[0]
        elif self.output_type == 'multi_scale':
            output = self.simple_fuse(outputs)
        else:
            raise ValueError()
        return {
            'output_feature': output,
            'intermediate_logits': intermediate_logits,
            'estimated_change_ratios': estimated_change_ratios
        }

    def change_region_predict(self, region_predictor, feature):
        feature = F.interpolate(feature, scale_factor=2., mode='nearest')

        change_region_logit = region_predictor(feature)
        change_region_prob = change_region_logit.sigmoid()
        indices, estimated_change_ratio = self.prob2indices(change_region_prob)

        return indices, change_region_logit, feature, estimated_change_ratio

    def prob2indices(self, prob):
        h, w = prob.size(2), prob.size(3)

        max_num_change_regions = (prob > self.change_threshold).long().sum(dim=(1, 2, 3)).max().item()
        max_num_change_regions = max(int(self.min_keep_ratio * h * w),
                                     min(max_num_change_regions, int(self.max_keep_ratio * h * w)))

        estimated_change_ratio = max_num_change_regions / (h * w)

        if self.training:
            max_num_change_regions = min(self.train_max_keep, max_num_change_regions)

        indices = torch.argsort(prob.flatten(2), dim=-1, descending=True)[:, 0, :max_num_change_regions]
        return indices, estimated_change_ratio

    def attention_refine(self, refine_blocks, feature, indices):
        for op in refine_blocks:
            feature = op(feature, indices)
        return feature


class ChangeSparseModel(PreTrainedModel):
    """
    ChangeSparse model for change detection using sparse transformer architecture.
    
    This model uses a sparse attention mechanism to focus on change regions
    and efficiently process bitemporal data for change detection.
    """
    
    config_class = ChangeSparseConfig
    base_model_prefix = "changesparse"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: ChangeSparseConfig):
        super().__init__(config)
        
        # Initialize backbone
        self.backbone, channels = get_backbone(
            config.backbone_name,
            config.backbone_pretrained
        )
        
        # Initialize temporal reduction
        self.temporal_reduce = TemporalReduction(
            channels, 
            config.temporal_reduction_type
        )
        
        # Initialize sparse transformer
        self.sparse_transformer = SparseChangeTransformer(
            channels,
            inner_channels=config.inner_channels,
            num_heads=config.num_heads,
            qkv_bias=config.qkv_bias,
            drop=config.drop,
            attn_drop=config.attn_drop,
            drop_path=config.drop_path,
            change_threshold=config.change_threshold,
            min_keep_ratio=config.min_keep_ratio,
            max_keep_ratio=config.max_keep_ratio,
            train_max_keep=config.train_max_keep,
            num_blocks=config.num_blocks,
            disable_attn_refine=config.disable_attn_refine,
            output_type=config.output_type,
            pc_upsample=config.pc_upsample,
        )
        
        # Initialize output head
        self.change_head = M.ConvUpsampling(
            config.inner_channels, 
            config.num_change_classes, 
            4, 1
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
        
        # Process bitemporal features
        x = rearrange(pixel_values, 'b (t c) h w -> (b t) c h w', t=2)
        x = self.backbone(x)
        x = self.temporal_reduce(x)
        
        # Apply sparse transformer
        outputs = self.sparse_transformer(x)
        output_feature = outputs['output_feature']
        
        # Generate change predictions
        change_logits = self.change_head(output_feature)
        
        # Prepare outputs
        preds = {
            CHANGE: change_logits,
        }
        
        if self.training and labels is not None:
            loss_dict = self.compute_loss(preds, labels, outputs)
            if not return_dict:
                return loss_dict
            return {"loss": loss_dict}
        
        if not return_dict:
            return preds
        
        return {"logits": preds}
    
    def compute_loss(self, preds, labels, transformer_outputs):
        """Compute the total loss."""
        gt_change = (labels['masks'][-1] > 0).float()
        
        loss_dict = dict()
        
        # Main change detection loss
        if 'change' in self.config.loss_config:
            change_loss_config = self.config.loss_config['change']
            
            if 'bce' in change_loss_config:
                weight = change_loss_config['bce'].get('weight', 1.0)
                ls = change_loss_config['bce'].get('label_smooth', 0.0)
                loss = weight * L.label_smoothing_binary_cross_entropy(
                    preds[CHANGE], gt_change, eps=ls, reduction='mean'
                )
                loss_dict['bce_loss'] = loss
            
            if 'dice' in change_loss_config:
                weight = change_loss_config['dice'].get('weight', 1.0)
                loss = weight * L.dice_loss_with_logits(preds[CHANGE], gt_change)
                loss_dict['dice_loss'] = loss
        
        # Region loss for intermediate predictions
        if 'region_loss' in self.config.loss_config:
            for i, region_logit in enumerate(transformer_outputs['intermediate_logits']):
                h, w = region_logit.size(2), region_logit.size(3)
                gt_region_change = F.adaptive_max_pool2d(gt_change.unsqueeze(0), (h, w)).squeeze(0)
                
                region_loss_config = self.config.loss_config['region_loss']
                if 'bce' in region_loss_config:
                    weight = region_loss_config['bce'].get('weight', 1.0)
                    ls = region_loss_config['bce'].get('label_smooth', 0.0)
                    loss = weight * L.label_smoothing_binary_cross_entropy(
                        region_logit, gt_region_change, eps=ls, reduction='mean'
                    )
                    loss_dict[f'region_{h}x{w}_bce_loss'] = loss
        
        # Log estimated change ratios
        for region_logit, ecr in zip(transformer_outputs['intermediate_logits'], 
                                   transformer_outputs['estimated_change_ratios']):
            h, w = region_logit.size(2), region_logit.size(3)
            loss_dict[f'{h}x{w}_ECR'] = torch.as_tensor(ecr).to(region_logit.device)
        
        return loss_dict


class ChangeSparseForChangeDetection(ChangeSparseModel):
    """
    ChangeSparse model for change detection with a specific head for change detection.
    """
    
    def __init__(self, config: ChangeSparseConfig):
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