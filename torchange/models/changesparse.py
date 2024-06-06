# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

import ever as er
import ever.module as M
import ever.module.loss as L

from einops import rearrange
from segmentation_models_pytorch.encoders import get_encoder
from torch.cuda.amp import autocast
from timm.models.layers import DropPath
from timm.models.swin_transformer import window_partition, window_reverse, to_2tuple, WindowAttention
import math
import numpy as np
from skimage import measure


class LossMixin:
    def loss(self, y_true: torch.Tensor, y_pred, loss_config):
        loss_dict = dict()

        if 'prefix' in loss_config:
            prefix = loss_config.prefix
        else:
            prefix = ''

        if 'mem' in loss_config:
            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(y_pred.device)

        if 'bce' in loss_config:
            weight = loss_config.bce.get('weight', 1.0)
            loss_dict[f'{prefix}bce@w{weight}_loss'] = weight * L.label_smoothing_binary_cross_entropy(
                y_pred,
                y_true.float(),
                eps=loss_config.bce.get('label_smooth', 0.),
                reduction='mean',
                ignore_index=loss_config.ignore_index
            )
            del weight

        if 'ce' in loss_config:
            weight = loss_config.ce.get('weight', 1.0)
            loss_dict[f'{prefix}ce@w{weight}_loss'] = weight * F.cross_entropy(y_pred, y_true.long(),
                                                                               ignore_index=loss_config.ignore_index)
            del weight

        if 'dice' in loss_config:
            ignore_channel = loss_config.dice.get('ignore_channel', -1)
            weight = loss_config.dice.get('weight', 1.0)
            loss_dict[f'{prefix}dice@w{weight}_loss'] = weight * L.dice_loss_with_logits(
                y_pred, y_true.float(),
                ignore_index=loss_config.ignore_index,
                ignore_channel=ignore_channel)
            del weight

        if 'tver' in loss_config:
            alpha = loss_config.tver.alpha
            beta = round(1. - alpha, 2)
            weight = loss_config.tver.get('weight', 1.0)
            gamma = loss_config.tver.get('gamma', 1.0)
            smooth_value = loss_config.tver.get('smooth_value', 1.0)
            loss_dict[f'{prefix}tver[{alpha},{beta},{gamma}]@w{weight}_loss'] = weight * L.tversky_loss_with_logits(
                y_pred, y_true.float(),
                alpha, beta, gamma,
                smooth_value=smooth_value,
                ignore_index=loss_config.ignore_index,
            )
            del weight

        if 'log_binary_iou_sigmoid' in loss_config:
            with torch.no_grad():
                _y_pred, _y_true = L.select(y_pred, y_true, loss_config.ignore_index)
                _binary_y_true = (_y_true > 0).float()
                cls = (_y_pred.sigmoid() > 0.5).float()

            loss_dict[f'{prefix}iou-1'] = self._iou_1(_binary_y_true, cls)
        return loss_dict

    @staticmethod
    def _iou_1(y_true, y_pred, ignore_index=None):
        with torch.no_grad():
            if ignore_index:
                y_pred = y_pred.reshape(-1)
                y_true = y_true.reshape(-1)
                valid = y_true != ignore_index
                y_true = y_true.masked_select(valid).float()
                y_pred = y_pred.masked_select(valid).float()
            y_pred = y_pred.float().reshape(-1)
            y_true = y_true.float().reshape(-1)
            inter = torch.sum(y_pred * y_true)
            union = y_true.sum() + y_pred.sum()
            return inter / torch.max(union - inter, torch.as_tensor(1e-6, device=y_pred.device))


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

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


class DenseAttentionBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                         drop_path, act_layer, norm_layer)
        mlp_hidden_dim = int(dim * mlp_ratio)
        del self.mlp
        self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
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


@er.registry.MODEL.register()
class ChangeSparseBCD(er.ERModule, LossMixin):
    def __init__(self, config):
        super().__init__(config)

        self.backbone, channels = get_backbone(
            self.cfg.backbone.name,
            self.cfg.backbone.pretrained,
            drop_path_rate=self.cfg.backbone.drop_path_rate)
        self.temporal_reduce = TemporalReduction(channels, self.cfg.temporal_reduction.reduce_type)
        self.multi_stage_attn = SparseChangeTransformer(
            channels,
            **self.cfg.transformer,
        )
        self.conv_change = M.ConvUpsampling(self.cfg.transformer.inner_channels, 1, 4, 1)

    def forward(self, x, y=None):
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
        x = self.backbone(x)
        x = self.temporal_reduce(x)
        outputs = self.multi_stage_attn(x)

        output_feature = outputs['output_feature']

        logit = self.conv_change(output_feature)

        if self.training:
            gt_change = (y['masks'][-1] > 0).float()

            loss_dict = self.loss(gt_change, logit, self.config.main_loss)

            # region loss
            for i, region_logit in enumerate(outputs['intermediate_logits']):
                h, w = region_logit.size(2), region_logit.size(3)
                gt_region_change = F.adaptive_max_pool2d(gt_change.unsqueeze(0), (h, w)).squeeze(0)
                self.config.region_loss[i].prefix = f'{h}x{w}_'
                loss_dict.update(self.loss(gt_region_change, region_logit, self.config.region_loss[i]))

            # log estimated change ratio
            for region_logit, ecr in zip(outputs['intermediate_logits'], outputs['estimated_change_ratios']):
                h, w = region_logit.size(2), region_logit.size(3)
                loss_dict.update({
                    f'{h}x{w}_ECR': torch.as_tensor(ecr).to(region_logit.device)
                })
            return loss_dict

        return {
            'change_prediction': logit.sigmoid()
        }

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                name='er.R18',
                pretrained=True,
                drop_path_rate=0.
            ),
            temporal_reduction=dict(
                reduce_type='conv'
            ),
            transformer=dict(
                inner_channels=96,
                num_heads=(3, 3, 3, 3),
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                change_threshold=0.5,
                min_keep_ratio=0.01,
                max_keep_ratio=0.1,
                train_max_keep=2000,
                num_blocks=(2, 2, 2, 2),
                disable_attn_refine=False,
                output_type='single_scale'
            ),
            main_loss=dict(
                bce=dict(),
                dice=dict(),
                mem=dict(),
                log_binary_iou_sigmoid=dict(),
                ignore_index=-1
            ),
            region_loss=[
                dict(
                    ignore_index=-1,
                    prefix='1'
                )
            ]
        ))

    def log_info(self):
        return {
            'encoder': self.backbone,
            'decoder': self.multi_stage_attn
        }

    def custom_param_groups(self):
        if self.cfg.backbone.name.startswith('mit'):
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for n, p in self.named_parameters():
                if 'norm' in n:
                    param_groups[0]['params'].append(p)
                elif 'pos_block' in n:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        elif self.cfg.backbone.name.startswith('swin'):
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


class ChangeSparseTransformer_multiclass_impl(nn.Module):
    def __init__(
            self,
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
            output_type='single_scale'
    ):
        super().__init__()
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
            nn.Conv2d(inner_channels, 5, 1)
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
                prob = logit.softmax(dim=1)
            else:
                top_2x = F.interpolate(top, scale_factor=2., mode='nearest')
                prob = F.interpolate(prob, scale_factor=2., mode='nearest')

                indices, _ = self.multi_class_prob2indices(prob)

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
        change_region_prob = change_region_logit.softmax(dim=1)
        indices, estimated_change_ratio = self.multi_class_prob2indices(change_region_prob)

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

    def multi_class_prob2indices(self, prob):
        h, w = prob.size(2), prob.size(3)

        # max_num_change_regions = (prob.argmax(dim=1) > 1).long().sum(dim=(1, 2)).max().item()
        max_num_change_regions = (prob.argmax(dim=1) > 0).long().sum(dim=(1, 2)).max().item()

        max_num_change_regions = max(int(self.min_keep_ratio * h * w),
                                     min(max_num_change_regions, int(self.max_keep_ratio * h * w)))

        estimated_change_ratio = max_num_change_regions / (h * w)

        if self.training:
            max_num_change_regions = min(self.train_max_keep, max_num_change_regions)
        max_prob, _ = torch.max(prob, dim=1, keepdim=True)
        indices = torch.argsort(max_prob.flatten(2), dim=-1, descending=True)[:, 0, :max_num_change_regions]
        return indices, estimated_change_ratio

    def attention_refine(self, refine_blocks, feature, indices):
        for op in refine_blocks:
            feature = op(feature, indices)
        return feature


class SemanticDecoder(nn.Module):
    def __init__(self,
                 in_channels_list,
                 inner_channels=192,
                 num_heads=(3,),
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.1,
                 num_blocks=(2,),
                 output_type='single_scale'
                 ):
        super().__init__()
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

        self.conv1x1s = nn.ModuleList(
            [M.ConvBlock(in_channels_list[i], inner_channels, 1, bias=False) for i in range(self.num_stages)])
        self.reduce_convs = nn.ModuleList(
            [M.ConvBlock(inner_channels * 2, inner_channels, 1, bias=False) for _ in range(self.num_stages)])

        self.output_type = output_type
        if output_type == 'multi_scale':
            self.simple_fuse = SimpleFusion(inner_channels * 4, inner_channels)

    def forward(self, features):
        j = -1
        outputs = [self.top_attn(features[j])]

        for i in range(len(features) - 1):
            top = outputs[-(i + 1)]

            top_2x = F.interpolate(top, scale_factor=2., mode='nearest')
            j -= 1
            down = features[j]
            down = self.conv1x1s[-(i + 1)](down)
            down = self.reduce_convs[i](torch.cat([down, top_2x], dim=1))

            outputs.insert(0, down)
        if self.output_type == 'single_scale':
            output = outputs[0]
        elif self.output_type == 'multi_scale':
            output = self.simple_fuse(outputs)
        else:
            raise ValueError()
        return {
            'output_feature': output,
        }


def object_based_infer(pre_logit, post_logit, logit_input=True):
    loc_thresh = 0. if logit_input else 0.5
    loc = (pre_logit > loc_thresh).cpu().squeeze(1).numpy()
    dam = post_logit.argmax(dim=1).cpu().squeeze(1).numpy()

    refined_dam = np.zeros_like(dam)
    for i, (single_loc, single_dam) in enumerate(zip(loc, dam)):
        refined_dam[i, :, :] = _object_vote(single_loc, single_dam)

    return loc, refined_dam


def _object_vote(loc, dam):
    damage_cls_list = [1, 2, 3, 4]
    local_mask = loc
    labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
    region_idlist = np.unique(labeled_local)
    if len(region_idlist) > 1:
        dam_mask = dam
        new_dam = local_mask.copy()
        for region_id in region_idlist:
            if all(local_mask[local_mask == region_id]) == 0:
                continue
            region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                for dam_cls_i, cls_weight in zip(damage_cls_list, [8., 38., 25., 11.])]
            dam_index = np.argmax(region_dam_count) + 1
            new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
    else:
        new_dam = local_mask.copy()
    return new_dam


class FuseConv(nn.Sequential):
    def __init__(self, inchannels, outchannels):
        super(FuseConv, self).__init__(nn.Conv2d(inchannels, outchannels, kernel_size=1),
                                       nn.BatchNorm2d(outchannels),
                                       )
        self.relu = nn.ReLU(True)
        self.se = M.SEBlock(outchannels, 16)

    def forward(self, x):
        out = super(FuseConv, self).forward(x)
        residual = out
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


@er.registry.MODEL.register()
class ChangeSparseO2M(er.ERModule, LossMixin):
    def __init__(self, config):
        super().__init__(config)

        self.backbone, channels = get_backbone(self.cfg.backbone.name,
                                               self.cfg.backbone.pretrained)

        self.temporal_reduce = TemporalReduction(channels, self.cfg.temporal_reduction.reduce_type)
        self.multi_stage_attn = ChangeSparseTransformer_multiclass_impl(
            channels,
            **self.cfg.transformer,
        )
        self.conv_change = M.ConvUpsampling(self.cfg.transformer.inner_channels, self.cfg.num_change_classes, 4, 1)

        self.semantic_decoder = SemanticDecoder(
            channels,
            **self.cfg.semantic_decoder.transformer
        )
        c = self.cfg.semantic_decoder.transformer.inner_channels
        self.conv_loc = M.ConvUpsampling(c, 1, 4, 1)

        c1 = self.cfg.semantic_decoder.transformer.inner_channels
        c2 = self.cfg.transformer.inner_channels

        self.fuse_conv = FuseConv(c1 + c2, c2)

    def forward(self, x, y=None):
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
        x = self.backbone(x)

        features = [rearrange(i, '(b t) c h w -> t b c h w', t=2) for i in x]
        loc_features = [i[0] for i in features]

        loc_features = self.semantic_decoder(loc_features)['output_feature']
        loc_logits = self.conv_loc(loc_features)

        x = self.temporal_reduce(x)

        outputs = self.multi_stage_attn(x)
        output_feature = outputs['output_feature']

        fused_features = self.fuse_conv(torch.cat([loc_features, output_feature], dim=1))
        dam_logits = self.conv_change(fused_features)

        if self.training:
            gt_pre = (y['masks'][0]).float()
            gt_post = y['masks'][1].long()

            loss_dict = self.loss(gt_post, dam_logits, self.config.main_loss)

            loss_dict.update(self.loss(gt_pre, loc_logits, self.config.seg_loss))

            # region loss
            for i, region_logit in enumerate(outputs['intermediate_logits']):
                h, w = region_logit.size(2), region_logit.size(3)

                region_logit = F.interpolate(region_logit, gt_post.shape[1:], mode='bilinear', align_corners=True)

                self.config.region_loss[i].prefix = f'{h}x{w}_'
                loss_dict.update(self.loss(gt_post, region_logit, self.config.region_loss[i]))

            # log estimated change ratio
            for region_logit, ecr in zip(outputs['intermediate_logits'], outputs['estimated_change_ratios']):
                h, w = region_logit.size(2), region_logit.size(3)
                loss_dict.update({
                    f'{h}x{w}_ECR': torch.as_tensor(ecr).to(region_logit.device)
                })
            return loss_dict

        prob = torch.cat([loc_logits.sigmoid(), dam_logits.softmax(dim=1)], dim=1)
        if self.config.return_probs:
            return prob
        return self.postprocess(prob, self.config.changeos_mode)

    def postprocess(self, prob, changeos_mode=False):
        pre_prob = prob[:, 0:1, :, :]
        post_prob = prob[:, 1:, :, :]
        if changeos_mode:
            pr_loc, pr_dam = object_based_infer(pre_prob, post_prob, logit_input=False)
            return torch.from_numpy(pr_loc), torch.from_numpy(pr_dam)

        pr_loc = pre_prob > 0.5
        pr_dam = post_prob.argmax(dim=1, keepdim=True)
        return pr_loc, pr_dam

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                name='er.R18',
                pretrained=True,
            ),
            temporal_reduction=dict(
                reduce_type='conv'
            ),
            transformer=dict(
                inner_channels=96,
                num_heads=(3, 3, 3, 3),
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                change_threshold=0.5,
                min_keep_ratio=0.01,
                max_keep_ratio=0.1,
                train_max_keep=2000,
                num_blocks=(2, 2, 2, 2),
                disable_attn_refine=False,
                output_type='single_scale'
            ),
            num_change_classes=5,
            return_probs=False,
            changeos_mode=False,
            semantic_decoder=dict(
                transformer=dict(
                    inner_channels=96 * 2,
                    num_heads=(3,),
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.1,
                    num_blocks=(2,),
                    output_type='single_scale'
                ),
            ),
            main_loss=dict(
            ),
            seg_loss=dict(
            ),
            region_loss=[
                dict(
                    ignore_index=-1,
                    prefix='1'
                )
            ]
        ))

    def log_info(self):
        return {
            'encoder': self.backbone,
            'change_decoder': self.multi_stage_attn,
            'semantic_decoder': self.semantic_decoder
        }

    def custom_param_groups(self):
        if self.cfg.backbone.name.startswith('mit'):
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for n, p in self.named_parameters():
                if 'norm' in n:
                    param_groups[0]['params'].append(p)
                elif 'pos_block' in n:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        else:
            return self.parameters()


@er.registry.MODEL.register()
class ChangeSparseM2M(er.ERModule, LossMixin):
    def __init__(self, config):
        super().__init__(config)

        self.backbone, channels = get_backbone(self.cfg.backbone.name,
                                               self.cfg.backbone.pretrained)

        self.temporal_reduce = TemporalReduction(channels, self.cfg.temporal_reduction.reduce_type)
        self.multi_stage_attn = SparseChangeTransformer(
            channels,
            **self.cfg.transformer,
        )

        self.semantic_decoder = SemanticDecoder(channels,
                                                **self.cfg.semantic_decoder.transformer
                                                )

        c = self.cfg.semantic_decoder.transformer.inner_channels
        self.conv_semantic = M.ConvUpsampling(c + self.cfg.transformer.inner_channels,
                                              self.cfg.semantic_decoder.num_classes, 4, 1)
        self.conv_change = M.ConvUpsampling(self.cfg.transformer.inner_channels, 1, 4, 1)

    def forward(self, x, y=None):
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
        x = self.backbone(x)
        semantic_features = self.semantic_decoder(x)

        x = self.temporal_reduce(x)

        outputs = self.multi_stage_attn(x)
        change_feature = outputs['output_feature']

        semantic_feature = semantic_features['output_feature']

        semantic_feature = rearrange(semantic_feature, '(b t) c h w -> t b c h w', t=2)

        semantic_feature1 = torch.cat([semantic_feature[0], change_feature], dim=1)
        semantic_feature2 = torch.cat([semantic_feature[1], change_feature], dim=1)

        semantic_logit1 = self.conv_semantic(semantic_feature1)
        semantic_logit2 = self.conv_semantic(semantic_feature2)

        logit = self.conv_change(change_feature)

        if self.training:
            gt_t1_seg = y['masks'][0]
            gt_t2_seg = y['masks'][1]
            gt_change = y['masks'][2].float()

            loss_dict = self.loss(gt_change, logit, self.config.main_loss)
            # ChangeMask impl.
            loss_dict.update(self.loss(gt_t1_seg, semantic_logit1, self.config.t1_seg_loss))
            loss_dict.update(self.loss(gt_t2_seg, semantic_logit2, self.config.t2_seg_loss))

            # region loss
            for i, region_logit in enumerate(outputs['intermediate_logits']):
                h, w = region_logit.size(2), region_logit.size(3)
                gt_region_change = F.adaptive_max_pool2d(gt_change.unsqueeze(0), (h, w)).squeeze(0)
                self.config.region_loss[i].prefix = f'{h}x{w}_'
                loss_dict.update(self.loss(gt_region_change, region_logit, self.config.region_loss[i]))

            # log estimated change ratio
            for region_logit, ecr in zip(outputs['intermediate_logits'], outputs['estimated_change_ratios']):
                h, w = region_logit.size(2), region_logit.size(3)
                loss_dict.update({
                    f'{h}x{w}_ECR': torch.as_tensor(ecr).to(region_logit.device)
                })
            return loss_dict

        cat_logits = torch.cat([semantic_logit1, semantic_logit2, logit], dim=1)

        return {
            't1_semantic_prediction': semantic_logit1.softmax(dim=1),
            't2_semantic_prediction': semantic_logit2.softmax(dim=1),
            'change_prediction': logit.sigmoid(),
            'merged_prediction': self.pcm_inference(cat_logits)
        }

    def pcm_inference(self, cat_logits):
        classes = self.cfg.semantic_decoder.num_classes

        s1 = cat_logits[:, :classes, :, :]
        s2 = cat_logits[:, classes:classes * 2, :]

        c = cat_logits[:, -1, :, :].sigmoid().unsqueeze(dim=1)

        s1 = s1.unsqueeze(dim=2)
        s2 = s2.unsqueeze(dim=1)
        cs = s1 * s2

        c = c.unsqueeze(dim=2)
        w = torch.eye(classes, device=c.device).reshape(1, classes, classes, 1, 1)
        w = w * (1 - c) + (1 - w) * c
        cs *= w

        cs = torch.flatten(cs, 1, 2)

        cs = cs.argmax(dim=1)

        for i in range(classes):
            unchange_idx = i * classes + i
            cs = torch.where(cs == unchange_idx, torch.zeros_like(cs), cs)

        cs = torch.where(cs == 0, torch.zeros_like(cs), cs + 1)

        return cs

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                name='er.R18',
                pretrained=True,
            ),
            temporal_reduction=dict(
                reduce_type='conv'
            ),
            transformer=dict(
                inner_channels=96,
                num_heads=(3, 3, 3, 3),
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                change_threshold=0.5,
                min_keep_ratio=0.01,
                max_keep_ratio=0.1,
                train_max_keep=2000,
                num_blocks=(2, 2, 2, 2),
                disable_attn_refine=False,
                output_type='single_scale'
            ),
            semantic_decoder=dict(
                num_classes=6,
                transformer=dict(
                    inner_channels=96 * 2,
                    num_heads=(3,),
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.1,
                    num_blocks=(2,),
                    output_type='single_scale'
                ),
            ),
            main_loss=dict(
                bce=dict(),
                dice=dict(),
                mem=dict(),
                log_binary_iou_sigmoid=dict(),
                ignore_index=-1
            ),
            seg_loss=dict(
                prefix='seg_',
                ce=dict(),
                ignore_index=-1
            ),
            region_loss=[
                dict(
                    ignore_index=-1,
                    prefix='1'
                )
            ]
        ))

    def log_info(self):
        return {
            'encoder': self.backbone,
            'change_decoder': self.multi_stage_attn,
            'semantic_decoder': self.semantic_decoder
        }

    def custom_param_groups(self):
        if self.cfg.backbone.name.startswith('mit'):
            param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
            for n, p in self.named_parameters():
                if 'norm' in n:
                    param_groups[0]['params'].append(p)
                elif 'pos_block' in n:
                    param_groups[0]['params'].append(p)
                else:
                    param_groups[1]['params'].append(p)
            return param_groups
        else:
            return self.parameters()
