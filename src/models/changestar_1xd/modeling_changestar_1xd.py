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


import ever as er
import ever.module as M
import ever.module.loss as L

from .configuration_changestar_1xd import ChangeStar1xdConfig

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


@torch.amp.autocast('cuda', dtype=torch.float32)
def sc_mse_loss(s1_logit, s2_logit, gt_masks):
    """Semantic consistency MSE loss."""
    c_gt = gt_masks[-1].to(torch.float32).unsqueeze(1)

    s1_p = s1_logit.log_softmax(dim=1).exp()
    s2_p = s2_logit.log_softmax(dim=1).exp()

    diff = (s1_p - s2_p) ** 2
    losses = (1 - c_gt) * diff + c_gt * (1 - diff)

    return losses.mean()


class ChangeMixinBiSupN1(nn.Module):
    """Change detection head with bidirectional supervision."""
    
    def __init__(self, in_channels, out_channels, temporal_symmetric=True,
                 num_semantic_classes=None, num_change_classes=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            M.LayerNorm2d(out_channels),
            nn.GELU()
        )
        if num_change_classes is None:
            num_change_classes = 1

        self.temporal_symmetric = temporal_symmetric
        self.change_conv = M.ConvUpsampling(out_channels, num_change_classes, scale_factor=4, kernel_size=1)
        self.num_semantic_classes = num_semantic_classes
        if isinstance(num_semantic_classes, int):
            self.semantic_conv = M.ConvUpsampling(out_channels, num_semantic_classes, scale_factor=4, kernel_size=1)
        elif isinstance(num_semantic_classes, (tuple, list)):
            self.semantic_conv = nn.ModuleList([
                M.ConvUpsampling(out_channels, nc, scale_factor=4, kernel_size=1)
                for nc in num_semantic_classes
            ])
        else:
            self.semantic_conv = nn.Identity()

    def forward(self, t1_feature, t2_feature):
        pre_logit = self.conv(torch.cat([t1_feature, t2_feature], dim=1))
        if self.temporal_symmetric:
            pre_logit = pre_logit + self.conv(torch.cat([t2_feature, t1_feature], dim=1))

        change_logit = self.change_conv(pre_logit)
        if isinstance(self.num_semantic_classes, int) or self.num_semantic_classes is None:
            t1_semantic_logit = self.semantic_conv(t1_feature)
            t2_semantic_logit = self.semantic_conv(t2_feature)
        else:
            t1_semantic_logit = self.semantic_conv[0](t1_feature)
            t2_semantic_logit = self.semantic_conv[1](t2_feature)

        if self.training:
            return {
                CHANGE: change_logit,
                T1SEM: t1_semantic_logit if self.num_semantic_classes else None,
                T2SEM: t2_semantic_logit if self.num_semantic_classes else None,
            }
        else:
            def _act(logit):
                if logit.size(1) > 1:
                    return logit.softmax(dim=1)
                else:
                    return logit.sigmoid()

            return {
                CHANGE: _act(change_logit),
                T1SEM: _act(t1_semantic_logit) if self.num_semantic_classes else None,
                T2SEM: _act(t2_semantic_logit) if self.num_semantic_classes else None,
            }


class ChangeStar1xdModel(PreTrainedModel):
    """
    ChangeStar1xd model for change detection.
    
    This model can be used for change detection tasks with optional semantic segmentation.
    """
    
    config_class = ChangeStar1xdConfig
    base_model_prefix = "changestar_1xd"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: ChangeStar1xdConfig):
        super().__init__(config)
        
        # Initialize encoder
        if hasattr(er.registry.MODEL, config.encoder_type):
            self.encoder = er.registry.MODEL[config.encoder_type](config.encoder_params)
        else:
            raise ValueError(f"Encoder type {config.encoder_type} not found in registry")
        
        # Initialize head
        head_in_channels = 2 * config.encoder_params.get("out_channels", config.out_channels)
        self.head = ChangeMixinBiSupN1(
            in_channels=head_in_channels,
            out_channels=config.out_channels,
            temporal_symmetric=config.temporal_symmetric,
            num_semantic_classes=config.num_semantic_classes,
            num_change_classes=config.num_change_classes
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
        
        if self.config.bitemporal_forward:
            bitemporal_features = bitemporal_forward(self.encoder, pixel_values)
        else:
            bitemporal_features = self.encoder(pixel_values)
        
        preds = self.head(*bitemporal_features)
        
        if self.training and labels is not None:
            loss_dict = self.compute_loss(preds, labels)
            if not return_dict:
                return loss_dict
            return {"loss": loss_dict}
        
        if not return_dict:
            return preds
        
        return {"logits": preds}
    
    def _semantic_loss(self, pred, target):
        """Compute semantic segmentation loss."""
        if pred.size(1) > 1:
            return {
                'ce_loss': F.cross_entropy(pred, target.to(torch.int64), reduction='mean', ignore_index=255),
                'dice_loss': L.dice_loss_with_logits(pred, target.to(torch.int64)),
            }
        else:
            target = target.to(torch.float32)
            return {
                'bce_loss': L.binary_cross_entropy_with_logits(
                    pred, target.reshape_as(pred), reduction='mean'),
                'dice_loss': L.dice_loss_with_logits(pred, target),
            }
    
    @torch.amp.autocast('cuda', dtype=torch.float32)
    def compute_loss(self, preds, labels):
        """Compute the total loss."""
        gt_change = labels['masks'][-1].to(torch.float32)
        change_logit = preds[CHANGE].to(torch.float32)
        
        loss_dict = dict()
        
        # Change detection loss
        if 'change' in self.config.loss_config:
            change_loss_config = self.config.loss_config['change']
            
            if ('bce' in change_loss_config) or ('ce' in change_loss_config):
                if change_logit.size(1) == 1:
                    ls = change_loss_config.get('bce', {}).get('ls', 0.0)
                    loss = L.label_smoothing_binary_cross_entropy(change_logit, gt_change, eps=ls)
                    loss_dict.update(c_bce_loss=loss)
                else:
                    ls = change_loss_config.get('ce', {}).get('ls', 0.0)
                    loss = F.cross_entropy(change_logit, gt_change.to(torch.int64), ignore_index=255, label_smoothing=ls)
                    loss_dict.update(c_ce_loss=loss)
            
            if 'dice' in change_loss_config:
                gamma = change_loss_config['dice'].get('gamma', 1.0)
                if change_logit.size(1) == 1:
                    loss_dict.update(
                        c_dice_loss=L.tversky_loss_with_logits(change_logit, gt_change, alpha=0.5, beta=0.5, gamma=gamma),
                    )
                else:
                    loss_dict.update(
                        c_dice_loss=L.dice_loss_with_logits(change_logit, gt_change),
                    )
        
        # Semantic segmentation losses
        if preds[T1SEM] is not None and 't1' in self.config.loss_config:
            t1_losses = self._semantic_loss(preds[T1SEM], labels['masks'][0])
            loss_dict.update({f"t1_{k}": v for k, v in t1_losses.items()})
        
        if preds[T2SEM] is not None and 't2' in self.config.loss_config:
            t2_losses = self._semantic_loss(preds[T2SEM], labels['masks'][1])
            loss_dict.update({f"t2_{k}": v for k, v in t2_losses.items()})
        
        # Semantic consistency loss
        if 'sc' in self.config.loss_config:
            loss_dict.update(dict(
                sc_mse_loss=sc_mse_loss(preds[T1SEM], preds[T2SEM], labels['masks'])
            ))
        
        return loss_dict


class ChangeStar1xdForChangeDetection(ChangeStar1xdModel):
    """
    ChangeStar1xd model for change detection with a specific head for change detection.
    """
    
    def __init__(self, config: ChangeStar1xdConfig):
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