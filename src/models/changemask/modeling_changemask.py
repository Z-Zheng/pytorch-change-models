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

from .configuration_changemask import ChangeMaskConfig

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
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


@torch.amp.autocast('cuda', dtype=torch.float32)
def sc_mse_loss(s1_logit, s2_logit, gt_masks):
    """Semantic consistency MSE loss."""
    c_gt = gt_masks[-1].to(torch.float32).unsqueeze(1)

    s1_p = s1_logit.log_softmax(dim=1).exp()
    s2_p = s2_logit.log_softmax(dim=1).exp()

    diff = (s1_p - s2_p) ** 2
    losses = (1 - c_gt) * diff + c_gt * (1 - diff)

    return losses.mean()


class Squeeze(nn.Module):
    """Squeeze operation for specific dimension."""
    
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(dim=self.dim)


class SpatioTemporalInteraction(nn.Sequential):
    """Spatio-temporal interaction module."""
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 type='conv3d'):
        if type == 'conv3d':
            padding = dilation * (kernel_size - 1) // 2
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, [2, kernel_size, kernel_size], stride=1,
                          dilation=(1, dilation, dilation),
                          padding=(0, padding, padding),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        elif type == 'conv1plus2d':
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, (2, 1, 1), stride=1,
                          padding=(0, 0, 0),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                          kernel_size // 2) if kernel_size > 1 else nn.Identity(),
                nn.BatchNorm2d(out_channels) if kernel_size > 1 else nn.Identity(),
                nn.ReLU(True) if kernel_size > 1 else nn.Identity(),
            )


class TemporalSymmetricTransformer(nn.Module):
    """Temporal symmetric transformer for change detection."""
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 interaction_type='conv3d',
                 symmetric_fusion='add'):
        super(TemporalSymmetricTransformer, self).__init__()

        if isinstance(in_channels, list) or isinstance(in_channels, tuple):
            self.t = nn.ModuleList([
                SpatioTemporalInteraction(inc, outc, kernel_size, dilation=dilation, type=interaction_type)
                for inc, outc in zip(in_channels, out_channels)
            ])
        else:
            self.t = SpatioTemporalInteraction(in_channels, out_channels, kernel_size, dilation=dilation,
                                               type=interaction_type)

        if symmetric_fusion == 'add':
            self.symmetric_fusion = lambda x, y: x + y
        elif symmetric_fusion == 'mul':
            self.symmetric_fusion = lambda x, y: x * y
        elif symmetric_fusion == None:
            self.symmetric_fusion = None

    def forward(self, features1, features2):
        if isinstance(features1, list):
            d12_features = [op(torch.stack([f1, f2], dim=2)) for op, f1, f2 in
                            zip(self.t, features1, features2)]
            if self.symmetric_fusion:
                d21_features = [op(torch.stack([f2, f1], dim=2)) for op, f1, f2 in
                                zip(self.t, features1, features2)]
                change_features = [self.symmetric_fusion(d12, d21) for d12, d21 in zip(d12_features, d21_features)]
            else:
                change_features = d12_features
        else:
            if self.symmetric_fusion:
                change_features = self.symmetric_fusion(self.t(torch.stack([features1, features2], dim=2)),
                                                        self.t(torch.stack([features2, features1], dim=2)))
            else:
                change_features = self.t(torch.stack([features1, features2], dim=2))
            change_features = change_features.squeeze(dim=2)
        return change_features


class ChangeMaskModel(PreTrainedModel):
    """
    ChangeMask model for change detection with semantic segmentation.
    
    This model uses a U-Net style architecture with temporal interaction
    for simultaneous change detection and semantic segmentation.
    """
    
    config_class = ChangeMaskConfig
    base_model_prefix = "changemask"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: ChangeMaskConfig):
        super().__init__(config)
        
        # Initialize encoder
        self.encoder = smp.encoders.get_encoder(
            config.encoder_type, 
            weights=config.encoder_weights
        )
        out_channels = self.encoder.out_channels
        
        # Initialize decoders
        self.semantic_decoder = UnetDecoder(
            encoder_channels=out_channels,
            decoder_channels=config.decoder_channels,
        )

        self.change_decoder = UnetDecoder(
            encoder_channels=out_channels,
            decoder_channels=config.decoder_channels,
        )

        # Initialize temporal transformer
        self.temporal_transformer = TemporalSymmetricTransformer(
            out_channels, out_channels,
            config.temporal_kernel_size, 
            interaction_type=config.temporal_interaction_type, 
            symmetric_fusion=config.temporal_symmetric_fusion,
        )
        
        # Initialize output heads
        self.semantic_head = nn.Conv2d(config.decoder_channels[-1], config.num_semantic_classes, 1)
        self.change_head = nn.Conv2d(config.decoder_channels[-1], config.num_change_classes, 1)
        
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
        t1_features, t2_features = bitemporal_forward(self.encoder, pixel_values)

        # Semantic segmentation predictions
        s1_logit = self.semantic_head(self.semantic_decoder(*t1_features))
        s2_logit = self.semantic_head(self.semantic_decoder(*t2_features))

        # Change detection predictions
        temporal_features = self.temporal_transformer(t1_features, t2_features)
        c_logit = self.change_head(self.change_decoder(*temporal_features))

        # Prepare outputs
        preds = {
            T1SEM: s1_logit,
            T2SEM: s2_logit,
            CHANGE: c_logit,
        }
        
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
        gt_masks = labels['masks']
        s1_gt = gt_masks[0].to(torch.int64)
        s2_gt = gt_masks[1].to(torch.int64)
        c_gt = gt_masks[-1].to(torch.float32)
        
        loss_dict = dict()
        
        # Semantic segmentation losses
        if 't1' in self.config.loss_config:
            t1_losses = self._semantic_loss(preds[T1SEM], s1_gt)
            loss_dict.update({f"s1_{k}": v for k, v in t1_losses.items()})
        
        if 't2' in self.config.loss_config:
            t2_losses = self._semantic_loss(preds[T2SEM], s2_gt)
            loss_dict.update({f"s2_{k}": v for k, v in t2_losses.items()})
        
        # Change detection loss
        if 'change' in self.config.loss_config:
            change_loss_config = self.config.loss_config['change']
            
            if ('bce' in change_loss_config) or ('ce' in change_loss_config):
                if preds[CHANGE].size(1) == 1:
                    ls = change_loss_config.get('bce', {}).get('ls', 0.0)
                    loss = L.label_smoothing_binary_cross_entropy(preds[CHANGE], c_gt, eps=ls)
                    loss_dict.update(c_bce_loss=loss)
                else:
                    ls = change_loss_config.get('ce', {}).get('ls', 0.0)
                    loss = F.cross_entropy(preds[CHANGE], c_gt.to(torch.int64), ignore_index=255, label_smoothing=ls)
                    loss_dict.update(c_ce_loss=loss)
            
            if 'dice' in change_loss_config:
                gamma = change_loss_config['dice'].get('gamma', 1.0)
                if preds[CHANGE].size(1) == 1:
                    loss_dict.update(
                        c_dice_loss=L.tversky_loss_with_logits(preds[CHANGE], c_gt, alpha=0.5, beta=0.5, gamma=gamma),
                    )
                else:
                    loss_dict.update(
                        c_dice_loss=L.dice_loss_with_logits(preds[CHANGE], c_gt),
                    )
        
        # Semantic consistency loss
        if 'sc' in self.config.loss_config:
            loss_dict.update(dict(
                sc_mse_loss=sc_mse_loss(preds[T1SEM], preds[T2SEM], gt_masks)
            ))
        
        return loss_dict


class ChangeMaskForChangeDetection(ChangeMaskModel):
    """
    ChangeMask model for change detection with a specific head for change detection.
    """
    
    def __init__(self, config: ChangeMaskConfig):
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