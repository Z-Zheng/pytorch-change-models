# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ChangeStar2.5: Universal Remote Sensing Change Detection Architecture from
https://link.springer.com/article/10.1007/s11263-024-02141-4

Upgrade the version of ChangeStar2 with consistent ConvNeXt-based feature refinement blocks.

This module implements the ChangeStar2_5 architecture for bitemporal remote sensing
change detection. The model combines dense feature extraction with temporal comparison
mechanisms to detect changes between two time points while optionally performing
semantic segmentation on each temporal image.

Key Features:
- Bitemporal change detection with configurable temporal symmetry
- Optional semantic segmentation for T1 and T2 images
- ConvNeXt-based feature refinement blocks
- Flexible loss computation supporting both binary and multi-class scenarios
- Integration with HuggingFace Hub for model sharing

Architecture Overview:
1. Image Dense Encoder: Extracts dense features from input images (both T1 and T2)
2. ChangeMixin2_5: Performs temporal comparison and generates predictions
   - Concatenation path: Processes concatenated T1 and T2 features
   - Difference path: Processes absolute difference between features
   - Optional temporal symmetry: Processes both (T1,T2) and (T2,T1) orders
3. Prediction Heads: Separate heads for semantic and change predictions

References:
- ChangeStar: https://arxiv.org/abs/2108.07002
- ChangeStar2: https://link.springer.com/article/10.1007/s11263-024-02141-4

"""

import ever as er
import ever.module as M
import torch
import torch.nn as nn
from timm.layers import DropPath
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from torchange.models.changestar_1xd import UniBitemporalSupervisedLoss
from torchange.models.changestar_1xd import GeneralizedSTAR, TrainMode
from torchange.utils.outputs import ChangeDetectionModelOutput


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = M.LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ChangeMixin2_5(nn.Module):
    """
    Change detection mixin module that performs temporal feature comparison.

    This module takes features from two temporal images and performs change detection
    through multiple pathways: concatenation, difference, and optional symmetric processing.
    It can optionally perform semantic segmentation on each temporal image.

    Args:
        dim (int): Number of input feature channels
        change_classes (int): Number of change detection classes. Default: 1 (binary)
        semantic_classes (int): Number of semantic segmentation classes. 
            Set to -1 to disable semantic segmentation. Default: -1
        temporal_symmetric (bool): Whether to apply temporal symmetry by processing
            both (T1,T2) and (T2,T1) orders. Default: True
        t1_on (bool): Whether to enable semantic segmentation for T1 image. Default: True
        t2_on (bool): Whether to enable semantic segmentation for T2 image. Default: True
        n_blocks (int): Number of ConvNeXt blocks for feature refinement. Default: 0

    Attributes:
        linear_cat (nn.Sequential): Feature processing path for concatenated features
        linear_diff (nn.Sequential): Feature processing path for difference features
        change_conv (nn.Conv2d): Convolution head for change prediction
        semantic_conv (nn.Conv2d or nn.Identity): Convolution head for semantic prediction
    """

    def __init__(
            self,
            dim,
            change_classes=1,
            semantic_classes=-1,
            temporal_symmetric=True,
            t1_on=True,
            t2_on=True,
            n_blocks=0,
    ):
        super().__init__()
        self.t1_on = t1_on
        self.t2_on = t2_on
        self.change_classes = change_classes
        self.semantic_classes = semantic_classes
        self.temporal_symmetric = temporal_symmetric
        self.linear_cat = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1, bias=False),
            M.LayerNorm2d(dim),
            nn.GELU(),
            nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)]) if n_blocks > 0 else nn.Identity()
        )
        self.linear_diff = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            M.LayerNorm2d(dim),
            nn.GELU(),
            nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)]) if n_blocks > 0 else nn.Identity()
        )

        self.change_conv = nn.Conv2d(dim, change_classes, 1)
        if semantic_classes > 0:
            self.semantic_conv = nn.Conv2d(dim, semantic_classes, 1)
        else:
            self.semantic_conv = nn.Identity()

    def forward(self, t1_feature, t2_feature):
        """
        Forward pass for temporal feature comparison and prediction.

        Processes features through three pathways:
        1. Concatenation: cat([T1, T2]) -> linear_cat
        2. Symmetric concatenation (optional): cat([T2, T1]) -> linear_cat
        3. Difference: |T1 - T2| -> linear_diff

        Args:
            t1_feature (torch.Tensor): Features from T1 image, shape (B, C, H, W)
            t2_feature (torch.Tensor): Features from T2 image, shape (B, C, H, W)

        Returns:
            tuple: (s1_logit, s2_logit, c_logit)
                - s1_logit (torch.Tensor or None): Semantic logits for T1, shape (B, S, H, W)
                - s2_logit (torch.Tensor or None): Semantic logits for T2, shape (B, S, H, W)
                - c_logit (torch.Tensor): Change logits, shape (B, C, H, W)
        """
        bifeature = self.linear_cat(torch.cat([t1_feature, t2_feature], dim=1))
        if self.temporal_symmetric:
            bifeature = bifeature + self.linear_cat(torch.cat([t2_feature, t1_feature], dim=1))
        bifeature += self.linear_diff((t1_feature - t2_feature).abs())

        c_logit = self.change_conv(bifeature)

        s1_logit = self.semantic_conv(t1_feature) if self.t1_on else None
        s2_logit = self.semantic_conv(t2_feature) if self.t2_on else None

        return s1_logit, s2_logit, c_logit

    def custom_param_groups(self):
        """
        Create custom parameter groups for optimization.

        Separates parameters into two groups:
        1. Normalization layers (no weight decay)
        2. Other parameters (with weight decay)

        Returns:
            list[dict]: List of parameter groups with different weight decay settings
        """
        param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
        for i, p in self.named_parameters():
            if 'norm' in i:
                param_groups[0]['params'].append(p)
            else:
                param_groups[1]['params'].append(p)
        return param_groups


up2x_op = nn.UpsamplingBilinear2d(scale_factor=2)
up4x_op = nn.UpsamplingBilinear2d(scale_factor=4)


def up2x(x):
    """
    Upsample input by 2x using bilinear interpolation.

    Args:
        x (torch.Tensor or None): Input tensor to upsample

    Returns:
        torch.Tensor or None: Upsampled tensor or None if input is None
    """
    if x is not None:
        return up2x_op(x)
    else:
        return None


def up4x(x):
    """
    Upsample input by 4x using bilinear interpolation.

    Args:
        x (torch.Tensor or None): Input tensor to upsample

    Returns:
        torch.Tensor or None: Upsampled tensor or None if input is None
    """
    if x is not None:
        return up4x_op(x)
    else:
        return None


@er.registry.MODEL.register(verbose=False)
class ChangeStar2_5(er.ERModule, PyTorchModelHubMixin, UniBitemporalSupervisedLoss, GeneralizedSTAR):
    """
    ChangeStar2_5: Universal change detection model with dense feature encoding and temporal comparison.
    
    This model implements a complete change detection pipeline that processes bitemporal images 
    through a shared dense encoder and performs change detection with optional semantic segmentation.
    It supports multiple encoder backbones (Swin, DINOv3, etc.) and flexible loss configurations.
    
    Architecture Flow:
        1. Input: Stacked bitemporal images with shape (B, 2*C, H, W)
           - First C channels: T1 (time point 1)
           - Last C channels: T2 (time point 2)
    
        2. Feature Extraction:
           - Rearrange to (2B, C, H, W) for batch processing
           - Process through shared image_dense_encoder
           - Output features at 1/4 resolution: (2B, D, H/4, W/4)
    
        3. Temporal Comparison:
           - Split features back to T1 and T2: each (B, D, H/4, W/4)
           - Apply ChangeMixin2_5 for temporal feature comparison
           - Generate change and optional semantic predictions
    
        4. Prediction Upsampling:
           - Upsample predictions to original input resolution
           - Apply appropriate upsampling scale (2x or 4x)
    
        5. Training/Inference:
           - Training: Compute multi-task losses (semantic + change)
           - Inference: Generate probability maps for predictions
    
    Args:
        cfg (Config): Configuration object containing model specifications:
        
            image_dense_encoder (dict): Dense feature encoder configuration
                - type (str): Encoder architecture name (e.g., 'SwinFarSeg', 'DINOv3FarSeg')
                - params (dict): Encoder-specific parameters
                    - out_channels (int): Number of output feature channels
                    - Additional encoder-specific parameters
        
            mixin (dict): Temporal comparison module configuration
                - c (int): Number of change detection classes
                    - 1 for binary change detection
                    - >1 for multi-class change classification
                - s (int): Number of semantic segmentation classes
                    - -1 to disable semantic segmentation
                    - 1 for binary semantic segmentation
                    - >1 for multi-class semantic segmentation
                - temporal_symmetric (bool): Enable temporal symmetry
                    - True: Process both (T1,T2) and (T2,T1) orders
                    - False: Only process (T1,T2) order
                - t1_on (bool): Enable semantic segmentation for T1 image
                - t2_on (bool): Enable semantic segmentation for T2 image
                - n_blocks (int, optional): Number of ConvNeXt refinement blocks. Default: 0
                - upsample_scale (int): Upsampling factor for final predictions
                    - 2: For encoders with 1/2 output stride
                    - 4: For encoders with 1/4 output stride (most common)
        
            loss (dict): Loss function configuration for each prediction task
                - t1 (dict, optional): T1 semantic segmentation loss config
                    - For binary: BCE + Tversky loss weights
                    - For multi-class: CE + Dice loss weights
                - t2 (dict, optional): T2 semantic segmentation loss config
                    - Similar to s1 configuration
                - change (dict): Change detection loss config
                    - For binary: BCE + Tversky loss weights
                    - For multi-class: CE + Dice loss weights
        
            train_mode (str, optional): Training mode selection
                - 'BSL': Bitemporal Supervised Learning (default)
                - 'STAR': Single-Temporal Supervised Learning
    
    Attributes:
        image_dense_encoder (nn.Module): Shared encoder for extracting dense features 
            from both temporal images
        mixin (ChangeMixin2_5): Temporal comparison module that performs feature 
            comparison and generates change/semantic predictions
    
    Example:
        >>> config = dict(
        ...     image_dense_encoder=dict(
        ...         type='SwinFarSeg',
        ...         params=dict(
        ...             name='swin_t',
        ...             weights=None,
        ...             out_channels=256
        ...         )
        ...     ),
        ...     mixin=dict(
        ...         c=1, s=7,
        ...         temporal_symmetric=True,
        ...         t1_on=True, t2_on=True,
        ...         upsample_scale=4
        ...     ),
        ...     loss=dict(change=dict(bce=dict(), dice=dict())),
        ...     train_mode=TrainMode.BSL,
        ... )
        >>> model = ChangeStar2_5(config)
        >>> model.eval()
        >>> x = torch.randn(2, 6, 256, 256)  # Bitemporal input (2*3 channels)
        >>> output = model(x)  # Returns ChangeDetectionModelOutput

    References:
        - ChangeStar2_5 architecture: https://link.springer.com/article/10.1007/s11263-024-02141-4
        - Original ChangeStar: https://arxiv.org/abs/2108.07002
    
    See Also:
        - ChangeMixin2_5: Temporal comparison module
        - UniBitemporalSupervisedLoss: Unified loss computation
        - GeneralizedSTAR: STAR training algorithm support
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.image_dense_encoder = er.builder.make_model(self.cfg.image_dense_encoder)
        dim = self.cfg.image_dense_encoder.params.out_channels
        self.mixin = ChangeMixin2_5(
            dim=dim,
            change_classes=self.cfg.mixin.c,
            semantic_classes=self.cfg.mixin.s,
            temporal_symmetric=self.cfg.mixin.temporal_symmetric,
            t1_on=self.cfg.mixin.t1_on,
            t2_on=self.cfg.mixin.t2_on,
        )

    def forward(self, x, y=None):
        """
        Forward pass of the ChangeStar2_5 model.

        Args:
            x (torch.Tensor): Stacked bitemporal images, shape (B, 2*C, H, W)
                where the first C channels are T1 and the last C channels are T2
            y (dict, optional): Ground truth labels for training, containing:
                - masks (list[torch.Tensor]): [t1_mask, t2_mask, change_mask]
                    Each mask has shape (B, H, W)

        Returns:
            dict: During training, returns loss dictionary with computed losses.
                  During inference, returns prediction dictionary with:
                  - t1_semantic_prediction (torch.Tensor, optional): T1 semantic predictions
                  - t2_semantic_prediction (torch.Tensor, optional): T2 semantic predictions
                  - change_prediction (torch.Tensor): Change predictions
        """
        if self.training and self.cfg.train_mode == TrainMode.STAR:
            x, y = self.apply_gstar(x, y)

        x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
        embed = self.image_dense_encoder(x)  # os 4
        t1_embed, t2_embed = rearrange(embed, '(b t) c h w -> t b c h w', t=2)
        s1_logit, s2_logit, c_logit = self.mixin(t1_embed, t2_embed)
        if self.cfg.mixin.upsample_scale == 4:
            upsample_op = up4x
        elif self.cfg.mixin.upsample_scale == 2:
            upsample_op = up2x
        else:
            raise ValueError(f'Invalid upsample scale: {self.cfg.mixin.upsample_scale}')

        s1_logit = upsample_op(s1_logit)
        s2_logit = upsample_op(s2_logit)
        c_logit = upsample_op(c_logit)

        output = ChangeDetectionModelOutput(
            t1_semantic_prediction=s1_logit, t2_semantic_prediction=s2_logit, change_prediction=c_logit
        )
        if self.training:
            return self.train_loss(output, y)
        else:
            return self.predict(output)

    def predict(self, output: ChangeDetectionModelOutput):
        output.logit_to_prob_()
        return output

    def train_loss(self, output: ChangeDetectionModelOutput, y):
        """
        Compute training losses for semantic and change predictions.

        Supports both binary and multi-class scenarios:
        - Binary (1 channel): Uses BCE + Tversky loss
        - Multi-class (>1 channel): Uses CE + Dice loss

        Loss components:
        - T1 semantic loss (optional): BCE/CE + Tversky/Dice
        - T2 semantic loss (optional): CE + Dice
        - Change loss: BCE/CE + Tversky/Dice

        Args:
            output (ChangeDetectionModelOutput): Model output object containing:
                - t1_semantic_prediction (torch.Tensor or None): T1 temporal semantic segmentation logits, shape (B, S, H, W)
                - t2_semantic_prediction (torch.Tensor or None): T2 temporal semantic segmentation logits, shape (B, S, H, W)
                - change_prediction (torch.Tensor): Change detection logits, shape (B, C, H, W)
            y (dict): Ground truth label dictionary containing:
                - masks (list[torch.Tensor]): [t1_mask, t2_mask, change_mask], each mask has shape (B, H, W)
     

        Returns:
            dict: Loss dictionary with computed loss values and memory usage:
                - mem (torch.Tensor): GPU memory usage in MB
                - t1_bce_loss/t1_ce_loss (torch.Tensor, optional): T1 classification loss
                - t1_tver_loss/t1_dice_loss (torch.Tensor, optional): T1 region loss
                - t2_ce_loss (torch.Tensor, optional): T2 classification loss
                - t2_dice_loss (torch.Tensor, optional): T2 region loss
                - c_bce_loss/c_ce_loss (torch.Tensor): Change classification loss
                - c_dice_loss (torch.Tensor, optional): Change region loss
        """
        loss_dict = {'max_mem': torch.as_tensor(torch.cuda.max_memory_allocated() / 1024 / 1024, dtype=torch.int32)}
        loss_dict |= self.loss(output, y, self.cfg.loss)

        return loss_dict

    def set_default_config(self):
        """
        Set default configuration for the ChangeStar2_5 model.

        Default configuration includes:
        - image_dense_encoder: None (must be specified by user)
        - mixin:
            - s: -1 (semantic segmentation disabled)
            - c: 1 (binary change detection)
            - temporal_symmetric: True (process both T1->T2 and T2->T1)
            - t1_on: True (enable T1 semantic prediction)
            - t2_on: True (enable T2 semantic prediction)
            - n_blocks: 0 (no ConvNeXt refinement blocks)
            - upsample_scale: 4 (4x upsampling)
        - loss: Empty dictionaries for t1, t2, and change loss configurations
        """
        self.config.update(dict(
            image_dense_encoder=None,
            mixin=dict(s=-1, c=1, temporal_symmetric=True, t1_on=True, t2_on=True, n_blocks=0, upsample_scale=4),
            loss=dict(),
            train_mode=TrainMode.BSL,
        ))

    def custom_param_groups(self):
        """
        Create custom parameter groups for the complete model.

        Collects parameter groups from both the encoder and mixin modules,
        allowing for different optimization strategies (e.g., learning rates,
        weight decay) for different components.

        Returns:
            list[dict]: List of parameter groups from encoder and mixin,
                        each potentially with custom optimization settings
        """
        param_groups = []

        if isinstance(self.image_dense_encoder, er.ERModule):
            param_groups += self.image_dense_encoder.custom_param_groups()
        else:
            param_groups += [{'params': self.image_dense_encoder.parameters()}]

        if isinstance(self.mixin, er.ERModule):
            param_groups += self.mixin.custom_param_groups()
        else:
            param_groups += [{'params': self.mixin.parameters()}]

        return param_groups
