# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ChangeStar 1xd model implementation for change detection.

This module implements the ChangeStar architecture with 1x decoder for bitemporal change
detection and semantic segmentation. The model supports both bitemporal supervised learning
and single-temporal supervised learning (STAR) approaches.

Key components:
    - ChangeStar1xd: Main model class combining encoder and change detection head
    - ChangeMixinBiSupN1: Change detection head with semantic segmentation support
    - UniBitemporalSupervisedLoss: Unified loss computation for bitemporal training
    - bitemporal_forward: Helper for processing bitemporal inputs through single-temporal encoders

References:
    ChangeStar architecture: https://arxiv.org/abs/2108.07002
    Bitemporal supervised learning: https://www.sciencedirect.com/science/article/abs/pii/S0924271621002835
"""
import enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import ever as er
import ever.module as M
import ever.module.loss as L
from torchange.utils.outputs import ChangeDetectionModelOutput
from torchange.utils.mask_data import Mask

from einops import rearrange


class TrainMode(str, enum.Enum):
    """Enum for different training modes."""
    STAR = 'star'
    BSL = 'bsl'


def bitemporal_forward(module: nn.Module, x: torch.Tensor) -> Tuple[
    Union[List[torch.Tensor], torch.Tensor], Union[List[torch.Tensor], torch.Tensor]]:
    """Forward bitemporal images through a single-temporal encoder.

    This function processes bitemporal images (stacked as t=2 in channel dimension) through
    a module designed for single-temporal inputs by treating each temporal image separately,
    then splits the outputs back into t1 and t2 features.

    Args:
        module: Encoder module that processes single-temporal images
        x: Bitemporal input tensor of shape (B, 2*C, H, W) where first C channels are t1 
           and last C channels are t2

    Returns:
        Tuple containing:
            - t1_features: Features from t1 images, either List[Tensor] or Tensor
            - t2_features: Features from t2 images, either List[Tensor] or Tensor

        If module returns multiple feature maps (e.g., pyramid features), each element
        in the lists corresponds to features at different scales.
    """
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
def sc_mse_loss(s1_logit: torch.Tensor, s2_logit: torch.Tensor, change_mask: torch.Tensor) -> torch.Tensor:
    """Compute semantic consistency MSE loss between t1 and t2 predictions.
    Ref: Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images

    This loss encourages semantic predictions to be consistent (similar) in unchanged regions
    and different in changed regions. It computes the squared difference between probability
    distributions and weights them by the change mask.

    Args:
        s1_logit: Semantic logits for t1 image of shape (B, num_classes, H, W)
        s2_logit: Semantic logits for t2 image of shape (B, num_classes, H, W)
        change_mask: Binary change mask of shape (B, H, W) where 1 indicates change

    Returns:
        Scalar tensor containing the mean semantic consistency loss
    """
    c_gt = change_mask.to(torch.float32).unsqueeze(1)

    s1_p = s1_logit.log_softmax(dim=1).exp()
    s2_p = s2_logit.log_softmax(dim=1).exp()

    diff = (s1_p - s2_p) ** 2
    losses = (1 - c_gt) * diff + c_gt * (1 - diff)

    return losses.mean()


def all_gather_tensor(data, group=None):
    if er.dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = er.dist._get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(output, data, group=group)
    return output


class UniBitemporalSupervisedLoss:
    """Unified bitemporal supervised loss computation for change detection models.

    This class provides loss computation methods for models trained with bitemporal supervision,
    where both change labels and semantic labels for t1 and t2 are available. It supports
    various loss functions including cross-entropy, binary cross-entropy, Dice loss, and
    Tversky loss for both semantic segmentation and change detection tasks.
    """

    def _semantic_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            loss_cfg: er.config.AttrDict,
    ) -> Dict[str, torch.Tensor]:
        """Compute semantic segmentation loss for a single temporal image.

        Args:
            pred: Predicted semantic logits of shape (B, num_classes, H, W) for multi-class
                  or (B, 1, H, W) for binary classification
            target: Ground truth semantic mask of shape (B, H, W)
            loss_cfg: Loss configuration containing settings for different loss terms

        Returns:
            Dictionary mapping loss names to scalar loss tensors. May contain:
                - 'ce_loss': Cross-entropy loss (for multi-class)
                - 'bce_loss': Binary cross-entropy loss (for binary)
                - 'dice_loss' or 'tver_loss': Tversky/Dice loss
        """
        ignore_index = loss_cfg.to_dict().get('ignore_index', 255)

        loss_dict = {}
        if pred.size(1) > 1:
            target = target.to(torch.int64)
            loss_dict['ce_loss'] = F.cross_entropy(pred, target.to(torch.int64), reduction='mean', ignore_index=ignore_index)
        else:
            target = target.to(torch.float32)
            loss_dict['bce_loss'] = L.binary_cross_entropy_with_logits(
                pred, target.reshape_as(pred), reduction='mean',
                ignore_index=ignore_index
            )

        if 'tver' in loss_cfg:
            tver = loss_cfg.tver.to_dict()  # make torch.compile happy
            alpha = tver.get('alpha', 0.5)
            gamma = tver.get('gamma', 1.0)
            loss_dict['tver_loss'] = L.tversky_loss_with_logits(
                pred, target, alpha=alpha, gamma=gamma, ignore_index=ignore_index
            )
        else:
            alpha = 0.5
            gamma = 1.0
            loss_dict['dice_loss'] = L.tversky_loss_with_logits(
                pred, target, alpha=alpha, gamma=gamma, ignore_index=ignore_index
            )
        return loss_dict

    def _change_loss(
            self,
            change_logit: torch.Tensor,
            gt_change: torch.Tensor,
            loss_cfg: er.config.AttrDict,
    ) -> Dict[str, torch.Tensor]:
        """Compute change detection loss.

        Args:
            change_logit: Predicted change logits of shape (B, num_change_classes, H, W)
                         or (B, 1, H, W) for binary change detection
            gt_change: Ground truth change mask of shape (B, H, W)
            loss_cfg: Loss configuration specifying which loss terms to compute

        Returns:
            Dictionary mapping loss names to scalar loss tensors. May contain:
                - 'c_bce_loss' or 'c_ce_loss': Binary/multi-class cross-entropy
                - 'c_dice_loss': Dice loss
                - 'c_tver_loss': Tversky loss
        """
        loss_dict = {}
        is_binary = change_logit.size(1) == 1
        ignore_index = loss_cfg.to_dict().get('ignore_index', 255)

        if ('bce' in loss_cfg) or ('ce' in loss_cfg):
            if is_binary:
                ls = loss_cfg.bce.to_dict().get('ls', 0.0)
                loss_dict['c_bce_loss'] = L.label_smoothing_binary_cross_entropy(
                    change_logit, gt_change, eps=ls,
                    ignore_index=ignore_index
                )
            else:
                ls = loss_cfg.ce.to_dict().get('ls', 0.0)
                loss_dict['c_ce_loss'] = F.cross_entropy(
                    change_logit, gt_change.to(torch.int64),
                    ignore_index=ignore_index,
                    label_smoothing=ls
                )

        if 'dice' in loss_cfg:
            dice = loss_cfg.dice.to_dict()  # make torch.compile happy
            gamma = dice.get('gamma', 1.0)
            loss_dict['c_dice_loss'] = L.tversky_loss_with_logits(change_logit, gt_change, alpha=0.5, gamma=gamma,
                                                                  ignore_index=ignore_index)

        if 'tver' in loss_cfg:
            tver = loss_cfg.tver.to_dict()  # make torch.compile happy
            alpha = tver.get('alpha', 0.5)
            gamma = tver.get('gamma', 1.0)
            loss_dict['c_tver_loss'] = L.tversky_loss_with_logits(change_logit, gt_change, alpha=alpha, gamma=gamma,
                                                                  ignore_index=ignore_index)

        return loss_dict

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def loss(
            self,
            preds: ChangeDetectionModelOutput,
            y: Dict[str, Any],
            loss_cfg: er.config.AttrDict
    ) -> Dict[str, torch.Tensor]:
        """Compute bitemporal supervised learning (BSL) loss.

        This method computes the total loss for bitemporal supervised training, combining
        change detection loss, semantic segmentation losses for both temporal images,
        and optional semantic consistency loss.

        Args:
            preds: Model predictions containing change and semantic predictions
            y: Ground truth dictionary containing 'masks' with change and semantic labels
            loss_cfg: Configuration for loss computation with terms like 'change', 't1', 't2', 'sc'

        Returns:
            Dictionary mapping loss term names to scalar loss tensors. Always contains
            change loss terms, and optionally contains semantic loss terms for t1/t2
            and semantic consistency loss.

        Note:
            The masks in y can have different formats:
                - [t1_semantic, t2_semantic, change]: Full supervision
                - [t1_semantic, change]: Only t1 semantic available
                - [change]: Only change labels available
        """
        # masks[0] - cls, masks[1] - cls, masks[2] - change
        # masks[0] - cls, masks[1] - change
        # masks[0] - change
        assert hasattr(loss_cfg, 'change'), 'loss must contain change term'

        y_masks = y['masks']
        if not isinstance(y_masks, Mask):
            y_masks = Mask.from_list(y_masks)

        gt_change = y_masks.change_mask.to(torch.float32)
        change_logit = preds.change_prediction.to(torch.float32)

        loss_dict = {}
        loss_dict |= self._change_loss(change_logit, gt_change, loss_cfg.change)

        for t in ['t1', 't2']:
            pred = getattr(preds, f'{t}_semantic_prediction')
            mask = getattr(y_masks, f'{t}_semantic_mask')

            if pred is not None and t in loss_cfg:
                s_loss_cfg = getattr(loss_cfg, t)
                losses = self._semantic_loss(pred, mask, s_loss_cfg)
                loss_dict |= {f"{t}_{k}": v for k, v in losses.items()}

        if 'sc' in loss_cfg:
            loss_dict['sc_mse_loss'] = sc_mse_loss(
                preds.t1_semantic_prediction,
                preds.t2_semantic_prediction,
                y_masks.change_mask
            )

        return loss_dict


class GeneralizedSTAR:
    """Generalized single-temporal supervised loss for change representation learning.

    This class implements the G-STAR approach, where models are trained using only single-temporal semantic labels without
    explicit change labels.

    Note:
        See the STAR and G-STAR paper
        (https://arxiv.org/abs/2108.07002)
        (https://link.springer.com/article/10.1007/s11263-024-02141-4)
        for details on single-temporal supervised learning.
    """

    def apply_gstar(self, x: torch.Tensor, y: Optional[Dict[str, Any]]):
        x, y = self.generate_pseudo_bitemporal_image_pair(
            images=x,
            images_view=y['images_view'],
            masks=y['masks'],
            masks_view=y['masks_view'],
        )
        return x, y

    def generate_pseudo_bitemporal_image_pair(
            self,
            images: torch.Tensor,
            images_view: torch.Tensor,
            masks: torch.Tensor,
            masks_view: torch.Tensor,
            shuffle_prob: float = 0.5
    ):
        """
        Generates pseudo-bitemporal image pairs by pairing local images with
        randomly selected views from the global batch (across all GPUs).
        Unifies single-GPU and multi-GPU (DDP) logic efficiently.
        """
        device = images.device
        bs = images.size(0)

        # Get distributed context
        rank = er.dist.get_rank()
        world_size = er.dist.get_world_size()

        # 1. Prepare Global Candidate Pools
        # We only need to gather the 'view' data (candidates).
        # The source 'images' (x1) remain local to save memory.
        if world_size > 1:
            # Gather all views from all ranks into a single tensor
            global_images_view = torch.cat(all_gather_tensor(images_view), dim=0)
            global_masks_view = torch.cat(all_gather_tensor(masks_view), dim=0)
            total_N = bs * world_size
        else:
            # Fallback for single GPU: global pool is just the local batch
            global_images_view = images_view
            global_masks_view = masks_view
            total_N = bs

        # 2. Generate Global Indices
        # Determine which image pairs with which based on probability.
        # We generate indices for the entire global batch size (total_N).
        shuffle_inds = torch.randperm(total_N, device=device)
        regular_inds = torch.arange(total_N, device=device)
        prob = torch.rand(total_N, device=device)

        # Create the final index mapping
        final_global_inds = torch.where(prob < shuffle_prob, shuffle_inds, regular_inds)

        # 3. Synchronize Indices (DDP only)
        # Ensure all GPUs use the exact same shuffle logic.
        if world_size > 1:
            dist.broadcast(final_global_inds, src=0)

            # 4. Slice Indices for Local Rank
            # Current rank 'k' owns the x1 images in range [k*bs : (k+1)*bs].
            # We only need the target indices corresponding to this range.
            start_idx = rank * bs
            end_idx = (rank + 1) * bs
            local_inds = final_global_inds[start_idx:end_idx]
        else:
            local_inds = final_global_inds

        # 5. Construct Pairs
        # x1 is always the local batch (no gather needed)
        x1 = images
        # x2 is fetched from the global pool using the sliced indices
        pseudo_x2 = global_images_view[local_inds]
        pseudo_mask2 = global_masks_view[local_inds]

        # 6. Concatenate and Create Mask Object
        x = torch.cat([x1, pseudo_x2], dim=1)

        # Compute change masks (float32 for consistency)
        cmasks = (masks != pseudo_mask2).to(torch.float32)

        y = Mask.from_list([masks, pseudo_mask2, cmasks])

        return x, y


@er.registry.MODEL.register(verbose=False)
class ChangeStar1xd(er.ERModule, UniBitemporalSupervisedLoss, GeneralizedSTAR):
    """ChangeStar model with 1x decoder for bitemporal change detection.

    This model implements the ChangeStar architecture that processes bitemporal images through
    an encoder (either bitemporal-specific or single-temporal with bitemporal forward) and
    produces change detection and optional semantic segmentation predictions via a unified head.

    The architecture consists of:
        - Encoder: Extracts features from bitemporal images
        - Head (ChangeMixinBiSupN1): Fuses temporal features and predicts change/semantics

    Training supports:
        - Bitemporal supervised learning (change + semantic labels)
        - Optional semantic consistency regularization
        - Flexible loss configurations

    Attributes:
        encoder: Feature extraction backbone (e.g., Swin, DINOv3, ResNet-based FarSeg)
        head: Change detection head that fuses temporal features

    Example:
        >>> config = dict(
        ...     encoder=dict(type='SwinFarSeg',
        ...            params=dict(
        ...            name='swin_t',
        ...            weights=None,
        ...            out_channels=256),
        ...            bitemporal_forward=True
        ...     ),
        ...     head=dict(num_semantic_classes=7, num_change_classes=1),
        ...     loss=dict(change=dict(bce=dict(), dice=dict())),
        ...     train_mode=TrainMode.BSL,
        ... )
        >>> model = ChangeStar1xd(config)
        >>> model.eval()
        >>> x = torch.randn(2, 6, 256, 256)  # Bitemporal input (2*3 channels)
        >>> output = model(x)  # Returns ChangeDetectionModelOutput
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        self.encoder = er.builder.make_model(self.config.encoder)

        self.cfg.head.in_channels = 2 * self.encoder.config.out_channels
        self.cfg.head.out_channels = self.encoder.config.out_channels

        self.head = ChangeMixinBiSupN1(**self.cfg.head)

    def forward(self, x: torch.Tensor, y: Optional[Dict[str, Any]] = None) -> Union[
        Dict[str, torch.Tensor], ChangeDetectionModelOutput]:
        """Forward pass of ChangeStar model.

        Args:
            x: Bitemporal input tensor of shape (B, 2*C, H, W) where first C channels
               represent t1 image and last C channels represent t2 image
            y: Optional ground truth dictionary for training, containing 'masks' with
               change and semantic labels

        Returns:
            During training: Dictionary of loss terms and their values
            During inference: ChangeDetectionModelOutput containing change and semantic
                            predictions as probability maps

        Note:
            The encoder can process bitemporal inputs in two ways:
                1. bitemporal_forward=True: Uses bitemporal_forward() to process each
                   temporal image separately through a single-temporal encoder
                2. bitemporal_forward=False: Encoder directly handles bitemporal inputs
        """
        if self.training and self.cfg.train_mode == TrainMode.STAR:
            x, y = self.apply_gstar(x, y)

        if self.cfg.encoder.bitemporal_forward:
            bitemporal_features = bitemporal_forward(self.encoder, x)
        else:
            bitemporal_features = self.encoder(x)

        preds = self.head(*bitemporal_features)

        if self.training:
            return self.loss(preds, y, self.cfg.loss)

        return preds

    def set_default_config(self) -> None:
        self.config.update(dict(
            encoder=dict(type=None, params=dict(), bitemporal_forward=False),
            head=dict(
                in_channels=-1,
                out_channels=-1,
                temporal_symmetric=True,
                num_semantic_classes=None,
                num_change_classes=None
            ),
            loss=dict(),
            train_mode=TrainMode.BSL,
        ))

    def log_info(self) -> Dict[str, Any]:
        return dict(
            encoder=self.encoder,
            head=self.head
        )

    def custom_param_groups(self) -> List[Dict[str, Any]]:
        param_groups = []

        if isinstance(self.encoder, er.ERModule):
            param_groups += self.encoder.custom_param_groups()
        else:
            param_groups += [{'params': self.encoder.parameters()}]

        if isinstance(self.head, er.ERModule):
            param_groups += self.head.custom_param_groups()
        else:
            param_groups += [{'params': self.head.parameters()}]

        return param_groups


class ChangeMixinBiSupN1(nn.Module):
    """Change detection head with semantic segmentation support (Bitemporal Supervision N=1).

    This head module fuses bitemporal features to predict change detection masks and optional
    semantic segmentation masks. It uses a simple concatenation-based fusion strategy with
    optional temporal symmetry for more robust predictions.

    Architecture:
        1. Concatenate t1 and t2 features
        2. Apply conv-norm-activation
        3. Optionally apply temporal symmetric fusion (concat in reverse order and average)
        4. Predict change mask via upsampling convolution
        5. Predict semantic masks for t1 and t2 separately

    Args:
        in_channels: Number of input channels (2 * encoder feature channels)
        out_channels: Number of output channels for the fusion layer
        temporal_symmetric: If True, applies symmetric fusion by averaging forward and
                          backward temporal concatenation (helps with temporal consistency)
        num_semantic_classes: Number of semantic classes. Can be:
            - int: Same number of classes for both t1 and t2
            - tuple/list of two ints: Different classes for t1 and t2
            - None: No semantic segmentation
        num_change_classes: Number of change classes (default: 1 for binary change)

    Example:
        >>> head = ChangeMixinBiSupN1(
        ...     in_channels=512, out_channels=256,
        ...     temporal_symmetric=True,
        ...     num_semantic_classes=7, num_change_classes=1
        ... )
        >>> t1_feat = torch.randn(2, 256, 128, 128)
        >>> t2_feat = torch.randn(2, 256, 128, 128)
        >>> output = head(t1_feat, t2_feat)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temporal_symmetric: bool = True,
            num_semantic_classes: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
            num_change_classes: Optional[int] = None
    ) -> None:
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

    def forward(self, t1_feature: torch.Tensor, t2_feature: torch.Tensor) -> ChangeDetectionModelOutput:
        """Forward pass to generate change and semantic predictions.

        Args:
            t1_feature: Feature map from t1 image of shape (B, C, H, W)
            t2_feature: Feature map from t2 image of shape (B, C, H, W)

        Returns:
            ChangeDetectionModelOutput containing:
                - change_prediction: Change logits/probabilities of shape (B, num_change_classes, H*4, W*4)
                - t1_semantic_prediction: T1 semantic logits/probabilities (if num_semantic_classes is set)
                - t2_semantic_prediction: T2 semantic logits/probabilities (if num_semantic_classes is set)

            During training, returns logits. During inference, returns probabilities.

        Note:
            All predictions are upsampled 4x to match input resolution. If temporal_symmetric
            is True, the change prediction benefits from bidirectional temporal fusion.
        """
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

        output = ChangeDetectionModelOutput(
            change_prediction=change_logit,
            t1_semantic_prediction=t1_semantic_logit if self.num_semantic_classes else None,
            t2_semantic_prediction=t2_semantic_logit if self.num_semantic_classes else None,
        )

        if self.training:
            return output
        else:
            return output.logit_to_prob_()
