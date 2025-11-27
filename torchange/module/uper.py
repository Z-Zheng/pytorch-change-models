# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ever.module import LayerNorm2d


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in UPerNet.

    Args:
        in_channels (int): Number of input channels of the top feature map.
        out_channels (int): Number of output channels for each pooled branch.
        pool_scales (tuple[int]): Adaptive pooling scales.
    """

    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.pool_scales = pool_scales
        self.ppm_convs = nn.ModuleList()

        # Convolution on each pooled feature
        for scale in pool_scales:
            self.ppm_convs.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    LayerNorm2d(out_channels),
                    nn.GELU(),
                )
            )

        # Bottleneck to fuse concatenated PPM results
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        """Forward pass of PPM."""
        ppm_outs = [x]
        for ppm in self.ppm_convs:
            pooled = ppm(x)
            up = F.interpolate(
                pooled,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            ppm_outs.append(up)

        x = torch.cat(ppm_outs, dim=1)
        x = self.bottleneck(x)
        return x


class UPerHead(nn.Module):
    """
    UPerHead module for semantic segmentation (UPerNet decoder head).

    Args:
        in_channels (list[int]): Channels of input backbone feature maps
                                 (high → low resolution), e.g., [64, 128, 320, 512].
        channels (int): Unified channel width after lateral/PPM/FPN convs.
        num_classes (int): Number of segmentation categories.
        pool_scales (tuple[int]): PPM pooling scales.
        dropout_ratio (float): Dropout ratio before classifier.
        align_corners (bool): Align corners for bilinear interpolation.
    """

    def __init__(
            self,
            in_channels,
            channels,
            num_classes,
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            align_corners=False,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners

        # PPM on the last (lowest resolution) feature
        self.ppm = PyramidPoolingModule(self.in_channels[-1], channels, pool_scales)

        # Lateral 1×1 convs and 3×3 FPN convs (except the last feature which uses PPM)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for ch in self.in_channels[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, channels, kernel_size=1, bias=False),
                    LayerNorm2d(channels),
                    nn.GELU(),
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    LayerNorm2d(channels),
                    nn.GELU(),
                )
            )

        # FPN conv for the PPM output branch
        self.fpn_convs.append(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(channels),
                nn.GELU(),
            )
        )

        # Fuse all FPN outputs after upsampling
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * channels, channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): Feature maps from backbone, ordered
                high → low resolution. Shapes like:
                [(N, C1, H, W), (N, C2, H/2, W/2), (N, C3, H/4, W/4), (N, C4, H/8, W/8)]
        Returns:
            Tensor: Segmentation logits (N, num_classes, H, W)
        """
        assert len(inputs) == len(self.in_channels)

        # Apply PPM on the last feature
        psp_out = self.ppm(inputs[-1])

        # Lateral features (FPN inputs)
        laterals = []
        for i, conv in enumerate(self.lateral_convs):
            laterals.append(conv(inputs[i]))  # other layers
        laterals.append(psp_out)  # last layer uses PPM output

        # Top-down FPN fusion
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply 3×3 conv on each FPN branch
        fpn_outs = []
        for i, conv in enumerate(self.fpn_convs):
            fpn_outs.append(conv(laterals[i]))

        # Upsample all branches to the highest resolution and concatenate
        ref_size = fpn_outs[0].shape[2:]
        for i, feat in enumerate(fpn_outs):
            if feat.shape[2:] != ref_size:
                fpn_outs[i] = F.interpolate(
                    feat,
                    size=ref_size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )

        fused = torch.cat(fpn_outs, dim=1)
        fused = self.fuse_conv(fused)

        if self.dropout is not None:
            fused = self.dropout(fused)

        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    # Sanity test
    model = UPerHead(
        in_channels=[64, 128, 320, 512],
        channels=128,
        num_classes=19,
    )

    x = [
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 320, 32, 32),
        torch.randn(2, 512, 16, 16),
    ]

    out = model(x)
    print(out.shape)  # expected: (2, 19, 128, 128)
