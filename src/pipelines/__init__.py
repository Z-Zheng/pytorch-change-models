# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Pipelines Package

This package contains pipeline implementations for various tasks using the models
in this repository.
"""

from .change_detection_pipeline import ChangeDetectionPipeline

__all__ = [
    "ChangeDetectionPipeline",
] 