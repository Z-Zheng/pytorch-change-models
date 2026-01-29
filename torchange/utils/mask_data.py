# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Iterator, Optional, List
import torch


@dataclass
class Mask:
    """Container for change detection masks.

    Attributes:
        change_mask: Required change mask tensor.
        t1_semantic_mask: Optional semantic mask for time-1.
        t2_semantic_mask: Optional semantic mask for time-2.
    """
    change_mask: torch.Tensor

    t1_semantic_mask: Optional[torch.Tensor] = None
    t2_semantic_mask: Optional[torch.Tensor] = None

    @classmethod
    def from_list(cls, mask_list: List[torch.Tensor]):
        """Create a ``Mask`` from a list of tensors.

        Expected layouts:
        - [change]
        - [t1_semantic, change]
        - [t1_semantic, t2_semantic, change]
        """
        if len(mask_list) == 1:
            return cls(change_mask=mask_list[-1])
        if len(mask_list) == 2:
            return cls(t1_semantic_mask=mask_list[0], change_mask=mask_list[-1])
        if len(mask_list) == 3:
            return cls(
                t1_semantic_mask=mask_list[0],
                t2_semantic_mask=mask_list[1],
                change_mask=mask_list[-1],
            )
        raise ValueError(f"Invalid number of masks: {len(mask_list)}")

    def items(self) -> Iterator:
        """Return items for dict-like usage."""
        return self.__dict__.items()

    def __repr__(self) -> str:
        info = []
        for k, v in self.items():
            if v is None:
                info.append(f"{k}=None")
            elif isinstance(v, torch.Tensor):
                info.append(f"{k}=Tensor(shape={v.shape}, device={v.device})")
            else:
                info.append(f"{k}={v}")
        return f"{self.__class__.__name__}(\n    " + ",\n    ".join(info) + "\n)"
