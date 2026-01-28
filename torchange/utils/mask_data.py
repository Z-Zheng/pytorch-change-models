# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Iterator, Optional, List
import torch


@dataclass
class Mask:
    change_mask: torch.Tensor

    t1_semantic_mask: Optional[torch.Tensor] = None
    t2_semantic_mask: Optional[torch.Tensor] = None

    @classmethod
    def from_list(cls, mask_list: List[torch.Tensor]):
        # masks[0] - cls, masks[1] - cls, masks[2] - change
        # masks[0] - cls, masks[1] - change
        # masks[0] - change
        if len(mask_list) == 1:
            return cls(change_mask=mask_list[-1])
        elif len(mask_list) == 2:
            return cls(t1_semantic_mask=mask_list[0], change_mask=mask_list[-1])
        elif len(mask_list) == 3:
            return cls(t1_semantic_mask=mask_list[0], t2_semantic_mask=mask_list[1], change_mask=mask_list[-1])
        else:
            raise ValueError(f"Invalid number of masks: {len(mask_list)}")

    def items(self) -> Iterator:
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


if __name__ == '__main__':
    m = Mask.from_list([torch.rand(1, 256, 256), torch.rand(1, 256, 256), torch.rand(1, 256, 256)])
    print(m)
