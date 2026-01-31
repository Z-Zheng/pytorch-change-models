# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, asdict
from typing import Dict, Iterator, Optional
import torch


@dataclass
class ChangeDetectionModelOutput:
    """Container for change detection model outputs.

    This dataclass provides typed attributes and dictionary-style access for
    flexible downstream usage.

    Attributes:
        change_prediction: Required change mask/logits tensor.
        t1_semantic_prediction: Optional semantic prediction for time-1.
        t2_semantic_prediction: Optional semantic prediction for time-2.
    """
    change_prediction: torch.Tensor

    t1_semantic_prediction: Optional[torch.Tensor] = None
    t2_semantic_prediction: Optional[torch.Tensor] = None

    def __getitem__(self, key: str) -> torch.Tensor:
        """Dictionary-style access, e.g. ``output['change_prediction']``."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: torch.Tensor):
        """Dictionary-style assignment for existing fields only."""
        if not hasattr(self, key):
            raise KeyError(f"Field '{key}' is not a valid field of {self.__class__.__name__}")
        setattr(self, key, value)

    def keys(self):
        """Return keys for dict-like usage."""
        return self.__dict__.keys()

    def values(self):
        """Return values for dict-like usage."""
        return self.__dict__.values()

    def items(self) -> Iterator:
        """Return items for dict-like usage."""
        return self.__dict__.items()

    def __len__(self) -> int:
        return len(self.__dict__)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to a standard Python dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        """Custom representation that handles ``None`` tensors safely."""
        info = []
        for k, v in self.items():
            if v is None:
                info.append(f"{k}=None")
            elif isinstance(v, torch.Tensor):
                info.append(f"{k}=Tensor(shape={v.shape}, device={v.device})")
            else:
                info.append(f"{k}={v}")
        return f"{self.__class__.__name__}(\n    " + ",\n    ".join(info) + "\n)"

    def logit_to_prob_(self):
        def _ada_act(x):
            return x.sigmoid() if x.size(1) == 1 else x.softmax(dim=1)

        self.change_prediction = _ada_act(self.change_prediction)
        if self.t1_semantic_prediction is not None:
            self.t1_semantic_prediction = _ada_act(self.t1_semantic_prediction)
        if self.t2_semantic_prediction is not None:
            self.t2_semantic_prediction = _ada_act(self.t2_semantic_prediction)
        return self
