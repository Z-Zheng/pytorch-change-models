# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, asdict
from typing import Dict, Iterator, Optional
import torch


@dataclass
class ChangeDetectionModelOutput:
    """
    Base output class for the model.

    It behaves like a rigid dataclass (for type safety and IDE autocomplete)
    but also supports dictionary-like access patterns.
    Fields:
    - t1_semantic_prediction: Optional tensor (can be None).
    - t2_semantic_prediction: Optional tensor (can be None).
    - change_prediction: Mandatory tensor.
    """
    change_prediction: torch.Tensor

    t1_semantic_prediction: Optional[torch.Tensor] = None
    t2_semantic_prediction: Optional[torch.Tensor] = None

    def __getitem__(self, key: str) -> torch.Tensor:
        """
        Allows dictionary-style access: output['key']
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: torch.Tensor):
        """
        Allows dictionary-style assignment: output['key'] = value
        Only allows setting existing fields to ensure structural integrity.
        """
        if not hasattr(self, key):
            raise KeyError(f"Field '{key}' is not a valid field of {self.__class__.__name__}")
        setattr(self, key, value)

    def keys(self):
        """Enables dict(output).keys()"""
        return self.__dict__.keys()

    def values(self):
        """Enables dict(output).values()"""
        return self.__dict__.values()

    def items(self) -> Iterator:
        """Enables dict(output).items()"""
        return self.__dict__.items()

    def __len__(self) -> int:
        return len(self.__dict__)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Explicitly convert to a standard Python dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        """
        Custom print format.
        Safely handles cases where tensors are None.
        """
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


# --- Usage Example ---
if __name__ == "__main__":
    # 1. Simulate dummy tensors
    t1 = torch.rand(1, 256, 256)
    t2 = torch.rand(1, 256, 256)
    change = torch.rand(1, 256, 256)

    # 2. Initialize the object
    out = ChangeDetectionModelOutput(
        t1_semantic_prediction=t1,
        t2_semantic_prediction=t2,
        change_prediction=change
    )

    # 3. Attribute Access (Best for IDE autocomplete)
    print(f"Attribute Access: {out.t1_semantic_prediction.shape}")

    # 4. Dictionary Access (Best for dynamic loops or legacy code)
    print(f"Dict Access:      {out['change_prediction'].shape}")

    # 5. Iteration
    print("\nIterating over items:")
    for k, v in out.items():
        print(f" - {k}: {v.shape}")

    # 6. Clean Print
    print("\nLog Representation:")
    print(out)

    out = ChangeDetectionModelOutput(
        change_prediction=change
    )
    print(out)
