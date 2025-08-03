# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict

from .bcd import binary_change_detection_evaluate
from .second import semantic_change_detection_evaluate
from .xview2 import evaluate as xview2_evaluate

_EVAL_REGISTRY: Dict[str, Callable] = {
    'bcd': binary_change_detection_evaluate,
    'second': semantic_change_detection_evaluate,
    'xview2': xview2_evaluate,
}


def evaluate(task: str, model, dataloader, *args, **kwargs):
    """Unified evaluation interface across different change detection tasks.

    Parameters
    ----------
    task: str
        Name of the evaluation task. Options are ``'bcd'``, ``'second'``, and
        ``'xview2'``.
    model: torch.nn.Module
        Model to be evaluated.
    dataloader: Iterable
        Dataloader providing evaluation data.
    *args, **kwargs:
        Additional arguments forwarded to the underlying evaluation function.
    """
    if task not in _EVAL_REGISTRY:
        raise KeyError(f"Unknown evaluation task: {task}. Supported tasks: {list(_EVAL_REGISTRY.keys())}")
    return _EVAL_REGISTRY[task](model, dataloader, *args, **kwargs)

