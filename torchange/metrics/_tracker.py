# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Make evaluation callbacks survive a resume.

``er.metric.ScoreTracker`` is in-memory only and its ``to_csv`` rewrites the whole file, while
``ever`` rebuilds callbacks from scratch on every process start (``Trainer.build_callbacks``).
``resume_from_last`` therefore restores the model and global_step but *not* the evaluation history,
so the first evaluation after a restart would overwrite ``model-best.pth`` with a possibly worse
model and truncate the score CSV to a single row. Seeding the tracker from the CSV it previously
wrote fixes both.
"""
import os

import pandas as pd

__all__ = ['restore_tracker']


def restore_tracker(tracker, csv_path, max_step=None):
    """Repopulate ``tracker`` from a CSV it wrote earlier.

    Parameters
    ----------
    tracker : er.metric.ScoreTracker
        Tracker to seed, expected to be empty.
    csv_path : str
        Path the tracker previously wrote with ``to_csv``. Missing file is not an error.
    max_step : int, optional
        Drop rows recorded after this step. On resume this is the restored ``global_step``:
        rows beyond it describe weights that were discarded by rolling back to the checkpoint.

    Returns
    -------
    int
        Number of rows restored.
    """
    if not os.path.exists(csv_path):
        return 0

    df = pd.read_csv(csv_path)
    if 'step' not in df.columns or len(df) == 0:
        return 0

    if max_step is not None:
        df = df[df['step'] <= max_step]
    if len(df) == 0:
        return 0

    # Write through the public `scores` view rather than tracker.append(), which would re-emit
    # every restored point to wandb under a stale step.
    data = tracker.scores
    for col in df.columns:
        data.setdefault(col, []).extend(df[col].tolist())

    return len(df)
