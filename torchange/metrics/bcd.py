# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path

import gc
import torch
import ever as er
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def binary_change_detection_evaluate(model, dataloader, log_dir=None, logger=None, class_names=None):
    model.eval()
    pm = er.metric.PixelMetric(2, log_dir, logger=logger, class_names=class_names)

    for img, gt in tqdm(dataloader, disable=not er.dist.is_main_process()):
        img = img.to(er.auto_device())
        predictions = model(img)

        pr_change = (predictions['change_prediction'] > 0.5).cpu()
        pr_change = pr_change.numpy().astype(np.uint8)
        gt_change = gt['masks'][-1]
        gt_change = gt_change.numpy()
        y_true = gt_change.ravel()
        y_pred = pr_change.ravel()

        y_true = np.where(y_true > 0, np.ones_like(y_true), np.zeros_like(y_true))

        pm.forward(y_true, y_pred)

    results = pm.summary_all()

    torch.cuda.empty_cache()
    gc.collect()

    return {
        'eval/iou': results.iou(1),
        'eval/f1': results.f1(1),
        'eval/prec': results.precision(1),
        'eval/rec': results.recall(1),
    }


@er.registry.CALLBACK.register()
class BinaryChangeDetectionPixelEval(er.Callback):
    def __init__(self, data_cfg, epoch_interval, prior=101):
        super().__init__(
            epoch_interval=epoch_interval,
            only_master=False,
            prior=prior,
            before_train=False,
            after_train=True,
        )
        dataloader = er.builder.make_dataloader(data_cfg)
        self.dataloader = er.data.as_ddp_inference_loader(dataloader)
        self.score_tracker = er.metric.ScoreTracker()

        self.score_table_name = data_cfg.type

    def func(self):
        score = binary_change_detection_evaluate(self.unwrapped_model, self.dataloader, self.model_dir, self.logger)

        best_score = self.score_tracker.highest_score('eval/f1')
        if score['eval/f1'] > best_score['eval/f1']:
            self.save_model('model-best.pth')

        self.score_tracker.append(score, self.global_step)
        self.score_tracker.to_csv(os.path.join(self.model_dir, f'{self.score_table_name}_scores.csv'))

        best_score = self.score_tracker.highest_score('eval/f1')
        self.logger.info(f"best F1: {best_score['eval/f1']}, at step {best_score['step']}")
