# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ever as er
import gc
import os
from typing import Dict
import torch
from tqdm import tqdm

__all__ = [
    'MultiClassPixelEval',
]


@er.registry.CALLBACK.register(verbose=False)
class MultiClassPixelEval(er.Callback):
    def __init__(self, data_cfg: Dict, num_classes, epoch_interval, prior=101, class_names=None):
        super().__init__(
            epoch_interval=epoch_interval,
            only_master=False,
            prior=prior,
            before_train=False,
            after_train=True,
        )
        self.num_classes = num_classes
        self.class_names = class_names
        dataloader = er.builder.make_dataloader(data_cfg)
        self.dataloader = er.data.as_ddp_inference_loader(dataloader)
        self.score_tracker = er.metric.ScoreTracker()
        self.score_table_name = data_cfg['type']

    @property
    def best_key(self):
        return 'eval/mIoU'

    def func(self):
        result = self.evaluate()
        mIoU = result.iou(-3)

        best_score = self.score_tracker.highest_score(self.best_key)
        if mIoU > best_score[self.best_key]:
            self.save_model('model-best.pth')

        score = self.extract_score(result)
        self.score_tracker.append(score, self.global_step)
        self.score_tracker.to_csv(os.path.join(self.model_dir, f'{self.score_table_name}_scores.csv'))

        best_score = self.score_tracker.highest_score(self.best_key)
        self.launcher.logger.info(f"best {self.best_key}: {best_score[self.best_key]}, at step {best_score['step']}")

    @torch.no_grad()
    def evaluate(self) -> er.metric.pixel.AccTable:
        self.model.eval()
        pm = er.metric.PixelMetric(self.num_classes, self.model_dir, logger=self.logger, class_names=self.class_names)

        for img, gt in tqdm(self.dataloader, disable=not er.dist.is_main_process()):
            img = img.to(er.auto_device())
            predictions = self.model(img)

            pr_change = predictions['change_prediction'].argmax(dim=1).cpu()
            pr_change = pr_change.numpy()
            gt_change = gt['masks'][-1].numpy()

            y_true = gt_change.ravel()
            y_pred = pr_change.ravel()

            pm.forward(y_true, y_pred)

        results = pm.summary_all()

        torch.cuda.empty_cache()
        gc.collect()
        return results

    def extract_score(self, result: er.metric.pixel.AccTable):
        return {
            'eval/mIoU': result.iou(-3),
        }
