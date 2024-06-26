# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ever as er
import albumentations as A
import albumentations.pytorch
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


def mixed_score(loc_tb, dam_tb):
    loc_f1 = loc_tb.f1(1)

    nodamage_f1 = dam_tb.f1(1)
    minor_f1 = dam_tb.f1(2)
    major_f1 = dam_tb.f1(3)
    destroyed_f1 = dam_tb.f1(4)

    # https://github.com/DIUx-xView/xView2_scoring/blob/ea0793da6f66a71236f2c4a34536d51beff483ab/xview2_metrics.py#L247

    harmonic_mean = lambda xs: len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    dam_f1 = harmonic_mean([nodamage_f1, minor_f1, major_f1, destroyed_f1])

    final_f1 = 0.3 * loc_f1 + 0.7 * dam_f1
    return loc_f1, dam_f1, final_f1, [nodamage_f1, minor_f1, major_f1, destroyed_f1]


def _accumulate_loc(op, gt, pr):
    gt = gt.numpy().ravel()
    gt = np.where(gt > 0, np.ones_like(gt), np.zeros_like(gt))
    op.forward(gt, pr)


def _accumulate_dam(op, gt, pr):
    IGNORE_INDEX = 255
    gt_dam = gt.numpy().ravel()
    # https://github.com/DIUx-xView/xView2_scoring/blob/ea0793da6f66a71236f2c4a34536d51beff483ab/xview2_metrics.py#L100
    valid_inds = np.where((gt_dam != IGNORE_INDEX) & (gt_dam != 0))[0]
    gt_dam = gt_dam[valid_inds]
    dam_pred = pr.cpu().numpy().ravel()[valid_inds]
    op.forward(gt_dam, dam_pred)


def parse_prediction_v1(pred):
    loc_pred = pred['t1_semantic_prediction'] > 0.5
    dam_pred = pred['change_prediction'].argmax(dim=1)
    return loc_pred, dam_pred


@torch.no_grad()
def evaluate(model, test_dataloader, logger, model_dir, split):
    dataloder = test_dataloader
    torch.cuda.empty_cache()
    model.eval()
    # ppe = ProcessPoolExecutor(max_workers=4)
    loc_metric_op = er.metric.PixelMetric(2, model_dir, logger=logger)
    damage_metric_op = er.metric.PixelMetric(5, model_dir,
                                             logger=logger)
    binary_dam_metric_op = er.metric.PixelMetric(2, model_dir, logger=logger)

    for x, y in tqdm(dataloder, disable=not er.dist.is_main_process()):
        x = x.to(er.auto_device())
        gt_loc, gt_dam = y['masks']
        pred = model(x)
        loc_pred, dam_pred = parse_prediction_v1(pred)

        # https://github.com/DIUx-xView/xView2_scoring/blob/ea0793da6f66a71236f2c4a34536d51beff483ab/xview2_metrics.py#L99
        dam_pred = loc_pred * dam_pred

        _accumulate_loc(loc_metric_op, gt_loc, loc_pred)

        _accumulate_dam(damage_metric_op, gt_dam, dam_pred)

    er.dist.synchronize()
    loc_tb = loc_metric_op.summary_all()
    dam_tb = damage_metric_op.summary_all()
    binary_dam_tb = binary_dam_metric_op.summary_all()

    loc_f1, dam_f1, final_f1, dam_f1s = mixed_score(loc_tb, dam_tb)

    logger.info(f'\nOverall F1, Localization F1, Damage F1\n{final_f1:.4f}, {loc_f1:.4f}, {dam_f1:.4f}')
    logger.info(f'dam f1 per class\n{dam_f1s[0]:.4f}, {dam_f1s[1]:.4f}, {dam_f1s[2]:.4f}, {dam_f1s[3]:.4f}')
    binary_dam_f1 = binary_dam_tb._rows[1][2]
    logger.info(f'binary dam F1 = {binary_dam_f1:.4f}')

    torch.cuda.empty_cache()

    return {
        f'{split}/loc_f1': loc_f1,
        f'{split}/dam_f1': dam_f1,
        f'{split}/final_f1': final_f1,
        f'{split}/non': dam_f1s[0],
        f'{split}/minor': dam_f1s[1],
        f'{split}/major': dam_f1s[2],
        f'{split}/destroyed': dam_f1s[3],
        f'{split}/binary_dam_f1': binary_dam_f1
    }


@er.registry.CALLBACK.register()
class xView2StandardEvalCallback(er.Callback):
    def __init__(self,
                 dataset_dir,
                 epoch_interval=10,
                 only_master=False,
                 prior=101,
                 after_train=True):
        super().__init__(epoch_interval=epoch_interval, only_master=only_master, prior=prior,
                         after_train=after_train)
        split = Path(dataset_dir).name
        assert split in ['test', 'hold']
        self.split = split
        dataloader = er.builder.make_dataloader(dict(
            type='xView2',
            params=dict(
                dataset_dir=dataset_dir,
                training=False,
                transforms=A.Compose([
                    A.Normalize(),
                    A.pytorch.ToTensorV2(),
                ]),
                batch_size=1,
                num_workers=2,
            ),
        ))
        self.dataloader = er.data.as_ddp_inference_loader(dataloader)

        self.tracked_scores = er.metric.ScoreTracker()
        self.best_final_f1 = 0.
        self.best_step = 0

    def func(self):
        self.launcher.logger.info(f'Split: {self.split}')
        model = self.launcher.er_model
        logger = self.launcher.logger
        model_dir = self.launcher.model_dir

        scores = evaluate(model, self.dataloader, logger, model_dir, self.split)
        self.tracked_scores.append(scores, self.launcher.checkpoint.global_step)

        if self.launcher._master:
            self.tracked_scores.to_csv(os.path.join(model_dir, f'{self.split}_tracked_scores.csv'))

            if scores[f'{self.split}/final_f1'] > self.best_final_f1:
                self.launcher.checkpoint.save('model-best.pth')
                self.best_final_f1 = scores[f'{self.split}/final_f1']
                self.best_step = self.launcher.checkpoint.global_step

            self.launcher.logger.info(f'best scores: {self.best_final_f1}, at step: {self.best_step}')
