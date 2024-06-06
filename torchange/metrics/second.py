# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
import torch
import ever as er
from tqdm import tqdm
import os

palette = np.array([
    [255, 255, 255],
    [0, 0, 255],  # water
    [128, 128, 128],  # ground
    [0, 128, 0],  # low veg
    [0, 255, 0],  # tree
    [128, 0, 0],  # building
    [255, 0, 0],  # playground
], dtype=np.uint8)

land_use = ['W', 'G', 'L', 'T', 'B', 'P']
change_types = ['unchanged'] + [''] * 36

for i in range(6):
    for j in range(6):
        change_types[i * 6 + j + 1] = f'{land_use[i]}2{land_use[j]}'


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def score_summary(hist):
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3 * IoU_mean + 0.7 * Sek

    return {
        'SECOND/kappa': kappa_n0,
        'SECOND/mIoU': IoU_mean,
        'SECOND/Sek': Sek,
        'SECOND/Score': Score,
        'SECOND/IoU_1': IoU_fg
    }


@er.registry.CALLBACK.register()
class SemanticChangeDetectionEval(er.Callback):
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
        second_score = self.evaluate_sek()
        self.info(second_score)

        score = self.evaluate_mIoU()
        self.info(score)
        score.update(second_score)

        best_key = 'eval/mIoU_scd'
        best_score = self.score_tracker.highest_score(best_key)
        if score[best_key] > best_score[best_key]:
            self.save_model('model-best.pth')

        self.score_tracker.append(score, self.global_step)

        self.score_tracker.to_csv(os.path.join(self.model_dir, f'{self.score_table_name}_scores.csv'))

        best_score = self.score_tracker.highest_score(best_key)
        self.launcher.logger.info(f"best mIoU_scd: {best_score[best_key]}, at step {best_score['step']}")

    @torch.no_grad()
    def evaluate_sek(self):
        self.model.eval()
        num_class = 37
        hist = np.zeros((num_class, num_class))

        for img, gt in tqdm(self.dataloader, disable=not er.dist.is_main_process()):
            img = img.to(er.auto_device())
            predictions = self.model(img)
            CLASS = predictions['t1_semantic_prediction'].size(1)
            s1 = predictions['t1_semantic_prediction'].argmax(dim=1)
            s2 = predictions['t2_semantic_prediction'].argmax(dim=1)
            c = predictions['change_prediction'] > 0.5

            pr_sc = torch.where(c, s1 * CLASS + s2 + 1, torch.zeros_like(s1))

            gt_s1 = gt['masks'][0].to(torch.int64)
            gt_s2 = gt['masks'][1].to(torch.int64)
            gt_sc = torch.where(gt['masks'][-1] > 0, gt_s1 * CLASS + gt_s2 + 1,
                                torch.zeros_like(gt['masks'][0]))

            hist += get_hist(pr_sc.cpu().numpy(), gt_sc.cpu().numpy(), num_class)

        return score_summary(hist)

    @torch.no_grad()
    def evaluate_mIoU(self):
        self.model.eval()
        bcd = er.metric.PixelMetric(2, self.model_dir, logger=self.logger)
        scd = er.metric.PixelMetric(6 * 6 + 1, self.model_dir, logger=self.logger, class_names=change_types)
        class_freq = torch.zeros([6 * 6 + 1, ], dtype=torch.int64)

        for img, gt in tqdm(self.dataloader, disable=not er.dist.is_main_process()):
            img = img.to(er.auto_device())
            predictions = self.model(img)
            CLASS = predictions['t1_semantic_prediction'].size(1)

            s1 = predictions['t1_semantic_prediction'].argmax(dim=1)
            s2 = predictions['t2_semantic_prediction'].argmax(dim=1)
            c = predictions['change_prediction'] > 0.5

            pr_sc = torch.where(c, s1 * CLASS + s2 + 1, torch.zeros_like(s1))

            gt_s1 = gt['masks'][0].to(torch.int64)
            gt_s2 = gt['masks'][1].to(torch.int64)
            gt_sc = torch.where(gt['masks'][-1] > 0,
                                gt_s1 * CLASS + gt_s2 + 1,
                                torch.zeros_like(gt['masks'][0]))

            bcd.forward(gt['masks'][-1], c)
            scd.forward(gt_sc, pr_sc)

            idx, cnt = torch.unique(gt_sc, return_counts=True)
            class_freq.scatter_add_(dim=0, index=idx.to(torch.int64), src=cnt)

        valid_cls_indices = class_freq.nonzero(as_tuple=True)[0].numpy()
        er.info(f'effective number of change types: {valid_cls_indices.shape[0]}')

        er.dist.synchronize()
        bcd_results = bcd.summary_all()
        scd_results = scd.summary_all()

        ious = []
        for i in valid_cls_indices:
            ious.append(scd_results.iou(int(i)))
        mIoU = sum(ious) / len(ious)

        return {
            'eval/mIoU_scd': mIoU,
            'eval/IoU_bcd': bcd_results.iou(1),
            'eval/f1_bcd': bcd_results.f1(1),
            'eval/prec_bcd': bcd_results.precision(1),
            'eval/rec_bcd': bcd_results.recall(1),
        }
