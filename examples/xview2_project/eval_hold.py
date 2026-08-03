# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Final evaluation of a trained model_dir on the xView2 `hold` split.

Deliberately NOT a training callback. `_xView2StandardEval.func()` writes `model-best.pth`
whenever *its own* split improves, so registering a second callback for `hold` would overwrite
the checkpoint that the `test` split selected. The protocol is:

    train on train+tier3  ->  select on `test` (model-best.pth)  ->  report on `hold`

Must be launched with torchrun even for a single GPU: `er.builder.make_dataloader` constructs a
StepDistributedSampler, which needs an initialized process group.

    cd examples/xview2_project
    torchrun --nnodes=1 --nproc_per_node=2 --master_port $RANDOM eval_hold.py \
      --model_dir logs/xview2_swint_cos

Writes `<split>_scores.csv` into model_dir.
"""
import argparse
import importlib
import logging
import os

import albumentations as A
import albumentations.pytorch
import ever as er
import torch
from ever.core.logger import get_console_file_logger
from ever.data import as_ddp_inference_loader

import torchange  # noqa: F401 -- registers HFxView2 and the metrics
from torchange.metrics.xview2 import evaluate


def build_dataloader(split, num_workers=2):
    # batch_size must stay 1: the single-map constraint in metrics/xview2.py multiplies a
    # (B,1,H,W) localization mask by a (B,H,W) damage map, which only broadcasts at B=1.
    dataloader = er.builder.make_dataloader(dict(
        type='HFxView2',
        params=dict(
            hf_repo_name='EVER-Z/torchange_xView2',
            splits=[split],
            training=False,
            transform=A.Compose([
                A.Normalize(),
                A.pytorch.ToTensorV2(),
            ]),
            batch_size=1,
            num_workers=num_workers,
        ),
    ))
    return as_ddp_inference_loader(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True,
                        help='training output dir; must contain config.pkl and a checkpoint')
    parser.add_argument('--checkpoint_name', default=None,
                        help='default: model-best.pth if present, else the latest checkpoint-*.pth')
    parser.add_argument('--split', default='hold', choices=['hold', 'test'])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--model_module', default='torchange.models.changeos',
                        help='imported to register the model named in config.pkl. Needed because '
                             'torchange/models/ is not auto-imported and, unlike the config .py, '
                             'the pickled config does not re-run that import.')
    args = parser.parse_args()

    assert 'LOCAL_RANK' in os.environ, 'launch with torchrun (use --nproc_per_node=1 for one GPU)'
    torch.set_float32_matmul_precision('high')
    er.registry.register_all()
    importlib.import_module(args.model_module)
    er.dist.init_dist_env()

    # checkpoint_name=None -> ever prefers model-best.pth, i.e. exactly the `test`-selected one
    model, tag = er.infer_tool.build_from_model_dir(
        model_dir=args.model_dir, checkpoint_name=args.checkpoint_name
    )
    model = model.to(er.auto_device()).eval()

    logger = get_console_file_logger(f'eval_{args.split}', logging.INFO, args.model_dir)
    logger.info(f'model_dir       : {args.model_dir}')
    logger.info(f'checkpoint      : {args.checkpoint_name or tag}')
    logger.info(f'evaluation split: {args.split}')

    dataloader = build_dataloader(args.split, num_workers=args.num_workers)
    scores = evaluate(model, dataloader, logger, args.model_dir, args.split)

    if er.dist.is_main_process():
        tracker = er.metric.ScoreTracker()
        tracker.append(scores, 0)
        out = os.path.join(args.model_dir, f'{args.split}_scores.csv')
        tracker.to_csv(out)
        for k, v in scores.items():
            logger.info(f'{k}: {v:.4f}')
        logger.info(f'wrote {out}')


if __name__ == '__main__':
    main()
