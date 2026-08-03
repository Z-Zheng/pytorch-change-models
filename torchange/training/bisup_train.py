# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset-agnostic bitemporal-supervised training entry point.

Unlike ``bisup_train_bright.py``, no evaluator is hardcoded here. Declare them in
``config.train.callbacks`` instead; ``ever`` builds them from the ``CALLBACK`` registry:

    train=dict(
        callbacks=[
            dict(type='HFxView2StandardEval', params=dict(split='test', epoch_interval=5)),
        ],
    )
"""
import torch

import ever as er
from ever.trainer import get_default_parser

# importing the package populates er.registry DATASET/MODEL/CALLBACK from torchange/data/,
# torchange/module/ and torchange/metrics/, so configs can name any of them.
# torchange/models/ stays opt-in -- a config must import the algorithm file it uses.
import torchange  # noqa: F401

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    er.registry.register_all()

    parser = get_default_parser()
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--deterministic", action='store_true')
    trainer, args = er.trainer.get_trainer(parser=parser, return_args=True)
    er.seed_torch(args.seed, deterministic=args.deterministic)

    trainer: er.trainer.THDDPTrainer
    trainer.run()
