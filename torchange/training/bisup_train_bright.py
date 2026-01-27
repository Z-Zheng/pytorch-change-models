# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

import ever as er
from ever.trainer import get_default_parser
from torchange.metrics.bright import BRIGHTEval

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    er.registry.register_all()

    parser = get_default_parser()
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--eval_epoch_interval", type=int, default=1)
    trainer, args = er.trainer.get_trainer(parser=parser, return_args=True)
    er.seed_torch(args.seed, deterministic=args.deterministic)

    trainer: er.trainer.THDDPTrainer
    evaluator = BRIGHTEval(
        epoch_interval=args.eval_epoch_interval,
        splits=['test', ]
    )
    trainer.register_callback(evaluator)
    trainer.run()
