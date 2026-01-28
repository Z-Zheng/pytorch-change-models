# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torchvision.models import Swin_T_Weights
from configs import comm

config = dict(
    model=dict(
        type='mmChangeOS',
        params=dict(
            encoder=dict(
                type='TVSwinTransformer',
                params=dict(
                    name='swin_t',
                    weights=Swin_T_Weights.IMAGENET1K_V1
                ),
            ),
            decoder=dict(
                in_channels_list=[96 * (2 ** i) for i in range(4)],
                out_channels=256,
                fusion_type='2mlps'
            ),
        )
    ),
    data=comm.train_data,
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=6e-5,
            power=0.9,
            max_iters=40000,
        )
    ),
    optimizer=dict(
        type='adamw',
        params=dict(
            weight_decay=0.01
        ),
    ),
    train=dict(
        torch_compile=dict(),
        forward_times=1,
        num_iters=40000,
        distributed=True,
        sync_bn=True,
        log_interval_step=50,
        save_ckpt_interval_epoch=100000,
        callbacks=[
        ]
    ),
    test=dict()
)