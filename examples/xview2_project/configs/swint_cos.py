# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torchvision.models import Swin_T_Weights

# torchange/models/ is not auto-imported, so importing the model here is what registers
# ChangeOS/ChangeOSDecoder/ChangeOSHead. import_config() executes this file as a module.
from torchange.models.changeos import ChangeOS  # noqa: F401
from configs import comm

# ChangeOS.set_default_config() is already xView2-shaped: loc_head.num_classes=1 (building
# footprint) and dam_head.num_classes=5 (bg / no-damage / minor / major / destroyed), both at
# upsample_scale=4. matching the stride-4 decoder output. Only the decoder needs the Swin widths.
config = dict(
    model=dict(
        type='ChangeOS',
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
            max_iters=60000,
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
        num_iters=60000,
        distributed=True,
        sync_bn=True,
        log_interval_step=50,
        # resume_from_last defaults to True, so a walltime kill only needs the same command
        # re-submitted -- but only if checkpoints were actually written along the way.
        save_ckpt_interval_epoch=2,
        ckpt_save_max_keep=3,
        callbacks=[
            dict(
                type='HFxView2StandardEval',
                params=dict(split='test', epoch_interval=5),
            ),
        ]
    ),
    test=dict()
)
