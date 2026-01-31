from configs import comm
from torchvision.models import Swin_T_Weights

config = dict(
    model=dict(
        type='mmChangeStar2_5',
        params=dict(
            GLOBAL=dict(weight=dict(path=None)),
            image_dense_encoder=dict(
                type='SwinFarSeg',
                params=dict(
                    name='swin_t',
                    weights=Swin_T_Weights.IMAGENET1K_V1,
                    out_channels=256,
                ),
            ),
            mixin=dict(s=1, c=4, t2_on=False, n_blocks=3),
            loss=dict(
                t1=dict(tver=dict(alpha=0.9)),
                change=dict(ce=dict(ls=0.), tver=dict()),
            )
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
