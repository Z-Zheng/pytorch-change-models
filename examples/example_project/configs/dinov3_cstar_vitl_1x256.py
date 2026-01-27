from configs import comm

config = dict(
    model=dict(
        type='mmChangeStar1xd',
        params=dict(
            GLOBAL=dict(weight=dict(path=None)),
            encoder=dict(
                bitemporal_forward=False,
                type='DINOv3ViTLFarSeg',
                params=dict(
                    pretrained='pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
                    lora=dict(r=32, lora_alpha=320),
                    fpn_channels=256,
                    out_channels=256,
                    drop_path_rate=0.3,
                ),
            ),
            head=dict(num_semantic_classes=1, num_change_classes=4),
            loss=dict(
                t1=dict(),
                change=dict(ce=dict(ls=0.), dice=dict(gamma=1.0)),
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
