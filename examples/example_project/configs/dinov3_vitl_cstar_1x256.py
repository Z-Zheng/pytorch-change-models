from configs import comm
import albumentations as A
import albumentations.pytorch

comm.train_data['train']['params']['transform'] = A.Compose([
    A.RandomCrop(640, 640),
    A.D4(),
    A.Normalize(
        (0.430, 0.411, 0.296, 0.225),
        (0.213, 0.156, 0.143, 0.151),
        max_pixel_value=255
    ),
    A.pytorch.ToTensorV2(),
])

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
                    out_channels=256,
                    drop_path_rate=0.3,
                    dinov3_forward_mode='one_level'
                ),
            ),
            head=dict(num_semantic_classes=1, num_change_classes=4),
            loss=dict(
                t1=dict(tver=dict(alpha=0.9)),
                change=dict(ce=dict(ls=0.), dice=dict()),
            )
        )
    ),
    data=comm.train_data,
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=1e-4,
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
