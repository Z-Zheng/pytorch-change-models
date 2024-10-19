D = 256

config = dict(
    model=dict(
        type='ChangeStar1xd',
        params=dict(
            GLOBAL=dict(weight=dict(path=None)),
            encoder=dict(
                bitemporal_forward=True,
                type='SAMEncoderFarSeg',
                params=dict(
                    checkpoint=None,
                    vit_type='vit_b',
                    fpn_channels=D,
                    out_channels=D,
                    freeze_vit=False,
                ),
            ),
            head=dict(num_semantic_classes=9),
        )
    ),
)
