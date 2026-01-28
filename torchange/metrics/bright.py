# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ever as er
import albumentations as A
from torchange.metrics.scd import MultiClassPixelEval
from torchange.data.bright import Setup, HFBRIGHT


@er.registry.CALLBACK.register(verbose=False)
class BRIGHTEval(MultiClassPixelEval):
    def __init__(self, epoch_interval, splits=('test',), setting=Setup.STANDARD):
        super().__init__(
            data_cfg=dict(
                type='HFBRIGHT',
                params=dict(
                    hf_repo_name='EVER-Z/torchange_bright',
                    setting=setting,
                    setting_splits=splits,
                    transform=A.Compose([
                        A.Normalize(
                            (0.485, 0.456, 0.406, 0.225),
                            (0.229, 0.224, 0.225, 0.151),
                            max_pixel_value=255
                        ),
                        A.pytorch.ToTensorV2(),
                    ]),
                    batch_size=1,
                    num_workers=2,
                ),
            ),
            num_classes=4,
            class_names=['Background', 'Intact', 'Damaged', 'Destroyed'],
            epoch_interval=epoch_interval,
            prior=101,
        )

    def extract_score(self, result: er.metric.pixel.AccTable):
        return {
            'eval/mIoU': result.iou(-3),
            'eval/Background': result.iou(0),
            'eval/Intact': result.iou(1),
            'eval/Damaged': result.iou(2),
            'eval/Destroyed': result.iou(3),
        }
