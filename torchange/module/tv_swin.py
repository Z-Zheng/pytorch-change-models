# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision as tv
from torchvision.models import swin_transformer
import ever as er


def get_stages(self):
    return [
        self.features[:2],
        self.features[2:4],
        self.features[4:6],
        self.features[6:],
    ]


def forward(self, x):
    feats = []
    for s in self.get_stages():
        x = s(x)
        feats.append(x.permute(0, 3, 1, 2).contiguous())

    return feats


swin_transformer.SwinTransformer.get_stages = get_stages
swin_transformer.SwinTransformer.forward = forward


@er.registry.MODEL.register(verbose=False)
class TVSwinTransformer(er.ERModule):
    OUT_CHANNELS = {
        'swin_s': [96 * (2 ** i) for i in range(4)],
        'swin_t': [96 * (2 ** i) for i in range(4)],
        'swin_b': [128 * (2 ** i) for i in range(4)],
        'swin_l': [192 * (2 ** i) for i in range(4)],
        'swin_v2_s': [96 * (2 ** i) for i in range(4)],
        'swin_v2_t': [96 * (2 ** i) for i in range(4)],
        'swin_v2_b': [128 * (2 ** i) for i in range(4)],
        'swin_v2_l': [192 * (2 ** i) for i in range(4)],
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        name = self.cfg.name
        weights = self.cfg.weights
        self.swin = getattr(tv.models, name)(weights=weights, progress=er.dist.is_main_process())

        del self.swin.norm
        del self.swin.head

    def forward(self, x):
        return self.swin(x)

    def out_channels(self):
        return self.OUT_CHANNELS[self.cfg.name]

    def custom_param_groups(self):
        param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
        for i, p in self.named_parameters():
            if 'norm' in i:
                param_groups[0]['params'].append(p)
            elif 'relative_position_bias_table' in i:
                param_groups[0]['params'].append(p)
            elif 'absolute_pos_embed' in i:
                param_groups[0]['params'].append(p)
            else:
                param_groups[1]['params'].append(p)
        return param_groups

    def set_default_config(self):
        self.cfg.update(dict(name='swin_t', weights=tv.models.Swin_T_Weights))