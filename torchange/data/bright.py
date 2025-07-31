# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torchange.data.bitemporal import HFBitemporalDataset
from enum import StrEnum


class Setting(StrEnum):
    STANDARD = 'std_split'
    ONESHOT = 'oneshot_split'
    UMCD = 'umcd_split'
    EVENT = 'event'


EVENTS = [
    'morocco-earthquake',
    'turkey-earthquake',
    'la_palma-volcano',
    'ukraine-conflict',
    'congo-volcano',
    'libya-flood',
    'beirut-explosion',
    'mexico-hurricane',
    'hawaii-wildfire',
    'noto-earthquake',
    'marshall-wildfire',
    'myanmar-hurricane',
    'bata-explosion',
    'haiti-earthquake'
]


class HFBRIGHT(HFBitemporalDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert self.cfg.setting in [Setting.STANDARD, Setting.ONESHOT, Setting.UMCD, Setting.EVENT]
        self.hfd = self.hfd.filter(lambda x: x in self.cfg.setting_splits, input_columns=self.cfg.setting)

    def set_default_config(self):
        super().set_default_config()
        self.cfg.update(dict(
            hf_repo_name='EVER-Z/torchange_bright',
            splits=['full'],
            setting=Setting.STANDARD,
            setting_splits=['train'],
        ))

    @property
    def events(self):
        return self.hfd.unique('event')
