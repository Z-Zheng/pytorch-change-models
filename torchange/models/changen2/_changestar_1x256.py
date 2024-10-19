# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import ever as er
from torchange.models.changestar_1xd import ChangeStar1xd

__all__ = [
    'changestar_1x256',
    's1_init_s1c1_changestar_vitb_1x256',
    's1_init_s1c1_changestar_vitl_1x256',
    's9_init_s9c1_changestar_vitb_1x256',
    's0_init_s1c1_changestar_vitb_1x256',
    's0_init_s1c5_changestar_vitb_1x256',
    's0_init_s9c1_changestar_vitb_1x256',
]


def changestar_1x256(backbone_type, modeling_type, changen2_pretrained=None) -> ChangeStar1xd:
    import json
    from huggingface_hub import hf_hub_download
    from torchange.module.farseg import SAMEncoderFarSeg
    assert modeling_type in ['s1c1', 's9c1', 's1c5', ]
    assert backbone_type in ['vitb', 'vitl']
    assert changen2_pretrained in [None, 's0', 's1', 's9']
    pretrain_data = {
        None: None,
        's0': 'Changen2-S0-1.2M',
        's1': 'Changen2-S1-15k',
        's9': 'Changen2-S9-27k'
    }

    model_name = f'{modeling_type}_cstar_{backbone_type}_1x256'
    # build model
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg = er.config.import_config(os.path.join(package_root, 'configs', 'changen2', model_name))
    model: ChangeStar1xd = er.builder.make_model(cfg.model)

    # load Changen2 pre-trained weight
    if changen2_pretrained:
        available_config = hf_hub_download('EVER-Z/Changen2-ChangeStar1x256', 'config.json')
        with open(available_config, "r", encoding="utf-8") as reader:
            text = reader.read()
        available_config = json.loads(text)
        weight_name = f'{changen2_pretrained}_changestar_{backbone_type}_1x256'
        assert weight_name in available_config, f'{weight_name} is not available'
        weights = hf_hub_download('EVER-Z/Changen2-ChangeStar1x256', available_config[weight_name])

        model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')), strict=changen2_pretrained != 's0')
        er.info(f'Load Changen2 pre-trained weight from EVER-Z/Changen2-ChangeStar1x256/{available_config[weight_name]}')

    er.info(
        f'architecture: changestar_1x256 | backbone: {backbone_type} | pre-trained data: {pretrain_data[changen2_pretrained]}')
    return model


def s1_init_s1c1_changestar_vitb_1x256(): return changestar_1x256('vitb', 's1c1', 's1')


def s1_init_s1c1_changestar_vitl_1x256(): return changestar_1x256('vitl', 's1c1', 's1')


def s9_init_s9c1_changestar_vitb_1x256(): return changestar_1x256('vitb', 's9c1', 's9')


def s0_init_s1c1_changestar_vitb_1x256(): return changestar_1x256('vitb', 's1c1', 's0')


def s0_init_s9c1_changestar_vitb_1x256(): return changestar_1x256('vitb', 's9c1', 's0')


def s0_init_s1c5_changestar_vitb_1x256(): return changestar_1x256('vitb', 's1c5', 's0')
