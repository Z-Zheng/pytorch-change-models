# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torchange.data.bitemporal import HFBitemporalDataset

HF_DATASETS = {
    'levircd': 'EVER-Z/torchange_levircd',
    's2looking': 'EVER-Z/torchange_s2looking',
    'second': 'EVER-Z/torchange_second',
    'xView2': 'EVER-Z/torchange_xView2',
    'Changen2-S1-15k': 'EVER-Z/torchange_Changen2-S1-15k',
    'Changen2-S9-27k': 'EVER-Z/torchange_Changen2-S9-27k',
    'bright': 'EVER-Z/torchange_bright',
    'dsifn-cd': 'EVER-Z/torchange_dsifn-cd',
}


def build_dataset(dataset_name, splits, transform, **kwargs):
    """Build a HuggingFace-backed dataset by name.

    Parameters
    ----------
    dataset_name : str
        Key in :data:`HF_DATASETS`.
    splits : sequence of str
        Dataset splits to combine.
    transform : callable
        Transformations applied to the data.
    **kwargs
        Additional arguments required by some datasets.

    Returns
    -------
    :class:`HFBitemporalDataset`
        Instantiated dataset.
    """
    assert dataset_name in HF_DATASETS

    if dataset_name == 'xView2':
        from torchange.data.xView2 import HFxView2
        assert 'crop_size' in kwargs
        assert 'stride' in kwargs
        assert 'training' in kwargs

        return HFxView2(dict(
            hf_repo_name=HF_DATASETS[dataset_name],
            splits=splits,
            transform=transform,
            crop_size=kwargs['crop_size'],
            stride=kwargs['stride'],
            training=kwargs['training'],
        ))

    if dataset_name == 'bright':
        from torchange.data.bright import HFBRIGHT
        assert 'setting' in kwargs
        assert 'setting_splits' in kwargs

        return HFBRIGHT(dict(
            hf_repo_name=HF_DATASETS[dataset_name],
            splits=['full'],
            transform=transform,
            setting=kwargs['setting'],
            setting_splits=kwargs['setting_splits'],
        ))

    dataset = HFBitemporalDataset(dict(
        hf_repo_name=HF_DATASETS[dataset_name],
        splits=splits,
        transform=transform,
    ))

    return dataset
