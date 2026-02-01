import torch

import torchange.module.tv_swin as tv_swin


def test_tvswin_out_channels():
    cfg = tv_swin.er.config.AttrDict(name='swin_t', weights=None)
    model = tv_swin.TVSwinTransformer(cfg)
    assert model.out_channels() == tv_swin.TVSwinTransformer.OUT_CHANNELS['swin_t']


def test_tvswin_forward_outputs_list():
    cfg = tv_swin.er.config.AttrDict(name='swin_t', weights=None)
    model = tv_swin.TVSwinTransformer(cfg)
    x = torch.randn(1, 3, 8, 8)
    feats = model(x)

    assert isinstance(feats, list)
    assert len(feats) == 4
    assert all(f.dim() == 4 for f in feats)
