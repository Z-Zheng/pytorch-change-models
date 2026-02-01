import torch

import torchange.models.changestar2 as cs2


def test_changestar2_forward_eval(monkeypatch):
    class DummySeg(torch.nn.Module):
        def forward(self, x):
            return x

    class DummyDetector(torch.nn.Module):
        def forward(self, features):
            t, b, c, h, w = features.shape
            return torch.zeros((b, 1, h, w))

    class DummyTargetGen:
        def __init__(self, *args, **kwargs):
            return None

        def __call__(self, x, y):
            return x, y

    monkeypatch.setattr(cs2, "Segmentation", lambda cfg: DummySeg())
    monkeypatch.setattr(cs2, "get_detector", lambda **cfg: DummyDetector())
    monkeypatch.setattr(cs2, "TargetGenerator", lambda name, **kwargs: DummyTargetGen())

    cfg = cs2.er.config.AttrDict(
        segmentation=dict(model_type="farseg"),
        semantic_classifier=dict(in_channels=3, out_channels=1, scale=4.0),
        change_detector=dict(name="dummy", in_channels=3, scale=4.0),
        target_generator=dict(name="sync_generate_target_v3", shuffle_prob=0.0),
        loss=dict(),
        pcm_m2m_inference=False,
    )

    model = cs2.ChangeStar2(cfg)
    model.eval()

    out = model(torch.randn(1, 6, 4, 4))
    assert out["type"] == "bcd"
    assert out["change_prediction"].shape == (1, 1, 16, 16)
