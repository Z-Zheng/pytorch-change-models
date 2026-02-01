import torch

import torchange.models.changestar2_5 as cs25


def test_changestar2_5_forward_eval(monkeypatch):
    class DummyEncoder(torch.nn.Module):
        def __init__(self, out_channels=4):
            super().__init__()
            self.out_channels = out_channels

        def forward(self, x):
            b, _, h, w = x.shape
            return torch.zeros((b, self.out_channels, h // 2, w // 2))

    dummy = DummyEncoder(out_channels=4)
    monkeypatch.setattr(cs25.er.builder, "make_model", lambda cfg: dummy)

    cfg = cs25.er.config.AttrDict(
        image_dense_encoder=dict(type="dummy", params=dict(out_channels=4)),
        mixin=dict(s=2, c=1, temporal_symmetric=True, t1_on=True, t2_on=True, n_blocks=0, upsample_scale=2),
        loss=dict(),
        train_mode=cs25.TrainMode.BSL,
    )

    model = cs25.ChangeStar2_5(cfg)
    model.eval()

    x = torch.randn(1, 6, 8, 8)
    out = model(x)

    assert out.change_prediction.shape == (1, 1, 8, 8)
    assert out.t1_semantic_prediction.shape == (1, 2, 8, 8)
