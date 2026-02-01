import torch

import torchange.models.changestar_1xd as cs1


def test_changestar1xd_forward_eval(monkeypatch):
    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cs1.er.config.AttrDict(out_channels=4)
            self.conv = torch.nn.Conv2d(3, 4, 1)

        def forward(self, x):
            t1 = self.conv(x[:, :3])
            t2 = self.conv(x[:, 3:])
            return t1, t2

    monkeypatch.setattr(cs1.er.builder, "make_model", lambda cfg: DummyEncoder())

    cfg = cs1.er.config.AttrDict(
        encoder=dict(type="dummy", params=dict(), bitemporal_forward=False),
        head=dict(num_semantic_classes=2, num_change_classes=1, temporal_symmetric=True),
        loss=dict(),
        train_mode=cs1.TrainMode.BSL,
    )

    model = cs1.ChangeStar1xd(cfg)
    model.eval()

    x = torch.randn(1, 6, 8, 8)
    out = model(x)

    assert out.change_prediction.shape == (1, 1, 32, 32)
    assert out.t1_semantic_prediction.shape == (1, 2, 32, 32)
    assert out.t2_semantic_prediction.shape == (1, 2, 32, 32)
