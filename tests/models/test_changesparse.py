import torch

import torchange.models.changesparse as changesparse


def test_changesparse_bcd_forward_eval(monkeypatch):
    class DummyBackbone(torch.nn.Module):
        def forward(self, x):
            b, _, h, w = x.shape
            return [torch.zeros((b, 4, h, w)) for _ in range(4)]

    class DummyAttn(torch.nn.Module):
        def __init__(self, inner_channels=4, *args, **kwargs):
            super().__init__()
            self.inner_channels = inner_channels

        def forward(self, features):
            b, _, h, w = features[0].shape
            out = torch.zeros((b, self.inner_channels, h, w))
            return {
                "output_feature": out,
                "intermediate_logits": [out[:, :1, :, :]],
                "estimated_change_ratios": [0.0],
            }

    monkeypatch.setattr(changesparse, "get_backbone", lambda *args, **kwargs: (DummyBackbone(), (4, 4, 4, 4)))
    monkeypatch.setattr(changesparse, "SparseChangeTransformer", lambda *args, **kwargs: DummyAttn())

    cfg = changesparse.er.config.AttrDict(transformer=dict(inner_channels=4))
    model = changesparse.ChangeSparseBCD(cfg)
    model.eval()

    out = model(torch.randn(1, 6, 4, 4))
    assert "change_prediction" in out


def test_changesparse_o2m_forward_eval(monkeypatch):
    class DummyBackbone(torch.nn.Module):
        def forward(self, x):
            b, _, h, w = x.shape
            return [torch.zeros((b, 4, h, w)) for _ in range(4)]

    class DummyAttn(torch.nn.Module):
        def __init__(self, inner_channels=4, *args, **kwargs):
            super().__init__()
            self.inner_channels = inner_channels

        def forward(self, features):
            b, _, h, w = features[0].shape
            out = torch.zeros((b, self.inner_channels, h, w))
            return {
                "output_feature": out,
                "intermediate_logits": [out[:, :1, :, :]],
                "estimated_change_ratios": [0.0],
            }

    class DummySemanticDecoder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, features):
            b, _, h, w = features[0].shape
            out = torch.zeros((b, 4, h, w))
            return {"output_feature": out}

    monkeypatch.setattr(changesparse, "get_backbone", lambda *args, **kwargs: (DummyBackbone(), (4, 4, 4, 4)))
    monkeypatch.setattr(changesparse, "ChangeSparseTransformer_multiclass_impl", lambda *args, **kwargs: DummyAttn())
    monkeypatch.setattr(changesparse, "SemanticDecoder", lambda *args, **kwargs: DummySemanticDecoder())

    cfg = changesparse.er.config.AttrDict(
        num_change_classes=2,
        transformer=dict(inner_channels=4),
        semantic_decoder=dict(transformer=dict(inner_channels=4)),
    )
    model = changesparse.ChangeSparseO2M(cfg)
    model.eval()

    out = model(torch.randn(1, 6, 4, 4))
    assert "change_prediction" in out


def test_changesparse_m2m_forward_eval(monkeypatch):
    class DummyBackbone(torch.nn.Module):
        def forward(self, x):
            b, _, h, w = x.shape
            return [torch.zeros((b, 4, h, w)) for _ in range(4)]

    class DummyAttn(torch.nn.Module):
        def __init__(self, inner_channels=4, *args, **kwargs):
            super().__init__()
            self.inner_channels = inner_channels

        def forward(self, features):
            b, _, h, w = features[0].shape
            out = torch.zeros((b, self.inner_channels, h, w))
            return {
                "output_feature": out,
                "intermediate_logits": [out[:, :1, :, :]],
                "estimated_change_ratios": [0.0],
            }

    class DummySemanticDecoder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, features):
            b, _, h, w = features[0].shape
            out = torch.zeros((b, 4, h, w))
            return {"output_feature": out}

    monkeypatch.setattr(changesparse, "get_backbone", lambda *args, **kwargs: (DummyBackbone(), (4, 4, 4, 4)))
    monkeypatch.setattr(changesparse, "SparseChangeTransformer", lambda *args, **kwargs: DummyAttn())
    monkeypatch.setattr(changesparse, "SemanticDecoder", lambda *args, **kwargs: DummySemanticDecoder())

    cfg = changesparse.er.config.AttrDict(
        transformer=dict(inner_channels=4),
        semantic_decoder=dict(num_classes=2, transformer=dict(inner_channels=4)),
    )
    model = changesparse.ChangeSparseM2M(cfg)
    model.eval()

    out = model(torch.randn(1, 6, 4, 4))
    assert "merged_prediction" in out
