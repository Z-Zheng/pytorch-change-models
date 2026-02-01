import pytest
import torch
import torch.nn.functional as F

from torchange.module.lora import LoraLinear


@pytest.mark.parametrize("r,merge_weights", [(0, False), (2, False), (2, True)])
def test_lora_linear_forward_shapes(r, merge_weights):
    torch.manual_seed(0)
    layer = LoraLinear(4, 3, bias=True, r=r, lora_alpha=2, lora_dropout=0.0, merge_weights=merge_weights)
    x = torch.randn(2, 4)

    out = layer(x)
    assert out.shape == (2, 3)

    if r == 0:
        ref = F.linear(x, layer.weight, layer.bias)
        assert torch.allclose(out, ref, atol=1e-6)

    if merge_weights and r > 0:
        out_train = out.clone()
        layer.train(False)
        assert layer.merged is True
        out_eval = layer(x)
        assert torch.allclose(out_train, out_eval, atol=1e-6)
        layer.train(True)
        assert layer.merged is False


def test_convert_lora_linear_recurses():
    base = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2, bias=False),
    )

    converted = LoraLinear.convert_lora_linear(base, r=2, lora_alpha=2)
    assert isinstance(converted[0], LoraLinear)
    assert isinstance(converted[2], LoraLinear)
    assert isinstance(converted[1], torch.nn.ReLU)
