import pytest
import torch

from torchange.module.uper import PyramidPoolingModule, UPerHead


@pytest.mark.parametrize("pool_scales", [(1, 2), (1, 2, 3)])
def test_pyramid_pooling_module_shape(pool_scales):
    ppm = PyramidPoolingModule(in_channels=8, out_channels=4, pool_scales=pool_scales)
    x = torch.randn(2, 8, 8, 8)
    out = ppm(x)
    assert out.shape == (2, 4, 8, 8)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("align_corners", [True, False])
def test_uper_head_output_shape(align_corners):
    head = UPerHead(in_channels=[64, 128, 320, 512], channels=16, num_classes=3, align_corners=align_corners)
    inputs = [
        torch.randn(1, 64, 8, 8),
        torch.randn(1, 128, 4, 4),
        torch.randn(1, 320, 2, 2),
        torch.randn(1, 512, 1, 1),
    ]
    out = head(inputs)
    assert out.shape == (1, 3, 8, 8)


def test_uper_head_input_length_mismatch():
    head = UPerHead(in_channels=[64, 128, 320, 512], channels=16, num_classes=3)
    inputs = [torch.randn(1, 64, 8, 8)]
    with pytest.raises(AssertionError):
        head(inputs)
