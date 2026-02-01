import pytest
import torch

import torchange.module.sam_vit as sam_vit


@pytest.mark.parametrize("shape,window_size", [((1, 5, 6, 3), 4), ((2, 8, 8, 4), 2)])
def test_window_partition_roundtrip(shape, window_size):
    x = torch.randn(*shape)
    windows, pad_hw = sam_vit.window_partition(x, window_size=window_size)
    out = sam_vit.window_unpartition(windows, window_size, pad_hw, (shape[1], shape[2]))
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6)


def test_get_rel_pos_shape():
    rel_pos = torch.randn(5, 2)
    out = sam_vit.get_rel_pos(2, 3, rel_pos)
    assert out.shape == (2, 3, 2)


def test_resample_abs_pos_embed_nhwc():
    posemb = torch.randn(1, 4, 4, 8)
    same = sam_vit.resample_abs_pos_embed_nhwc(posemb, [4, 4])
    assert same.shape == posemb.shape

    resized = sam_vit.resample_abs_pos_embed_nhwc(posemb, [2, 2])
    assert resized.shape == (1, 2, 2, 8)
