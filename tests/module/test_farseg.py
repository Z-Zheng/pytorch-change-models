import pytest
import torch

import torchange.module.farseg as farseg


@pytest.mark.parametrize("scale_aware_proj", [True, False])
def test_fsrelationv3_output_shapes(scale_aware_proj):
    module = farseg.FSRelationV3(
        scene_embedding_dim=8,
        in_channels_list=[4, 4],
        out_channels=6,
        scale_aware_proj=scale_aware_proj,
    )
    scene = torch.randn(1, 8, 1, 1)
    feats = [torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4)]
    out = module(scene, feats)
    assert len(out) == 2
    assert out[0].shape == (1, 6, 8, 8)
    assert out[1].shape == (1, 6, 4, 4)
