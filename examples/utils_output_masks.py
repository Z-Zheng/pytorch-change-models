"""Small usage examples for torchange.utils outputs and masks."""
import torch

from torchange.utils.mask_data import Mask
from torchange.utils.outputs import ChangeDetectionModelOutput


def demo_change_detection_output() -> None:
    t1 = torch.rand(1, 256, 256)
    t2 = torch.rand(1, 256, 256)
    change = torch.rand(1, 256, 256)

    out = ChangeDetectionModelOutput(
        t1_semantic_prediction=t1,
        t2_semantic_prediction=t2,
        change_prediction=change,
    )

    print(f"Attribute access: {out.t1_semantic_prediction.shape}")
    print(f"Dict access:      {out['change_prediction'].shape}")

    print("\nIterating over items:")
    for k, v in out.items():
        print(f" - {k}: {v.shape}")

    print("\nLog representation:")
    print(out)


def demo_mask_from_list() -> None:
    masks = [
        torch.rand(1, 256, 256),
        torch.rand(1, 256, 256),
        torch.rand(1, 256, 256),
    ]
    mask = Mask.from_list(masks)
    print(mask)


if __name__ == "__main__":
    demo_change_detection_output()
    demo_mask_from_list()
