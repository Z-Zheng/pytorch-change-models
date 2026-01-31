# Segment Any Change (NeurIPS 2024)

This is the unofficial SAM2 version for the NeurIPS 2024 paper 
"_Segment Any Change_".  

Authors: 
[ChunLei Chen](chenchunlei@live.cn)


## Get Started
### install sam2
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 7d148a50107c5bb4ac36078454b4a3bda6251fe3
pip install -e .
```
### Case 1: automatic mode (segment any change)
```python
import matplotlib.pyplot as plt
from skimage.io import imread
from torchange.models.segment_any_change import AnyChange_sam2, show_change_masks_sam2

# initialize AnyChange
m = AnyChange_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", sam2_checkpoint="./sam2.1_hiera_tiny.pt")
# customize the hyperparameters of SAM's mask generator
m.set_hyperparameters(change_confidence_threshold=145, use_normalized_feature=True, bitemporal_match=True)

img1 = imread('https://raw.githubusercontent.com/Z-Zheng/pytorch-change-models/main/demo_images/t1_img.png')
img2 = imread('https://raw.githubusercontent.com/Z-Zheng/pytorch-change-models/main/demo_images/t2_img.png')

changemasks, _, _ = m.forward(img1, img2) # automatic mode
fig, axes, _ = show_change_masks_sam2(img1, img2, changemasks)

plt.show()
```

### Case 2: point query mode (not yet)



## Citation
If you find our project helpful, please cite our paper:
```
@inproceedings{
zheng2024anychange,
title={Segment Any Change},
author={Zhuo Zheng and Yanfei Zhong and Liangpei Zhang and Stefano Ermon},
booktitle={Advances in Neural Information Processing Systems},
year={2024},
}
```
