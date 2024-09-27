# Segment Any Change (NeurIPS 2024)

This is the official repository for the NeurIPS 2024 paper 
"_Segment Any Change_".  

Authors: 
[Zhuo Zheng](https://zhuozheng.top/)
[Yanfei Zhong](http://rsidea.whu.edu.cn/)
[Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)
[Stefano Ermon](https://cs.stanford.edu/~ermon/).

Abstract: Visual foundation models have achieved remarkable results in zero-shot image classification and segmentation, but zero-shot change detection remains an open problem.
In this paper, we propose the segment any change models (AnyChange), a new type of change detection model that supports zero-shot prediction and generalization on unseen change types and data distributions.
AnyChange is built on the segment anything model (SAM) via our training-free adaptation method, bitemporal latent matching.
By revealing and exploiting intra-image and inter-image semantic similarities in SAM's latent space, bitemporal latent matching endows SAM with zero-shot change detection capabilities in a training-free way. 
We also propose a point query mechanism to enable AnyChange's zero-shot object-centric change detection capability.

## Get Started
TBD


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