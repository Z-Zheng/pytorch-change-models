## torchange - A Unified Change Detection Benchmark Library

torchange aims to provide out-of-box contemporary change detection model implementations, in pursuit of benchmarking and reproducibility. 


>This project is still under development. Other repositories would be gradually merged into ```torchange```.

> The ```torchange``` API is in beta and may change in the near future.

> Note: ```torchange``` is designed to provide straightforward implementations, thus we will adopt a single file for each algorithm without any modular encapsulation.
If you find any bugs, please post them in the issue, since algorithms released before 2024 will be transferred here from our internal codebase.
If possible, be patient for new releases and bug fixes, since this is a huge burden for a single maintainer.

> Our default training engine is [ever](https://github.com/Z-Zheng/ever/). 


### Model zoo (in progress)

This is also a tutorial for junior researchers interested in contemporary change detection.


#### 0. change modeling theory
- (PCM) Unifying Remote Sensing Change Detection via Deep Probabilistic Change Models: from Principles, Models to Applications, ISPRS P&RS 2024. [[`Paper`]], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changesparse.py)]
- (GPCM) Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process, ICCV 2023 [[`Paper`](https://arxiv.org/pdf/2309.17031)], [[`Code`](https://github.com/Z-Zheng/Changen)]


#### 1.0 unified architecture
- (ChangeStar) Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery, ICCV 2021. [[`Paper`](https://arxiv.org/abs/2108.07002)], [[`Project`](https://zhuozheng.top/changestar/)], [[`Code`](https://github.com/Z-Zheng/ChangeStar)]
- (ChangeStar2) Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection, IJCV 2024. [[`Paper`]]
- (ChangeSparse) Unifying Remote Sensing Change Detection via Deep Probabilistic Change Models: from Principles, Models to Applications, ISPRS P&RS 2024. [[`Paper`]], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changesparse.py)]

#### 1.1 one-to-many semantic change detection
- (ChangeOS) Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: from natural disasters to man-made disasters, RSE 2021. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0034425721003564)], [[`Code`](https://github.com/Z-Zheng/ChangeOS)]

#### 1.2 many-to-many semantic change detection
- (ChangeMask) ChangeMask: Deep Multi-task Encoder-Transformer-Decoder Architecture for Semantic Change Detection, ISPRS P&RS 2022. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0924271621002835)]


#### 2.0 learning change representation via single-temporal supervision
- (ChangeStar) Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery, ICCV 2021. [[`Paper`](https://arxiv.org/abs/2108.07002)], [[`Project`](https://zhuozheng.top/changestar/)], [[`Code`](https://github.com/Z-Zheng/ChangeStar)]
- (ChangeStar2) Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection, IJCV 2024. [[`Paper`]]
- (Changen) Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process, ICCV 2023 [[`Paper`](https://arxiv.org/pdf/2309.17031)], [[`Code`](https://github.com/Z-Zheng/Changen)]

#### 3.0 zero-shot change detection
- (AnyChange) Segment Any Change, arxiv 2024 [[`Paper`](https://arxiv.org/abs/2402.01188)]

