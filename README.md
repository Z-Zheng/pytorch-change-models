## torchange - A Unified Change Representation Learning Benchmark Library

torchange aims to provide out-of-box contemporary spatiotemporal change model implementations, standard metrics, and datasets, in pursuit of benchmarking and reproducibility. 

>This project is still under development. Other repositories would be gradually merged into ```torchange```.

> The ```torchange``` API is in beta and may change in the near future.

> Note: ```torchange``` is designed to provide straightforward implementations, thus we will adopt a single file for each algorithm without any modular encapsulation.
Algorithms released before 2024 will be transferred here from our internal codebase.
If you encounter any bugs, please report them in the issue section. Please be patient with new releases and bug fixes, as this is a significant burden for a single maintainer. 
Technical consultations are only accepted via email inquiry.

> Our default training engine is [ever](https://github.com/Z-Zheng/ever/). 

### News

- 2024/06, we launch the project of ``torchange``.

### Features

- Out-of-box and straightforward model implementations
- Highly-optimized implementations, e.g., multi-gpu sync dice loss.
- Multi-gpu metric computation and score tracker, supporting wandb.
- Including the latest research advancements in ``Change``, not just architecture games.

### Installation


#### nightly version (master)
```bash
pip install -U --no-deps --force-reinstall git+https://github.com/Z-Zheng/pytorch-change-models
```

### Model zoo (in progress)

This is also a tutorial for junior researchers interested in contemporary change detection.


#### 0. change modeling principle
- (PCM) Unifying Remote Sensing Change Detection via Deep Probabilistic Change Models: from Principles, Models to Applications, ISPRS P&RS 2024. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0924271624002624)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changesparse.py)]
- (GPCM) Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process, ICCV 2023 [[`Paper`](https://arxiv.org/pdf/2309.17031)], [[`Code`](https://github.com/Z-Zheng/Changen)]


#### 1.0 unified architecture
- (ChangeStar) Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery, ICCV 2021. [[`Paper`](https://arxiv.org/abs/2108.07002)], [[`Project`](https://zhuozheng.top/changestar/)], [[`Code`](https://github.com/Z-Zheng/ChangeStar)]
- (ChangeStar2) Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection, IJCV 2024. [[`Paper`](https://link.springer.com/article/10.1007/s11263-024-02141-4)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changestar2.py)]
- (ChangeSparse) Unifying Remote Sensing Change Detection via Deep Probabilistic Change Models: from Principles, Models to Applications, ISPRS P&RS 2024. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0924271624002624)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changesparse.py)]

#### 1.1 one-to-many semantic change detection
- (ChangeOS) Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: from natural disasters to man-made disasters, RSE 2021. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0034425721003564)], [[`Code`](https://github.com/Z-Zheng/ChangeOS)]

#### 1.2 many-to-many semantic change detection
- (ChangeMask) ChangeMask: Deep Multi-task Encoder-Transformer-Decoder Architecture for Semantic Change Detection, ISPRS P&RS 2022. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0924271621002835)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changemask.py)]


#### 2.0 learning change representation via single-temporal supervision
- (STAR) Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery, ICCV 2021. [[`Paper`](https://arxiv.org/abs/2108.07002)], [[`Project`](https://zhuozheng.top/changestar/)], [[`Code`](https://github.com/Z-Zheng/ChangeStar)]
- (G-STAR) Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection, IJCV 2024. [[`Paper`](https://link.springer.com/article/10.1007/s11263-024-02141-4)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changestar2.py)]
- (Changen) Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process, ICCV 2023 [[`Paper`](https://arxiv.org/pdf/2309.17031)], [[`Code`](https://github.com/Z-Zheng/Changen)]
- (Changen2) Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model, IEEE TPAMI 2024 [[`Paper`](https://arxiv.org/abs/2406.17998)]，[[`Code`](https://github.com/Z-Zheng/pytorch-change-models/tree/main/torchange/models/changen2)]


#### 2.1 change data synthesis from single-temporal data
- (Changen) Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process, ICCV 2023 [[`Paper`](https://arxiv.org/pdf/2309.17031)], [[`Code`](https://github.com/Z-Zheng/Changen)]
- (Changen2) Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model, IEEE TPAMI 2024 [[`Paper`](https://arxiv.org/abs/2406.17998)]，[[`Code`](https://github.com/Z-Zheng/pytorch-change-models/tree/main/torchange/models/changen2)]


#### 3.0 zero-shot change detection
- (AnyChange) Segment Any Change, NeurIPS 2024 [[`Paper`](https://arxiv.org/abs/2402.01188)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/segment_any_change)]
- (Changen2) Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model, IEEE TPAMI 2024 [[`Paper`](https://arxiv.org/abs/2406.17998)]，[[`Code`](https://github.com/Z-Zheng/pytorch-change-models/tree/main/torchange/models/changen2)]


### License
This project is under the Apache 2.0 License. See [LICENSE](https://github.com/Z-Zheng/pytorch-change-models/blob/main/LICENSE) for details.

If you find it useful in your work — whether in research, demos, products, or educational materials — we’d love to hear from you!

Sharing your use case helps us:

📌 Understand real-world impact

📣 Highlight your work in our future talks or papers

🚀 Improve future versions of this project

📬 Please drop us a short message at:
zhuozheng@cs.stanford.edu
Feel free to include a brief description, your institution/company, a link to your work (if available), or any suggestions.


![visitors](https://visitor-badge.laobi.icu/badge?page_id=Z-Zheng/pytorch-change-models)