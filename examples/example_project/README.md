## 🚀 Example project: Benchmark Swin-based ChangeOS model on BRIGHT dataset

This project provides a whole benchmark pipeline and pre-trained weights for two architectures on the [**BRIGHT**](https://github.com/ChenHongruixuan/BRIGHT) dataset to demonstrate a good practice of `torchange`.

### 📊 Performance Summary

All models are evaluated on the standard test split using set-level mIoU.

| arch                                                                                    | backbone | mIoU  | Bg    | intact | Damaged | Destroyed | Weights                                                                                    |
|:----------------------------------------------------------------------------------------|:---------|:-----:|:------|:-------|:--------|:----------|:-------------------------------------------------------------------------------------------|
| [**ChangeOS**](https://www.sciencedirect.com/science/article/abs/pii/S0034425721003564) | Swin-T   | 67.26 | 96.81 | 75.15  | 40.17   | 56.92     | [🤗link](https://huggingface.co/EVER-Z/torchange_example_changeos_swint_on_bright_ckpt40k) |
| [**ChangeStar**](https://arxiv.org/abs/2108.07002) (1x256)                              | DINOv3-L |   -   | -     | -      | -       | -         | *Coming soon*                                                                              |

---

### 🛠️ Reproduction Guide

#### 1. Swin-based ChangeOS
```bash
# remove --use_wandb and --project if you don't have wandb account
# Configuration
config_path='configs/swint_cos.py'
model_dir='logs/bright_swint_cos'

torchrun --nnodes=1 --nproc_per_node=2 --master_port $RANDOM -m torchange.training.bisup_train_bright \
  --config_path=${config_path} \
  --model_dir=${model_dir} \
  --mixed_precision='bf16' \
  --use_wandb \
  --project 'torchange_example_project_BRIGHT_bench' \
  --eval_epoch_interval 5 \
  data.train.params.batch_size 8 \
  data.train.params.num_workers 4

# Export your model to HuggingFace Hub
# remove --private if you want to make your model public
# if you don't specific --checkpoint_name, the best model will be exported
python -m torchange.utils.push_to_hub model_dir_to_hub \
  --model_dir ${model_dir} \
  --repo_id <your hf username/repo name> \
  --checkpoint_name 'checkpoint-40000.pth' \
  --private

# for example, the model at 40k steps is exported to
# https://huggingface.co/EVER-Z/torchange_example_changeos_swint_on_bright_ckpt40k
python -m torchange.utils.push_to_hub model_dir_to_hub \
  --model_dir ${model_dir} \
  --repo_id EVER-Z/torchange_example_changeos_swint_on_bright_ckpt40k \
  --checkpoint_name 'checkpoint-40000.pth'
```

#### 2. DINOv3-based ChangeStar (1x256)
```bash
# remove --use_wandb and --project if you don't have wandb account
config_path='configs/dinov3_cstar_vitl_1x256.py'
model_dir='logs/bright_dinov3_cstar_vitl_1x256'
torchrun --nnodes=1 --nproc_per_node=2 --master_port $RANDOM -m torchange.training.bisup_train_bright \
  --config_path=${config_path} \
  --model_dir=${model_dir} \
  --mixed_precision='bf16' \
  --use_wandb \
  --project 'torchange_example_project_BRIGHT_bench' \
  --eval_epoch_interval 5 \
  data.train.params.batch_size 8 \
  data.train.params.num_workers 4
```

### 📚 References

If you find this work or the models useful, please consider citing the following relevant papers:

```bibtex
@software{zheng2024torchange,
  author = {Zheng, Zhuo},
  title = {torchange: A Unified Change Representation Learning Benchmark Library},
  url = {https://github.com/Z-Zheng/pytorch-change-models},
  year = {2024}
}

# for BRIGHT dataset
@article{chen2025bright,
  title={Bright: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response},
  author={Chen, Hongruixuan and Song, Jian and Dietrich, Olivier and Broni-Bediako, Clifford and Xuan, Weihao and Wang, Junjue and Shao, Xinlei and Wei, Yimin and Xia, Jun and Lan, Cuiling and Schindler, Konrad and Yokoya, Naoto},
  journal={Earth System Science Data},
  volume={17},
  number={11},
  pages={6217--6253},
  year={2025},
  publisher={Copernicus Publications}
}

# for ChangeOS
@article{zheng2021changeos,
  title={Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: From natural disasters to man-made disasters},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong and Zhang, Liangpei},
  journal={Remote Sensing of Environment},
  volume={265},
  pages={112636},
  year={2021},
  publisher={Elsevier}
}

# for ChangeStar network architecture; 
# btw, someone is always confused about the difference between ChangeStar and STAR (single-temporal supervised learning).
# ChangeStar is a network architecture that can be trained with the STAR algorithm or bitemporal supervised learning.
@inproceedings{zheng2021changestar,
  title={Change is everywhere: Single-temporal supervised object change detection in remote sensing imagery},
  author={Zheng, Zhuo and Ma, Ailong and Zhang, Liangpei and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={15193--15202},
  year={2021}
}

# for ChangeStar (1x256) network architecture
@inproceedings{zheng2023scalable,
  title={Scalable multi-temporal remote sensing change data generation via simulating stochastic change process},
  author={Zheng, Zhuo and Tian, Shiqi and Ma, Ailong and Zhang, Liangpei and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21818--21827},
  year={2023}
}

# for Swin-T backbone
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={10012--10022},
  year={2021}
}

# for DINOv3-L backbone
@article{simeoni2025dinov3,
  title={Dinov3},
  author={Sim{\'e}oni, Oriane and Vo, Huy V and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and others},
  journal={arXiv preprint arXiv:2508.10104},
  year={2025}
}
