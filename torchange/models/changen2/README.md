# Changen2 (TPAMI 2024)

This is the official repository for IEEE TPAMI 2024 paper 
"_Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model_".  

Authors: 
[Zhuo Zheng](https://zhuozheng.top/)
[Stefano Ermon](https://cs.stanford.edu/~ermon/)
[Dongjun Kim](https://sites.google.com/view/dongjun-kim)
[Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)
[Yanfei Zhong](http://rsidea.whu.edu.cn/)

Abstract: Our understanding of the temporal dynamics of the Earth's surface has been significantly advanced by deep vision models, which often require a massive amount of labeled multi-temporal images for training.
However, collecting, preprocessing, and annotating multi-temporal remote sensing images at scale is non-trivial since it is expensive and knowledge-intensive.
In this paper, we present scalable multi-temporal change data generators based on generative models, which are cheap and automatic, alleviating these data problems. 
Our main idea is to simulate a stochastic change process over time.
We describe the stochastic change process as a probabilistic graphical model, namely the generative probabilistic change model (GPCM), which factorizes the complex simulation problem into two more tractable sub-problems, i.e., condition-level change event simulation and image-level semantic change synthesis. 
To solve these two problems, we present Changen2, a GPCM implemented with a resolution-scalable diffusion transformer which can generate time series of remote sensing images and corresponding semantic and change labels from labeled and even unlabeled single-temporal images.
Changen2 is a generative change foundation model that can be trained at scale via self-supervision, and is capable of producing change supervisory signals from unlabeled single-temporal images.
Unlike existing foundation models, our generative change foundation model synthesizes change data to train task-specific foundation models for change detection.
The resulting model possesses inherent zero-shot change detection capabilities and excellent transferability. 
Comprehensive experiments suggest Changen2 has superior spatiotemporal scalability in data generation, e.g., Changen2 model trained on 256$^2$ pixel single-temporal images can yield time series of any length and resolutions of 1,024^2 pixels.
Changen2 pre-trained models exhibit superior zero-shot performance (narrowing the performance gap to 3% on LEVIR-CD and approximately 10% on both S2Looking and SECOND, compared to fully supervised counterpart) and transferability across multiple types of change tasks, including ordinary and off-nadir building change, land-use/land-cover change, and disaster assessment.

## Get Started (continuously updated)

### Change Event Simulation
```python
from torchange.models.changen2 import change_event_simulation as ces

# Changen2, Sec. 3.2, Change Event Simulation
ces.add_object  # Object Creation
ces.remove_object  # Object Removal
ces.random_transition  # Attribute Edit
# Changen2, Sec. 3.5, Fig.7
ces.next_time_contour_gen
```

### Resolution-Scalable DiT models
You can use our RSDiT architecture as follows:
```python
from torchange.models.changen2 import RSDiT_models
```

### Changen2 pre-trained ChangeStar (1x256) models
We provide the ChangeStar (1x256) architecture along with several pre-trained models, available for both further research and direct application.

See this [notebook](https://github.com/Z-Zheng/pytorch-change-models/blob/main/examples/changen2_pretrained_changestar1x256_inference_demo.ipynb) demo for model inference

```python
from torchange.models.changestar_1xd import ChangeStar1xd
from torchange.models.changen2 import changestar_1x256

# zero-shot building change detection models
from torchange.models.changen2 import s1_init_s1c1_changestar_vitb_1x256
from torchange.models.changen2 import s1_init_s1c1_changestar_vitl_1x256

# zero-shot semantic (land use/land cover) change detection model
from torchange.models.changen2 import s9_init_s9c1_changestar_vitb_1x256

# zero-shot class-agnostic change detection model
# also can be used as a pre-trained startpoint for binary change detection models
from torchange.models.changen2 import s0_init_s1c1_changestar_vitb_1x256

# The two below load the above pre-trained weights and can only be used as pre-trained startpoint.
from torchange.models.changen2 import s0_init_s1c5_changestar_vitb_1x256
from torchange.models.changen2 import s0_init_s9c1_changestar_vitb_1x256

# This one was fine-tuned on xView2 dataset, see Table 6 in our paper for model performance
from torchange.models.changen2 import s0_init_xView2_ft_s1c5_changestar_vitb_1x256

# alias for s0_init_xView2_ft_s1c5_changestar_vitb_1x256
# providing an out-of-box building damage assessment model
from torchange.models.changen2 import building_damage_assessment_model
```

### Synthetic Change Datasets
[Changen2-S1-15k](https://huggingface.co/datasets/EVER-Z/Changen2-S1-15k), a building change dataset with 15k pairs and 2 change types), 0.3-1m spatial resolution, RGB bands

[Changen2-S9-27k](https://huggingface.co/datasets/EVER-Z/Changen2-S9-27k), an urban land-use/landcover change dataset with 27k pairs and 38 change types), 0.25-0.5m spatial resolution, RGB bands

We also provide a convenient version you can load via huggingface's datasets library, see [example](https://github.com/Z-Zheng/pytorch-change-models/blob/main/examples/load_Changen2_synthetic_change_dataset.ipynb)

## Citation
If you find our project helpful, we would greatly appreciate it if you could kindly cite our papers:
```
@article{changen2,
  author={Zheng, Zhuo and Ermon, Stefano and Kim, Dongjun and Zhang, Liangpei and Zhong, Yanfei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model}, 
  year={2025},
  volume={47},
  number={2},
  pages={725-741},
  doi={10.1109/TPAMI.2024.3475824}
}
@inproceedings{changen,
  title={Scalable multi-temporal remote sensing change data generation via simulating stochastic change process},
  author={Zheng, Zhuo and Tian, Shiqi and Ma, Ailong and Zhang, Liangpei and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21818--21827},
  year={2023}
}
```