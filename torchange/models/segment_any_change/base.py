# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from skimage.exposure import match_histograms

from torchvision.ops.boxes import batched_nms
from torchange.models.segment_any_change.simple_maskgen import SimpleMaskGenerator
from torchange.models.segment_any_change.segment_anything import sam_model_registry
from torchange.models.segment_any_change.segment_anything.utils.amg import MaskData
from safetensors.torch import load_file


class SegmentAnyChange:
    def __init__(self, model_type='vit_b', sam_checkpoint='./sam_weights/sam_vit_b_01ec64.pth'):
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sam = sam.to(self.device)
        self.maskgen = SimpleMaskGenerator(self.sam)

        self.set_hyperparameters()

        self.embed_data1 = None
        self.embed_data2 = None

    def set_hyperparameters(self, **kwargs):
        self.match_hist = kwargs.get('match_hist', False)
        self.area_thresh = kwargs.get('area_thresh', 0.8)

    def make_mask_generator(self, **kwargs):
        self.maskgen = SimpleMaskGenerator(self.sam, **kwargs)

    def extract_image_embedding(self, img1, img2):
        self.embed_data1 = self.maskgen.image_encoder(img1)
        self.embed_data2 = self.maskgen.image_encoder(img2)
        return self.embed_data1, self.embed_data2

    def set_cached_embedding(self, embedding):
        data = embedding
        oh, ow = data['original_size'].numpy()
        h, w = data['input_size']
        self.embed_data1 = {
            'image_embedding': data['t1'].to(self.device),
            'original_size': (oh, ow),
        }

        self.embed_data2 = {
            'image_embedding': data['t2'].to(self.device),
            'original_size': (oh, ow),
        }
        self.maskgen.predictor.input_size = (h, w)
        self.maskgen.predictor.original_size = (oh, ow)

    def load_cached_embedding(self, filepath):
        data = load_file(filepath, device='cpu')
        self.set_cached_embedding(data)

    def clear_cached_embedding(self):
        self.embed_data1 = None
        self.embed_data2 = None
        self.maskgen.predictor.input_size = None
        self.maskgen.predictor.original_size = None

    def proposal(self, img1, img2):
        h, w = img1.shape[:2]
        if self.embed_data1 is None:
            self.extract_image_embedding(img1, img2)

        mask_data1 = self.maskgen.generate_with_image_embedding(**self.embed_data1)
        mask_data2 = self.maskgen.generate_with_image_embedding(**self.embed_data2)
        mask_data1.filter((mask_data1['areas'] / (h * w)) < self.area_thresh)
        mask_data2.filter((mask_data2['areas'] / (h * w)) < self.area_thresh)

        return {
            't1_mask_data': mask_data1,
            't1_image_embedding': self.embed_data1['image_embedding'],
            't2_mask_data': mask_data2,
            't2_image_embedding': self.embed_data2['image_embedding'],
        }

    def bitemporal_match(self, t1_mask_data, t1_image_embedding, t2_mask_data, t2_image_embedding) -> MaskData:
        return NotImplementedError

    def forward(self, img1, img2):
        h, w = img1.shape[:2]

        if self.match_hist:
            img2 = match_histograms(image=img2, reference=img1, channel_axis=-1).astype(np.uint8)

        data = self.proposal(img1, img2)

        changemasks = self.bitemporal_match(**data)

        keep = batched_nms(
            changemasks["boxes"].float(),
            changemasks["iou_preds"],
            torch.zeros_like(changemasks["boxes"][:, 0]),
            iou_threshold=self.maskgen.box_nms_thresh,
        )
        changemasks.filter(keep)

        if len(changemasks['rles']) > 1000:
            scores = changemasks['change_confidence']
            sorted_scores, _ = torch.sort(scores, descending=True, stable=True)
            keep = scores > sorted_scores[1000]
            changemasks.filter(keep)

        return changemasks, data['t1_mask_data'], data['t2_mask_data']

    def to_eval_format_predictions(self, cmasks):
        boxes = cmasks['boxes']
        rle_masks = cmasks['rles']
        labels = torch.ones(boxes.size(0), dtype=torch.int64)
        scores = cmasks['change_confidence']
        predictions = {
            'boxes': boxes.to(torch.float32).cpu(),
            'scores': scores.cpu(),
            'labels': labels.cpu(),
            'masks': rle_masks
        }
        return predictions

    def __call__(self, img1, img2):
        cmasks, t1_masks, t2_masks = self.forward(img1, img2)
        predictions = self.to_eval_format_predictions(cmasks)
        self.clear_cached_embedding()
        return predictions
