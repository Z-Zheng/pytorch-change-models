# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import torch.nn.functional as F
from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask, MaskData
import copy
from skimage.filters.thresholding import threshold_otsu
from torchange.models.segment_any_change.base import SegmentAnyChange
import math
from torchvision.ops.boxes import batched_nms


def angle2cosine(a):
    assert 0 <= a <= 180
    return math.cos(a / 180 * math.pi)


def cosine2angle(c):
    assert -1 <= c <= 1
    return math.acos(c) * 180 / math.pi


class AnyChange(SegmentAnyChange):
    def __init__(self, model_type='vit_b', sam_checkpoint='./sam_weights/sam_vit_b_01ec64.pth'):
        super().__init__(model_type, sam_checkpoint)

        layernorm = self.sam.image_encoder.neck[3]
        w = layernorm.weight.data
        b = layernorm.bias.data
        w = w.reshape(w.size(0), 1, 1)
        b = b.reshape(b.size(0), 1, 1)

        self.inv_transform = lambda e: (e - b) / w

    def set_hyperparameters(
            self,
            change_confidence_threshold=155,
            auto_threshold=False,
            use_normalized_feature=True,
            area_thresh=0.8,
            match_hist=False,
            object_sim_thresh=60,
            bitemporal_match=True,
    ):
        self.area_thresh = area_thresh
        self.match_hist = match_hist

        self.change_confidence_threshold = change_confidence_threshold
        self.auto_threshold = auto_threshold
        self.use_normalized_feature = use_normalized_feature
        self.object_sim_thresh = object_sim_thresh
        self.use_bitemporal_match = bitemporal_match

    def bitemporal_match(self, t1_mask_data, t1_image_embedding, t2_mask_data, t2_image_embedding) -> MaskData:
        t1_img_embed = t1_image_embedding
        t2_img_embed = t2_image_embedding
        h, w = self.embed_data1['original_size']

        seq_img_embed = [t1_img_embed, t2_img_embed]
        seq_img_embed_data = [{'image_embedding': img_embed,
                               'original_size': self.embed_data1['original_size']}
                              for img_embed in seq_img_embed]

        seq_mask_data = [t1_mask_data, ]
        for img_embed_data in seq_img_embed_data[1:-1]:
            mask_data = self.maskgen.generate_with_image_embedding(**img_embed_data)
            mask_data.filter((mask_data['areas'] / (h * w)) < self.area_thresh)
            seq_mask_data.append(mask_data)

        seq_mask_data.append(t2_mask_data)

        if self.use_normalized_feature:
            t1_img_embed = self.inv_transform(t1_img_embed)
            t2_img_embed = self.inv_transform(t2_img_embed)

        t1_img_embed = F.interpolate(t1_img_embed, size=(h, w), mode='bilinear', align_corners=True)
        t2_img_embed = F.interpolate(t2_img_embed, size=(h, w), mode='bilinear', align_corners=True)
        t1_img_embed = t1_img_embed.squeeze_(0)
        t2_img_embed = t2_img_embed.squeeze_(0)

        if self.auto_threshold:
            cosv = -F.cosine_similarity(t1_img_embed, t2_img_embed, dim=0)
            cosv = cosv.reshape(-1).cpu().numpy()
            threshold = threshold_otsu(cosv, cosv.shape[0])
            self.change_confidence_threshold = cosine2angle(threshold)

        def _latent_match(mask_data, t1_img_embed, t2_img_embed):
            change_confidence = torch.zeros(len(mask_data['rles']), dtype=torch.float32, device=self.device)
            for i, rle in enumerate(mask_data['rles']):
                bmask = torch.from_numpy(rle_to_mask(rle)).to(self.device)
                t1_mask_embed = torch.mean(t1_img_embed[:, bmask], dim=-1)
                t2_mask_embed = torch.mean(t2_img_embed[:, bmask], dim=-1)
                score = -F.cosine_similarity(t1_mask_embed, t2_mask_embed, dim=0)
                change_confidence[i] += score

            keep = change_confidence > angle2cosine(self.change_confidence_threshold)

            mask_data = copy.deepcopy(mask_data)
            mask_data['change_confidence'] = change_confidence
            mask_data.filter(keep)
            return mask_data

        changemasks = MaskData()
        if self.use_bitemporal_match:
            for i in range(2):
                cmasks = _latent_match(seq_mask_data[i], t1_img_embed, t2_img_embed)
                changemasks.cat(cmasks)
        else:
            cmasks = _latent_match(seq_mask_data[1], t1_img_embed, t2_img_embed)
            changemasks.cat(cmasks)
        del cmasks

        return changemasks

    def single_point_q_mask(self, xy, img):
        point = np.array(xy).reshape(1, 2)

        embed_data = self.maskgen.image_encoder(img)

        embed_data.update(dict(points=point))
        mask_data = self.maskgen.embedding_point_to_mask(**embed_data)

        if len(mask_data['rles']) > 0:
            q_mask = torch.from_numpy(rle_to_mask(mask_data['rles'][0]))
        else:
            q_mask = torch.zeros(img.shape[0], img.shape[1])
        return q_mask

    def single_point_match(self, xy, temporal, img1, img2):
        h, w = img1.shape[:2]
        point = np.array(xy).reshape(1, 2)

        embed_data1 = self.maskgen.image_encoder(img1)
        embed_data2 = self.maskgen.image_encoder(img2)
        self.embed_data1 = embed_data1
        self.embed_data2 = embed_data2

        mask_data1 = self.maskgen.generate_with_image_embedding(**embed_data1)
        mask_data2 = self.maskgen.generate_with_image_embedding(**embed_data2)
        mask_data1.filter((mask_data1['areas'] / (h * w)) < self.area_thresh)
        mask_data2.filter((mask_data2['areas'] / (h * w)) < self.area_thresh)

        if temporal == 1:
            embed_data1.update(dict(points=point))
            mask_data = self.maskgen.embedding_point_to_mask(**embed_data1)
        elif temporal == 2:
            embed_data2.update(dict(points=point))
            mask_data = self.maskgen.embedding_point_to_mask(**embed_data2)
        else:
            raise ValueError

        q_area = mask_data['areas'][0]
        q_mask = torch.from_numpy(rle_to_mask(mask_data['rles'][0]))

        image_embedding1 = F.interpolate(embed_data1['image_embedding'], (h, w), mode='bilinear',
                                         align_corners=True).squeeze_(0)
        image_embedding2 = F.interpolate(embed_data2['image_embedding'], (h, w), mode='bilinear',
                                         align_corners=True).squeeze_(0)

        if temporal == 1:
            q_mask_features = torch.mean(image_embedding1[:, q_mask], dim=-1)
        elif temporal == 2:
            q_mask_features = torch.mean(image_embedding2[:, q_mask], dim=-1)
        else:
            raise ValueError

        cosmap1 = torch.cosine_similarity(q_mask_features.reshape(-1, 1, 1), image_embedding1, dim=0)
        cosmap2 = torch.cosine_similarity(q_mask_features.reshape(-1, 1, 1), image_embedding2, dim=0)

        obj_map1 = cosmap1 > angle2cosine(self.object_sim_thresh)
        obj_map2 = cosmap2 > angle2cosine(self.object_sim_thresh)

        def _filter_obj(obj_map, mask_data):
            mask_data = copy.deepcopy(mask_data)
            keep = (q_area * 0.25 < mask_data['areas']) & (mask_data['areas'] < q_area * 4)
            mask_data.filter(keep)
            keep = []
            for i, rle in enumerate(mask_data['rles']):
                mask = rle_to_mask(rle)
                keep.append(np.mean(obj_map[mask]) > 0.5)
            keep = torch.from_numpy(np.array(keep)).to(torch.bool)
            mask_data.filter(keep)
            return mask_data

        mask_data1 = _filter_obj(obj_map1.cpu().numpy(), mask_data1)
        mask_data2 = _filter_obj(obj_map2.cpu().numpy(), mask_data2)

        data = {
            't1_mask_data': mask_data1,
            't1_image_embedding': embed_data1['image_embedding'],
            't2_mask_data': mask_data2,
            't2_image_embedding': embed_data2['image_embedding'],
        }
        cmasks = self.bitemporal_match(**data)

        keep = batched_nms(
            cmasks["boxes"].float(),
            cmasks["iou_preds"],
            torch.zeros_like(cmasks["boxes"][:, 0]),
            iou_threshold=self.maskgen.box_nms_thresh,
        )
        cmasks.filter(keep)
        if len(cmasks['rles']) > 1000:
            scores = cmasks['change_confidence']
            sorted_scores, _ = torch.sort(scores, descending=True, stable=True)
            keep = scores > sorted_scores[1000]
            cmasks.filter(keep)

        return cmasks

    def multi_points_match(self, xyts, img1, img2):
        h, w = img1.shape[:2]

        embed_data1 = self.maskgen.image_encoder(img1)
        embed_data2 = self.maskgen.image_encoder(img2)
        self.embed_data1 = embed_data1
        self.embed_data2 = embed_data2

        mask_data1 = self.maskgen.generate_with_image_embedding(**embed_data1)
        mask_data2 = self.maskgen.generate_with_image_embedding(**embed_data2)
        mask_data1.filter((mask_data1['areas'] / (h * w)) < self.area_thresh)
        mask_data2.filter((mask_data2['areas'] / (h * w)) < self.area_thresh)

        image_embedding1 = F.interpolate(embed_data1['image_embedding'], (h, w), mode='bilinear',
                                         align_corners=True).squeeze_(0)
        image_embedding2 = F.interpolate(embed_data2['image_embedding'], (h, w), mode='bilinear',
                                         align_corners=True).squeeze_(0)

        q_areas = []
        q_features = []
        for xyt in xyts:
            t = xyt[-1]
            point = xyt[:2].reshape(1, 2)

            if t == 1:
                embed_data1.update(dict(points=point))
                mask_data = self.maskgen.embedding_point_to_mask(**embed_data1)
            elif t == 2:
                embed_data2.update(dict(points=point))
                mask_data = self.maskgen.embedding_point_to_mask(**embed_data2)
            else:
                raise ValueError

            q_area = mask_data['areas'][0]
            q_mask = torch.from_numpy(rle_to_mask(mask_data['rles'][0]))

            q_areas.append(q_area)

            if t == 1:
                q_mask_features = torch.mean(image_embedding1[:, q_mask], dim=-1)
            elif t == 2:
                q_mask_features = torch.mean(image_embedding2[:, q_mask], dim=-1)
            else:
                raise ValueError
            q_features.append(q_mask_features)

        q_area = sum(q_areas) / len(q_areas)
        q_mask_features = sum(q_features) / len(q_features)

        cosmap1 = torch.cosine_similarity(q_mask_features.reshape(-1, 1, 1), image_embedding1, dim=0)
        cosmap2 = torch.cosine_similarity(q_mask_features.reshape(-1, 1, 1), image_embedding2, dim=0)

        obj_map1 = cosmap1 > angle2cosine(self.object_sim_thresh)
        obj_map2 = cosmap2 > angle2cosine(self.object_sim_thresh)

        def _filter_obj(obj_map, mask_data):
            mask_data = copy.deepcopy(mask_data)
            keep = (q_area * 0.25 < mask_data['areas']) & (mask_data['areas'] < q_area * 4)
            mask_data.filter(keep)
            keep = []
            for i, rle in enumerate(mask_data['rles']):
                mask = rle_to_mask(rle)
                keep.append(np.mean(obj_map[mask]) > 0.5)
            keep = torch.from_numpy(np.array(keep)).to(torch.bool)
            mask_data.filter(keep)
            return mask_data

        mask_data1 = _filter_obj(obj_map1.cpu().numpy(), mask_data1)
        mask_data2 = _filter_obj(obj_map2.cpu().numpy(), mask_data2)

        data = {
            't1_mask_data': mask_data1,
            't1_image_embedding': embed_data1['image_embedding'],
            't2_mask_data': mask_data2,
            't2_image_embedding': embed_data2['image_embedding'],
        }
        cmasks = self.bitemporal_match(**data)

        keep = batched_nms(
            cmasks["boxes"].float(),
            cmasks["iou_preds"],
            torch.zeros_like(cmasks["boxes"][:, 0]),
            iou_threshold=self.maskgen.box_nms_thresh,
        )
        cmasks.filter(keep)
        if len(cmasks['rles']) > 1000:
            scores = cmasks['change_confidence']
            sorted_scores, _ = torch.sort(scores, descending=True, stable=True)
            keep = scores > sorted_scores[1000]
            cmasks.filter(keep)

        return cmasks
