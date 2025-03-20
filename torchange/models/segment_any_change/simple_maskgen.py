# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
from .segment_anything import SamPredictor
from .segment_anything.utils.amg import build_point_grid
from .segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    calculate_stability_score,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from torchvision.ops.boxes import batched_nms

__all__ = [
    'SimpleMaskGenerator',
]


class SimpleMaskGenerator:
    def __init__(
            self,
            model,
            points_per_side=32,
            points_per_batch: int = 64,
            pred_iou_thresh: float = 0.5,
            stability_score_thresh: float = 0.95,
            stability_score_offset: float = 1.0,
            box_nms_thresh: float = 0.7,
            point_grids=None,
            min_mask_region_area: int = 0,
    ):
        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area

        assert (points_per_side is None) != (
                point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_point_grid(points_per_side)
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

    @torch.no_grad()
    def image_encoder(self, image):
        orig_size = image.shape[:2]
        self.predictor.set_image(image)
        return {
            'image_embedding': self.predictor.get_image_embedding(),
            'original_size': orig_size,
        }

    @torch.no_grad()
    def generate_with_image_embedding(self, image_embedding, original_size):
        im_h, im_w = original_size
        # Get points for this crop
        points_scale = np.array(original_size)[None, ::-1]
        points_for_image = self.point_grids * points_scale

        data = MaskData()
        crop_box = [0, 0, im_w, im_h]
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(image_embedding, points, original_size, crop_box, original_size)

            data.cat(batch_data)
            del batch_data

        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            data.to_numpy()
            data = self.postprocess_small_regions(
                data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        data['areas'] = np.asarray([area_from_rle(rle) for rle in data['rles']])
        if isinstance(data['boxes'], torch.Tensor):
            data['areas'] = torch.from_numpy(data['areas'])

        return data

    @torch.no_grad()
    def generate_with_points(self, image, points):
        image_embedding_data = self.image_encoder(image)
        image_embedding_data.update(dict(points=points))
        return self.embedding_point_to_mask(**image_embedding_data)

    @torch.no_grad()
    def embedding_point_to_mask(self, image_embedding, original_size, points):
        h, w = original_size
        crop_box = [0, 0, w, h]
        data = self._process_batch(image_embedding, points, (h, w), crop_box, (h, w))

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        data['areas'] = np.asarray([area_from_rle(rle) for rle in data['rles']])
        if isinstance(data['boxes'], torch.Tensor):
            data['areas'] = torch.from_numpy(data['areas'])
        return data

    @torch.no_grad()
    def generate(self, image, mask_output_mode='rle'):
        image_embedding_data = self.image_encoder(image)
        data = self.generate_with_image_embedding(**image_embedding_data)

        if mask_output_mode == 'rle':
            data["segmentations"] = data["rles"]
        elif mask_output_mode == 'binary_mask':
            data["segmentations"] = np.stack([rle_to_mask(rle) for rle in data["rles"]], axis=0)
        else:
            raise ValueError

        if isinstance(data['boxes'], torch.Tensor):
            if mask_output_mode == 'binary_mask':
                data["segmentations"] = torch.from_numpy(data["segmentations"])

        orig_size = image_embedding_data['original_size']
        image_embedding = image_embedding_data['image_embedding']
        image_embedding = F.interpolate(image_embedding, size=orig_size, mode='bilinear', align_corners=True)
        image_embedding = image_embedding.squeeze_(0)
        return {
            'mask_data': data,
            'image_embedding': image_embedding
        }

    def _process_batch(
            self,
            image_embedding,
            points: np.ndarray,
            im_size,
            crop_box,
            orig_size,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predict_torch(
            self.predictor,
            image_embedding,
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
            mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def predict_torch(
            self, predictor,
            image_embedding,
            point_coords,
            point_labels,
            boxes=None,
            mask_input=None,
            multimask_output: bool = True,
            return_logits: bool = False,
    ):
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = predictor.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=predictor.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = predictor.model.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)

        if not return_logits:
            masks = masks > predictor.model.mask_threshold

        return masks, iou_predictions, low_res_masks
