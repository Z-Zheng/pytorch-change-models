import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用 GPU 2
import numpy as np
import torch
from skimage.exposure import match_histograms
from torchvision.ops.boxes import batched_nms
from safetensors.torch import load_file
import torch.nn.functional as F
from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask, MaskData
import copy
from skimage.filters.thresholding import threshold_otsu
import math

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def angle2cosine(a):
    assert 0 <= a <= 180
    return math.cos(a / 180 * math.pi)

def cosine2angle(c):
    assert -1 <= c <= 1
    return math.acos(c) * 180 / math.pi

class SegmentAnyChange:
    def __init__(self, model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml", sam2_checkpoint="/slow_disk/ccl/codes/pytorch-change-models/sam2.1_hiera_tiny.pt"):

        device = device = torch.device("cuda")
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sam2 = sam2.to(self.device)

        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)
        # self.mask_generator = SAM2AutomaticMaskGenerator(sam2,
        #                                             points_per_side=64,
        #                                             points_per_batch=128,
        #                                             pred_iou_thresh=0.7,
        #                                             stability_score_thresh=0.92,
        #                                             stability_score_offset=0.7,
        #                                             crop_n_layers=1,
        #                                             box_nms_thresh=0.7,
        #                                             crop_n_points_downscale_factor=2,
        #                                             min_mask_region_area=25,
        #                                             use_m2m=False,
        #                                             )
        # self.mask_generator = SAM2AutomaticMaskGenerator(sam2)

        self.set_hyperparameters()

        self.embed_data1 = None
        self.embed_data2 = None

        # layernorm = self.sam.image_encoder.neck[3]
        # w = layernorm.weight.data
        # b = layernorm.bias.data
        # w = w.reshape(w.size(0), 1, 1)
        # b = b.reshape(b.size(0), 1, 1)
        #
        # self.inv_transform = lambda e: (e - b) / w

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

        # seq_img_embed = [t1_img_embed, t2_img_embed]
        # seq_img_embed_data = [{'image_embedding': img_embed,
        #                        'original_size': self.embed_data1['original_size']}
        #                       for img_embed in seq_img_embed]

        seq_mask_data = [t1_mask_data, ]
        # for img_embed_data in seq_img_embed_data[1:-1]:
        #     mask_data = self.maskgen.generate_with_image_embedding(**img_embed_data)
        #     mask_data.filter((mask_data['areas'] / (h * w)) < self.area_thresh)
        #     seq_mask_data.append(mask_data)

        seq_mask_data.append(t2_mask_data)

        # if self.use_normalized_feature:
        #     t1_img_embed = self.inv_transform(t1_img_embed)
        #     t2_img_embed = self.inv_transform(t2_img_embed)

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

    def extract_image_embedding(self, img1, img2):
        self.mask_generator.predictor.set_image(img1)
        self.embed_data1 = {
            'image_embedding': self.mask_generator.predictor.get_image_embedding(),
            'original_size': img1.shape[:2],
        }
        self.mask_generator.predictor.set_image(img2)
        self.embed_data2 = {
            'image_embedding': self.mask_generator.predictor.get_image_embedding(),
            'original_size': img2.shape[:2],
        }
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
        self.mask_generator.predictor.input_size = None
        self.mask_generator.predictor.original_size = None

    def proposal(self, img1, img2):
        h, w = img1.shape[:2]
        if self.embed_data1 is None:
            self.extract_image_embedding(img1, img2)

        mask_data1 = self.mask_generator._generate_masks(img1)
        mask_data2 = self.mask_generator._generate_masks(img2)
        # mask_data1.filter((mask_data1['areas'] / (h * w)) < self.area_thresh)
        # mask_data2.filter((mask_data2['areas'] / (h * w)) < self.area_thresh)

        return {
            't1_mask_data': mask_data1,
            't1_image_embedding': self.embed_data1['image_embedding'],
            't2_mask_data': mask_data2,
            't2_image_embedding': self.embed_data2['image_embedding'],
        }

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
            iou_threshold=0.7,
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


if __name__ == '__main__':
    import os
    import glob
    import cv2
    from skimage.io import imread
    from torchange.models.segment_any_change import show_change_masks_sam2
    import matplotlib.pyplot as plt
    from tqdm import tqdm  # Import tqdm for the progress bar

    from ccl_utils import crop_images

    m = SegmentAnyChange("configs/sam2.1/sam2.1_hiera_l.yaml", "/slow_disk/ccl/codes/finetune_sam2.1/sam2_git/checkpoints/sam2.1_hiera_large.pt")
    m.set_hyperparameters(change_confidence_threshold=145, use_normalized_feature=True, bitemporal_match=True)

    # prefix = '/mnt/mnt108_hdd/ccl/data/2/'
    # prefix = '/mnt/mnt108_hdd/ccl/data/pcb1/'
    # prefix = '/slow_disk/ccl/data/两期变化/余姚/余姚一段运河_1vs余姚一段运河_2/原图/'
    # prefix = '/slow_disk/ccl/cd_test/'
    # prefix = '/mnt/mnt108_hdd/ccl/data/20250408/'
    prefix = '/home/ccl/Pictures/sam2/'
    folder_A = os.path.join(prefix, 'A')
    folder_B = os.path.join(prefix, 'B')
    folder_C = os.path.join(prefix, 'C')

    # Create folder C if it doesn't exist
    os.makedirs(folder_C, exist_ok=True)

    # Read all images from folder A
    image_files_A = glob.glob(os.path.join(folder_A, '*.png'))

    # Use tqdm to create a progress bar
    for img_path_A in tqdm(image_files_A, desc="Processing images", unit="image"):
        # Extract the filename without the directory
        filename = os.path.basename(img_path_A)
        img_path_B = os.path.join(folder_B, filename)

        # Check if the corresponding image exists in folder B
        if os.path.exists(img_path_B):
            # Read images
            img1 = imread(img_path_A)
            img2 = imread(img_path_B)

            # img1, img2 = crop_images(img1, img2, 2000, 400, 512)

            m.clear_cached_embedding()

            # Process images to find change masks
            changemasks, _, _ = m.forward(img1, img2)

            # Show change masks and save the result
            fig, axes, mask_only = show_change_masks_sam2(img1, img2, changemasks)

            if mask_only is None:
                mask_only = np.zeros((1024, 1024))

            # Save the change mask image to folder C
            plt.savefig(os.path.join(folder_C, filename))

            plt.close(fig)  # Close the figure to free memory

            filename = filename.split('.')
            filename_mask = filename[0] + "_mask." + filename[1]
            cv2.imwrite(os.path.join(folder_C, filename_mask), mask_only * 255)

        else:
            print(f"Warning: Corresponding image not found for {filename} in folder B.")