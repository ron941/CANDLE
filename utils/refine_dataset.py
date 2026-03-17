import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, RandomCrop


class RefineDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        patch_size=0,
        patch_height=0,
        patch_width=0,
        crop_mode="random",
        crop_x=-1,
        crop_y=-1,
        enable_aug=False,
        aug_hflip_prob=0.5,
        aug_vflip_prob=0.5,
        aug_rotate90=True,
        disable_aug_if_fullres=True,
        vit_ext=".npy",
    ):
        self.root = root
        self.split = split
        self.patch_size = int(patch_size) if patch_size else 0
        self.patch_height = int(patch_height) if patch_height else 0
        self.patch_width = int(patch_width) if patch_width else 0
        self.crop_mode = str(crop_mode).lower()
        self.crop_x = int(crop_x)
        self.crop_y = int(crop_y)
        self.enable_aug = bool(enable_aug)
        self.aug_hflip_prob = float(aug_hflip_prob)
        self.aug_vflip_prob = float(aug_vflip_prob)
        self.aug_rotate90 = bool(aug_rotate90)
        self.disable_aug_if_fullres = bool(disable_aug_if_fullres)
        self.vit_ext = vit_ext

        self.input_dir = os.path.join(root, split, "input")
        self.vit_dir = os.path.join(root, split, "vit")
        self.gt_dir = os.path.join(root, split, "gt")
        self.samples = self._scan()

    def _scan(self):
        samples = []
        vit_files = sorted(glob(os.path.join(self.vit_dir, f"*{self.vit_ext}")))
        if not vit_files:
            raise FileNotFoundError(f"No cached VIT files found in {self.vit_dir}")
        for vit_path in vit_files:
            name = os.path.basename(vit_path)
            if not name.endswith(f"_vit{self.vit_ext}"):
                continue
            sample_id = name[: -len(f"_vit{self.vit_ext}")]
            input_path = os.path.join(self.input_dir, f"{sample_id}_input.png")
            gt_path = os.path.join(self.gt_dir, f"{sample_id}_gt.png")
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Missing cached input for {sample_id}: {input_path}")
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Missing cached GT for {sample_id}: {gt_path}")
            samples.append({
                "id": sample_id,
                "input": input_path,
                "vit": vit_path,
                "gt": gt_path,
            })
        if not samples:
            raise RuntimeError(f"No valid refine samples found under {self.root}/{self.split}")
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _apply_spatial_ops(tensor, hflip=False, vflip=False, rot_k=0):
        out = tensor
        if hflip:
            out = torch.flip(out, dims=[-1])
        if vflip:
            out = torch.flip(out, dims=[-2])
        if rot_k % 4 != 0:
            out = torch.rot90(out, k=rot_k, dims=[-2, -1])
        return out

    def _load_vit(self, path):
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[0] == 3:
            tensor = torch.from_numpy(arr.astype(np.float32))
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)).astype(np.float32))
        else:
            raise ValueError(f"Unexpected cached vit shape {arr.shape} @ {path}")
        return tensor

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vit_img = self._load_vit(sample["vit"])
        gt_img = TF.to_tensor(Image.open(sample["gt"]).convert("RGB"))
        try:
            input_img = TF.to_tensor(Image.open(sample["input"]).convert("RGB"))
        except (OSError, ValueError) as e:
            # Cached input PNG is only used for visualization. If a cache write was interrupted,
            # keep training alive by falling back to the cached VIT image.
            print(f"[RefineDataset] fallback input for {sample['id']} due to read error: {e}")
            input_img = vit_img.clone()

        crop_h = self.patch_height if self.patch_height > 0 else (self.patch_size if self.patch_size > 0 else None)
        crop_w = self.patch_width if self.patch_width > 0 else (self.patch_size if self.patch_size > 0 else None)

        if self.enable_aug:
            aug_allowed = True
            if self.disable_aug_if_fullres and crop_h is not None and crop_w is not None:
                img_h, img_w = vit_img.shape[-2], vit_img.shape[-1]
                if crop_h >= img_h and crop_w >= img_w:
                    aug_allowed = False
            if aug_allowed:
                do_hflip = random.random() < self.aug_hflip_prob
                do_vflip = random.random() < self.aug_vflip_prob
                if self.aug_rotate90:
                    img_h, img_w = vit_img.shape[-2], vit_img.shape[-1]
                    if int(img_h) != int(img_w):
                        rot_k = random.choice([0, 2])
                    elif crop_h is not None and crop_w is not None and int(crop_h) != int(crop_w):
                        rot_k = random.choice([0, 2])
                    else:
                        rot_k = random.choice([0, 1, 2, 3])
                else:
                    rot_k = 0
                input_img = self._apply_spatial_ops(input_img, do_hflip, do_vflip, rot_k)
                vit_img = self._apply_spatial_ops(vit_img, do_hflip, do_vflip, rot_k)
                gt_img = self._apply_spatial_ops(gt_img, do_hflip, do_vflip, rot_k)

        if crop_h is not None and crop_w is not None and self.crop_mode != "none":
            pre_crop_h, pre_crop_w = vit_img.shape[-2], vit_img.shape[-1]
            h = min(int(crop_h), pre_crop_h)
            w = min(int(crop_w), pre_crop_w)
            if self.crop_mode == "fixed":
                if self.crop_x >= 0 and self.crop_y >= 0:
                    x0 = int(self.crop_x)
                    y0 = int(self.crop_y)
                else:
                    x0 = max((pre_crop_w - w) // 2, 0)
                    y0 = max((pre_crop_h - h) // 2, 0)
                x0 = max(0, min(x0, pre_crop_w - w))
                y0 = max(0, min(y0, pre_crop_h - h))
            else:
                y0, x0, h, w = RandomCrop.get_params(vit_img, output_size=(h, w))
            input_img = TF.crop(input_img, y0, x0, h, w)
            vit_img = TF.crop(vit_img, y0, x0, h, w)
            gt_img = TF.crop(gt_img, y0, x0, h, w)

        return {
            "id": sample["id"],
            "input": input_img,
            "vit": vit_img,
            "gt": gt_img,
        }
