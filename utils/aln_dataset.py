from PIL import Image
import os
import csv
import numpy as np
import cv2
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, RandomCrop
import torch
import random

class ALNDatasetGeom(Dataset):
    def __init__(
        self,
        input_folder,
        target_folder,
        geom_folder,
        resize_width_to=None,
        patch_size=None,
        patch_height=None,
        patch_width=None,
        crop_mode="random",
        crop_x=-1,
        crop_y=-1,
        filter_of_images=None,
        dino_folder=None,
        dino_suffix="_dino32",
        dino_dim=1024,
        dino_crop_manifest=None,
        dino_ref_width=None,
        dino_ref_height=None,
        enable_aug=False,
        aug_hflip_prob=0.5,
        aug_vflip_prob=0.5,
        aug_rotate90=True,
        disable_aug_if_fullres=True,
        allow_fallback_normal=False,
        dino_stages=(6, 12, 18, 24),
        return_aux_hist=False,
    ):
        super(ALNDatasetGeom, self).__init__()
        self.input_folder = input_folder
        self.geom_folder = geom_folder
        self.target_folder = target_folder
        self.resize_width_to = resize_width_to
        self.patch_size = patch_size
        self.patch_height = int(patch_height) if patch_height is not None and int(patch_height) > 0 else None
        self.patch_width = int(patch_width) if patch_width is not None and int(patch_width) > 0 else None
        self.crop_mode = str(crop_mode).lower() if crop_mode is not None else "random"
        self.crop_x = int(crop_x) if crop_x is not None else -1
        self.crop_y = int(crop_y) if crop_y is not None else -1
        self.filter_of_images = filter_of_images
        self.dino_folder = dino_folder
        self.dino_suffix = dino_suffix
        self.dino_dim = dino_dim
        self.dino_crop_manifest = dino_crop_manifest if dino_crop_manifest else None
        self.dino_ref_width = int(dino_ref_width) if dino_ref_width else 0
        self.dino_ref_height = int(dino_ref_height) if dino_ref_height else 0
        self.enable_aug = bool(enable_aug)
        self.aug_hflip_prob = float(aug_hflip_prob)
        self.aug_vflip_prob = float(aug_vflip_prob)
        self.aug_rotate90 = bool(aug_rotate90)
        self.disable_aug_if_fullres = bool(disable_aug_if_fullres)
        self.allow_fallback_normal = allow_fallback_normal
        self.dino_stages = tuple(dino_stages)
        self.return_aux_hist = bool(return_aux_hist)
        if self.dino_folder is None:
            raise ValueError("dino_folder must be provided.")
        if self.crop_mode not in ("random", "fixed", "none"):
            raise ValueError(f"Unsupported crop_mode={self.crop_mode}. Use random/fixed/none.")
        self._dino_crop_map = self._load_dino_crop_map(self.dino_crop_manifest)
        self._init_paths()

    def _load_dino_crop_map(self, manifest_path):
        if manifest_path is None:
            return {}
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"DINO crop manifest not found: {manifest_path}")
        crop_map = {}
        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Preferred schema from crop_manifest_perimage.csv
                input_name = row.get("input_name")
                if input_name:
                    key_name = os.path.basename(input_name)
                    key_stem = os.path.splitext(key_name)[0]
                    crop_map[key_name] = (
                        int(row["x0"]),
                        int(row["y0"]),
                        int(row["crop_w"]),
                        int(row["crop_h"]),
                    )
                    crop_map[key_stem] = crop_map[key_name]
                    continue

                # Backward-compatible schema: rel_path starts with input/
                rel_path = row.get("rel_path")
                if rel_path and rel_path.startswith("input/"):
                    key_name = os.path.basename(rel_path)
                    key_stem = os.path.splitext(key_name)[0]
                    crop_map[key_name] = (
                        int(row["x0"]),
                        int(row["y0"]),
                        int(row["crop_w"]),
                        int(row["crop_h"]),
                    )
                    crop_map[key_stem] = crop_map[key_name]
        print(f"Loaded DINO crop manifest: {manifest_path} ({len(crop_map)} keys)")
        return crop_map

    def _init_paths(self):
        self.image_paths = []
        for input_path in sorted(glob(os.path.join(self.input_folder, "*.png"))):
            fname = os.path.basename(input_path)  # e.g. 10_10_IN.png
            stem = os.path.splitext(fname)[0]
            base_id = fname.split("_")[0]
            use_multi_stage = all(
                os.path.isdir(os.path.join(self.dino_folder, f"feat{sid}")) for sid in self.dino_stages
            )
            dino_path = os.path.join(self.dino_folder, f"{stem}{self.dino_suffix}.npy")
            dino_paths_multi = {
                sid: os.path.join(self.dino_folder, f"feat{sid}", f"{stem}{self.dino_suffix}.npy")
                for sid in self.dino_stages
            }

            target_candidates = []
            # Per-image paired GT (e.g. 10_14_IN.png -> 10_14_GT.png)
            if stem.endswith("_IN"):
                target_candidates.append(os.path.join(self.target_folder, f"{stem[:-3]}_GT.png"))
            target_candidates.append(os.path.join(self.target_folder, f"{stem}_GT.png"))
            # Legacy shared GT by scene id (e.g. 10_14_IN.png -> 10_GT.png)
            target_candidates.append(os.path.join(self.target_folder, f"{base_id}_GT.png"))

            target_path = next((p for p in target_candidates if os.path.exists(p)), None)
            if target_path is None:
                raise FileNotFoundError(
                    f"Target file not found for input {fname}. Tried: {target_candidates}"
                )
            if use_multi_stage:
                missing = [p for p in dino_paths_multi.values() if not os.path.exists(p)]
                if missing:
                    raise FileNotFoundError(f"Missing multi-stage DINO token(s), e.g. {missing[0]}")
            else:
                if not os.path.exists(dino_path):
                    raise FileNotFoundError(f"DINO token not found: {dino_path}")

            item = {
                'input': input_path,
                'target': target_path,
            }
            if use_multi_stage:
                item['dino_multi'] = dino_paths_multi
            else:
                item['dino'] = dino_path

            crop_key_name = fname
            crop_key_stem = stem
            if crop_key_name in self._dino_crop_map:
                item['dino_crop'] = self._dino_crop_map[crop_key_name]
            elif crop_key_stem in self._dino_crop_map:
                item['dino_crop'] = self._dino_crop_map[crop_key_stem]
            self.image_paths.append(item)

        print(f"Found {len(self.image_paths)} image triplets (input, dino, gt)")
    
    def _generate_fallback_normal(self, input_path, output_path):
        """使用 Sobel 边界检测生成备用法线图"""
        import cv2
        from PIL import Image
        
        # 读取输入图像转为灰度
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Sobel 边界检测
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
        
        # 构造法向量
        normal = np.dstack((-gx, -gy, np.ones_like(img)))
        
        # 归一化
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (norm + 1e-10)
        
        # 保存
        np.save(output_path, normal.astype(np.float32))

        print(f"Found {len(self.image_paths)} image triplets (input, normal, gt)")

    @staticmethod
    def _calc_histogram_bgr(img_bgr):
        area = float(max(img_bgr.shape[0] * img_bgr.shape[1], 1))
        hist = []
        for c in range(3):
            h = cv2.calcHist([img_bgr], [c], None, [256], [0, 256]).flatten().astype(np.float32)
            hist.append(h / area)
        return np.stack(hist, axis=0)

    def _compute_hist_features(self, img_tensor):
        rgb = img_tensor.permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        rgb_hist = self._calc_histogram_bgr(bgr)
        lab_hist = self._calc_histogram_bgr(lab)
        return (
            torch.from_numpy(rgb_hist).float(),
            torch.from_numpy(lab_hist).float(),
        )

    def _crop_dino_by_pixel_box(self, dino_img, x0, y0, w, h):
        if self.dino_ref_width <= 0 or self.dino_ref_height <= 0:
            raise ValueError(
                "dino_ref_width/dino_ref_height must be set when using dino_crop_manifest"
            )
        tok_h, tok_w = dino_img.shape[-2], dino_img.shape[-1]
        tj = int(round(x0 / float(self.dino_ref_width) * tok_w))
        ti = int(round(y0 / float(self.dino_ref_height) * tok_h))
        tw = int(round(w / float(self.dino_ref_width) * tok_w))
        th = int(round(h / float(self.dino_ref_height) * tok_h))
        th = max(1, min(th, tok_h))
        tw = max(1, min(tw, tok_w))
        ti = max(0, min(ti, tok_h - th))
        tj = max(0, min(tj, tok_w - tw))
        if dino_img.ndim == 3:
            return dino_img[:, ti:ti + th, tj:tj + tw]
        if dino_img.ndim == 4:
            return dino_img[:, :, ti:ti + th, tj:tj + tw]
        raise ValueError(f"Unexpected dino tensor ndim: {dino_img.ndim}")

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        paths = self.image_paths[idx]
        input_img = TF.to_tensor(Image.open(paths['input']).convert('RGB'))
        target_img = TF.to_tensor(Image.open(paths['target']).convert('RGB'))

        # ---------------- 讀入 dino 特徵 ----------------
        if 'dino_multi' in paths:
            dino_stages = []
            for sid in self.dino_stages:
                spath = paths['dino_multi'][sid]
                dino_np = np.load(spath)
                if dino_np.ndim != 3:
                    raise ValueError(f"DINO token must be 3D, got {dino_np.shape} @ {spath}")
                if dino_np.shape[0] == self.dino_dim:
                    pass
                elif dino_np.shape[-1] == self.dino_dim:
                    dino_np = np.transpose(dino_np, (2, 0, 1))
                else:
                    raise ValueError(f"Unexpected DINO shape: {dino_np.shape} @ {spath}")
                dino_stages.append(torch.from_numpy(dino_np.astype(np.float32)))
            # [4, C, H, W]
            dino_img = torch.stack(dino_stages, dim=0)
        else:
            dino_np = np.load(paths['dino'])
            if dino_np.ndim != 3:
                raise ValueError(f"DINO token must be 3D, got {dino_np.shape} @ {paths['dino']}")
            if dino_np.shape[0] == self.dino_dim:
                pass
            elif dino_np.shape[-1] == self.dino_dim:
                dino_np = np.transpose(dino_np, (2, 0, 1))
            else:
                raise ValueError(f"Unexpected DINO shape: {dino_np.shape} @ {paths['dino']}")
            dino_img = torch.from_numpy(dino_np.astype(np.float32))

        # ---------------- Optional offline-crop alignment for pre-cropped datasets ----------------
        if 'dino_crop' in paths:
            x0, y0, cw, ch = paths['dino_crop']
            dino_img = self._crop_dino_by_pixel_box(dino_img, x0, y0, cw, ch)

        # ---------------- Resize & Augment ----------------
        if self.resize_width_to is not None and int(self.resize_width_to) > 0:
            new_h = int((input_img.shape[1] * self.resize_width_to) / input_img.shape[2])
            input_img = TF.resize(input_img, (new_h, self.resize_width_to))
            target_img = TF.resize(target_img, (new_h, self.resize_width_to))

        crop_h = self.patch_height if self.patch_height is not None else self.patch_size
        crop_w = self.patch_width if self.patch_width is not None else self.patch_size

        if self.enable_aug:
            aug_allowed = True
            if self.disable_aug_if_fullres and crop_h is not None and crop_w is not None:
                img_h, img_w = input_img.shape[-2], input_img.shape[-1]
                if crop_h >= img_h and crop_w >= img_w:
                    aug_allowed = False
            if aug_allowed:
                do_hflip = random.random() < self.aug_hflip_prob
                do_vflip = random.random() < self.aug_vflip_prob
                # For rectangular source images, 90/270 rotation swaps H/W and can
                # produce mixed token crop shapes in the same batch when DINO token
                # grids do not preserve the same aspect ratio behavior.
                if self.aug_rotate90:
                    img_h, img_w = input_img.shape[-2], input_img.shape[-1]
                    if int(img_h) != int(img_w):
                        rot_k = random.choice([0, 2])  # keep orientation (0° / 180°)
                    elif crop_h is not None and crop_w is not None and int(crop_h) != int(crop_w):
                        rot_k = random.choice([0, 2])  # keep orientation (0° / 180°)
                    else:
                        rot_k = random.choice([0, 1, 2, 3])
                else:
                    rot_k = 0
                input_img = self._apply_spatial_ops(input_img, do_hflip, do_vflip, rot_k)
                target_img = self._apply_spatial_ops(target_img, do_hflip, do_vflip, rot_k)
                dino_img = self._apply_spatial_ops(dino_img, do_hflip, do_vflip, rot_k)

        # ---------------- Random crop ----------------
        if crop_h is not None and crop_w is not None and self.crop_mode != "none":
            pre_crop_h, pre_crop_w = input_img.shape[1], input_img.shape[2]
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
                y0, x0, h, w = RandomCrop.get_params(input_img, output_size=(h, w))
            input_img = TF.crop(input_img, y0, x0, h, w)
            target_img = TF.crop(target_img, y0, x0, h, w)

            # Align token crop with image crop in token-grid coordinates.
            tok_h, tok_w = dino_img.shape[-2], dino_img.shape[-1]
            ti = int(round(y0 / max(pre_crop_h, 1) * tok_h))
            tj = int(round(x0 / max(pre_crop_w, 1) * tok_w))
            th = int(round(h / max(pre_crop_h, 1) * tok_h))
            tw = int(round(w / max(pre_crop_w, 1) * tok_w))
            th = max(1, min(th, tok_h))
            tw = max(1, min(tw, tok_w))
            ti = max(0, min(ti, tok_h - th))
            tj = max(0, min(tj, tok_w - tw))
            if dino_img.ndim == 3:
                dino_img = dino_img[:, ti:ti + th, tj:tj + tw]
            elif dino_img.ndim == 4:
                dino_img = dino_img[:, :, ti:ti + th, tj:tj + tw]
            else:
                raise ValueError(f"Unexpected dino tensor ndim: {dino_img.ndim}")

        sample_stem = os.path.splitext(os.path.basename(paths['input']))[0]
        if self.return_aux_hist:
            input_hist, input_lab_hist = self._compute_hist_features(input_img)
            target_hist, target_lab_hist = self._compute_hist_features(target_img)
            return (
                [sample_stem, 0],
                input_img,
                dino_img,
                target_img,
                input_hist,
                target_hist,
                input_lab_hist,
                target_lab_hist,
            )
        return [sample_stem, 0], input_img, dino_img, target_img

    
