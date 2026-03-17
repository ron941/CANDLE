import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract multi-layer DINOv3 patch tokens and save as (Htok, Wtok, C) arrays"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/raid/ron/ALN_768/dataset/CL3AN_id20/train/input",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/raid/ron/ALN_768/dataset/CL3AN_id20/train/dino_tokens32",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_IN.png",
        help="Glob pattern for input images",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=0,
        help="Legacy square resize target. If >0, target_w/target_h will both use this value.",
    )
    parser.add_argument(
        "--target_w",
        type=int,
        default=0,
        help="Image resize target width",
    )
    parser.add_argument(
        "--target_h",
        type=int,
        default=0,
        help="Image resize target height",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=16,
        help="Model patch size for grid reshape",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--save_fp32",
        action="store_true",
        help="Save float32 instead of float16",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="6,12,18,24",
        help="Comma-separated transformer block indices to save, e.g. 6,12,18,24",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Save only one final 4-stage comparison image after extraction",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for DINO forward pass",
    )
    return parser.parse_args()


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - x_min) / (x_max - x_min)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def feat_to_vis_image(feat_hwc: np.ndarray) -> Image.Image:
    # Use cosine similarity to global mean token as a stable scalar map.
    tokens = feat_hwc.reshape(-1, feat_hwc.shape[-1]).astype(np.float32)
    mean_token = tokens.mean(axis=0, keepdims=True)
    sim = tokens @ mean_token.T
    sim = sim.reshape(feat_hwc.shape[0], feat_hwc.shape[1])
    sim_u8 = normalize_to_uint8(sim)
    return Image.fromarray(sim_u8, mode="L")


def save_layer_grid(vis_images: list[Image.Image], out_path: str) -> None:
    n = len(vis_images)
    if n == 0:
        return
    cell_w, cell_h = vis_images[0].size
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    canvas = Image.new("L", (cols * cell_w, rows * cell_h))
    for idx, img in enumerate(vis_images):
        r = idx // cols
        c = idx % cols
        canvas.paste(img, (c * cell_w, r * cell_h))
    canvas.save(out_path)


def main():
    args = parse_args()

    if args.target_size and args.target_size > 0:
        target_w = int(args.target_size)
        target_h = int(args.target_size)
    else:
        target_w = int(args.target_w)
        target_h = int(args.target_h)
    if target_w <= 0 or target_h <= 0:
        raise ValueError("Please set either --target_size (>0) or both --target_w and --target_h (>0).")
    if target_w % args.patch != 0 or target_h % args.patch != 0:
        raise ValueError("--target_w and --target_h must be divisible by --patch")

    grid_w = target_w // args.patch
    grid_h = target_h // args.patch
    use_fp16_save = not args.save_fp32
    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if not layer_ids:
        raise ValueError("--layers cannot be empty")
    if any(x <= 0 for x in layer_ids):
        raise ValueError("--layers must be positive integers")
    if len(layer_ids) != 4:
        raise ValueError("For 4-stage extraction, --layers must contain exactly 4 indices")

    os.makedirs(args.out_dir, exist_ok=True)
    stage_dirs = {}
    for lid in layer_ids:
        d = os.path.join(args.out_dir, f"feat{lid}")
        os.makedirs(d, exist_ok=True)
        stage_dirs[lid] = d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("loading:", args.model_name)
    processor = AutoImageProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()
    if hasattr(Image, "Resampling"):
        bicubic = Image.Resampling.BICUBIC
    else:
        bicubic = Image.BICUBIC

    def process_batch(img_paths):
        work_items = []
        for img_path in img_paths:
            base = os.path.basename(img_path)
            stem = os.path.splitext(base)[0]
            out_paths = {
                lid: os.path.join(stage_dirs[lid], f"{stem}_dino32.npy") for lid in layer_ids
            }

            if (not args.overwrite) and all(os.path.exists(p) for p in out_paths.values()):
                continue
            work_items.append((img_path, stem, out_paths))

        if not work_items:
            return {"ok": [], "skip": len(img_paths), "fail": []}

        images = []
        for img_path, _, _ in work_items:
            img = Image.open(img_path).convert("RGB")
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.BICUBIC)
            images.append(img)

        inputs = processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states")

            max_layer = len(hidden_states) - 1
            bad_layers = [x for x in layer_ids if x > max_layer]
            if bad_layers:
                raise RuntimeError(
                    f"Requested layers {bad_layers} exceed max available layer index {max_layer}"
                )

            # DINOv3 includes special tokens (CLS + register tokens) before patch tokens.
            all_tokens = hidden_states[max_layer]
            expected_patches = grid_h * grid_w
            num_special = all_tokens.shape[1] - expected_patches
            if num_special < 0:
                raise RuntimeError(
                    f"Model returned fewer tokens than expected patches: {tuple(all_tokens.shape)}"
                )

            ok_items = []
            for bi, (_, stem, out_paths) in enumerate(work_items):
                feat_by_layer = {}
                for lid in layer_ids:
                    tokens = hidden_states[lid][bi : bi + 1, num_special:, :]
                    tokens = F.normalize(tokens, dim=-1)
                    feat = tokens[0].detach().cpu().numpy()

                    if feat.shape[0] != grid_h * grid_w:
                        raise RuntimeError(f"Token count mismatch at layer {lid}: {feat.shape}")

                    feat = feat.reshape(grid_h, grid_w, -1)
                    feat = feat.astype(np.float16 if use_fp16_save else np.float32)
                    feat_by_layer[lid] = feat

                # Write each stage to its own folder.
                for lid, out_path in out_paths.items():
                    tmp_out_path = out_path + ".tmp"
                    with open(tmp_out_path, "wb") as f:
                        np.save(f, feat_by_layer[lid])
                    os.replace(tmp_out_path, out_path)

                ok_items.append((stem, feat_by_layer))

        return {"ok": ok_items, "skip": len(img_paths) - len(work_items), "fail": []}

    def process_one(img_path):
        base = os.path.basename(img_path)
        stem = os.path.splitext(base)[0]
        out_paths = {
            lid: os.path.join(stage_dirs[lid], f"{stem}_dino32.npy") for lid in layer_ids
        }

        if (not args.overwrite) and all(os.path.exists(p) for p in out_paths.values()):
            return "skip"

        img = Image.open(img_path).convert("RGB")
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.BICUBIC)

        inputs = processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states")

            max_layer = len(hidden_states) - 1
            bad_layers = [x for x in layer_ids if x > max_layer]
            if bad_layers:
                raise RuntimeError(
                    f"Requested layers {bad_layers} exceed max available layer index {max_layer}"
                )

            # DINOv3 includes special tokens (CLS + register tokens) before patch tokens.
            all_tokens = hidden_states[max_layer]
            expected_patches = grid_h * grid_w
            num_special = all_tokens.shape[1] - expected_patches
            if num_special < 0:
                raise RuntimeError(
                    f"Model returned fewer tokens than expected patches: {tuple(all_tokens.shape)}"
                )
            feat_by_layer = {}
            for lid in layer_ids:
                tokens = hidden_states[lid][:, num_special:, :]
                tokens = F.normalize(tokens, dim=-1)
                feat = tokens[0].detach().cpu().numpy()

                if feat.shape[0] != grid_h * grid_w:
                    raise RuntimeError(f"Token count mismatch at layer {lid}: {feat.shape}")

                feat = feat.reshape(grid_h, grid_w, -1)
                feat = feat.astype(np.float16 if use_fp16_save else np.float32)
                feat_by_layer[lid] = feat

        # Write each stage to its own folder.
        for lid, out_path in out_paths.items():
            tmp_out_path = out_path + ".tmp"
            with open(tmp_out_path, "wb") as f:
                np.save(f, feat_by_layer[lid])
            os.replace(tmp_out_path, out_path)

        return "ok", stem, feat_by_layer

    img_list = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print("Found images:", len(img_list))

    ok = skip = fail = 0
    last_ok_stem = None
    last_ok_feats = None

    batch_size = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(img_list), batch_size)):
        batch_paths = img_list[i : i + batch_size]
        try:
            r = process_batch(batch_paths)
            skip += r["skip"]
            ok += len(r["ok"])
            fail += len(r["fail"])
            if r["ok"]:
                last_ok_stem, last_ok_feats = r["ok"][-1]
            for fp, fe in r["fail"]:
                print("[FAIL]", fp, fe)
        except Exception as e:
            # Fallback to per-image handling so one bad sample does not drop the whole batch.
            for p in batch_paths:
                try:
                    r1 = process_one(p)
                    if r1 == "skip":
                        skip += 1
                    else:
                        _, stem, feats = r1
                        ok += 1
                        last_ok_stem = stem
                        last_ok_feats = feats
                except Exception as ie:
                    fail += 1
                    print("[FAIL]", p, ie)

    if args.vis and last_ok_feats is not None:
        vis_images = [feat_to_vis_image(last_ok_feats[lid]) for lid in layer_ids]
        vis_hq_images = [img.resize((target_w, target_h), bicubic) for img in vis_images]
        save_layer_grid(vis_images, os.path.join(args.out_dir, "dino32_viz_compare.png"))
        save_layer_grid(vis_hq_images, os.path.join(args.out_dir, "dino32_viz_compare_hq.png"))
        print("viz sample:", last_ok_stem)
    elif args.vis:
        print("viz skipped: no newly processed sample in this run")

    print("\nDone.")
    print("ok   :", ok)
    print("skip :", skip)
    print("fail :", fail)
    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    main()
