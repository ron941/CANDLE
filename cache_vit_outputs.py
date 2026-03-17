import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CANDLE
from utils.aln_dataset import ALNDatasetGeom


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--frozen_vit_ckpt", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--train_input_dir", type=str, required=True)
    parser.add_argument("--train_target_dir", type=str, required=True)
    parser.add_argument("--train_dino_dir", type=str, required=True)
    parser.add_argument("--test_input_dir", type=str, required=True)
    parser.add_argument("--test_target_dir", type=str, required=True)
    parser.add_argument("--test_dino_dir", type=str, required=True)
    parser.add_argument("--train_normals_dir", type=str, default="")
    parser.add_argument("--test_normals_dir", type=str, default="")
    parser.add_argument("--dino_suffix", type=str, default="_dino32")
    parser.add_argument("--dino_dim", type=int, default=1024)
    parser.add_argument("--query_dim", type=int, default=32)
    parser.add_argument("--use_psf_dr", type=int, default=1)
    parser.add_argument("--dr_heads", type=int, default=4)
    parser.add_argument("--dr_dropout", type=float, default=0.0)
    parser.add_argument("--dr_alpha_init", type=float, default=0.0)
    parser.add_argument("--psf_gate_hidden", type=int, default=64)
    parser.add_argument("--use_sffb_decoder", type=int, default=0)
    parser.add_argument("--use_bfacg_decoder", type=int, default=1)
    parser.add_argument("--bfacg_hidden", type=int, default=64)
    parser.add_argument("--bfacg_res_scale_init", type=float, default=0.0)
    parser.add_argument("--bfacg_variant", type=str, default="v1")
    parser.add_argument("--bfacg_neutral_k", type=float, default=0.3)
    parser.add_argument("--use_clp_decoder", type=int, default=0)
    parser.add_argument("--clp_latent_ch", type=int, default=16)
    parser.add_argument("--clp_res_scale_init", type=float, default=0.0)
    parser.add_argument("--clp_conf_bias_init", type=float, default=-2.0)
    parser.add_argument("--use_ica7", type=int, default=0)
    parser.add_argument("--use_abc_ica", type=int, default=0)
    parser.add_argument("--ica_aux_ch", type=int, default=32)
    parser.add_argument("--ica_hidden", type=int, default=64)
    parser.add_argument("--ica_res_scale_init", type=float, default=0.0)
    parser.add_argument("--abc_hist_ckpt", type=str, default="/raid/ron/ALN_768/ABC-Former-main/ABC-Former/checkpoints_CL3AN_resize512/hist/Hist_d16_last.pth")
    parser.add_argument("--abc_lab_ckpt", type=str, default="/raid/ron/ALN_768/ABC-Former-main/ABC-Former/checkpoints_CL3AN_resize512/lab/Lab_d16_last.pth")
    parser.add_argument("--abc_aux_embed_dim", type=int, default=16)
    parser.add_argument("--abc_detach_aux_weight", type=int, default=1)
    parser.add_argument("--use_hvi_bottleneck", type=int, default=0)
    parser.add_argument("--sir_attn_dim", type=int, default=64)
    parser.add_argument("--sir_heads", type=int, default=4)
    parser.add_argument("--sir_dropout", type=float, default=0.0)
    parser.add_argument("--sir_lambda_init", type=float, default=0.0)
    parser.add_argument("--sir_use_blur_ill", type=int, default=1)
    parser.add_argument("--hvi_eps", type=float, default=1e-8)
    parser.add_argument("--save_vit_png", type=int, default=1)
    parser.add_argument("--train_patch_size", type=int, default=768)
    parser.add_argument("--train_patch_height", type=int, default=0)
    parser.add_argument("--train_patch_width", type=int, default=0)
    parser.add_argument("--train_crop_mode", type=str, default="random", choices=["random", "fixed", "none"])
    parser.add_argument("--train_crop_x", type=int, default=-1)
    parser.add_argument("--train_crop_y", type=int, default=-1)
    parser.add_argument("--train_cache_repeats", type=int, default=1)
    parser.add_argument("--test_patch_size", type=int, default=0)
    parser.add_argument("--test_patch_height", type=int, default=0)
    parser.add_argument("--test_patch_width", type=int, default=0)
    parser.add_argument("--test_crop_mode", type=str, default="none", choices=["random", "fixed", "none"])
    parser.add_argument("--test_crop_x", type=int, default=-1)
    parser.add_argument("--test_crop_y", type=int, default=-1)
    return parser.parse_args()


def build_candle(args):
    model = CANDLE(
        decoder=True,
        dino_dim=args.dino_dim,
        query_dim=args.query_dim,
        use_psf_dr=bool(args.use_psf_dr),
        dr_heads=args.dr_heads,
        dr_dropout=args.dr_dropout,
        dr_alpha_init=args.dr_alpha_init,
        psf_gate_hidden=args.psf_gate_hidden,
        use_sffb_decoder=bool(args.use_sffb_decoder),
        use_bfacg_decoder=bool(args.use_bfacg_decoder),
        bfacg_hidden=args.bfacg_hidden,
        bfacg_res_scale_init=args.bfacg_res_scale_init,
        bfacg_variant=args.bfacg_variant,
        bfacg_neutral_k=args.bfacg_neutral_k,
        use_clp_decoder=bool(args.use_clp_decoder),
        clp_latent_ch=args.clp_latent_ch,
        clp_res_scale_init=args.clp_res_scale_init,
        clp_conf_bias_init=args.clp_conf_bias_init,
        use_ica7=bool(args.use_ica7),
        use_abc_ica=bool(args.use_abc_ica),
        ica_aux_ch=args.ica_aux_ch,
        ica_hidden=args.ica_hidden,
        ica_res_scale_init=args.ica_res_scale_init,
        abc_hist_ckpt_path=args.abc_hist_ckpt,
        abc_lab_ckpt_path=args.abc_lab_ckpt,
        abc_aux_embed_dim=args.abc_aux_embed_dim,
        abc_detach_aux_weight=bool(args.abc_detach_aux_weight),
        use_hvi_bottleneck=bool(args.use_hvi_bottleneck),
        sir_attn_dim=args.sir_attn_dim,
        sir_heads=args.sir_heads,
        sir_dropout=args.sir_dropout,
        sir_lambda_init=args.sir_lambda_init,
        sir_use_blur_ill=bool(args.sir_use_blur_ill),
        hvi_eps=args.hvi_eps,
    )
    ckpt = torch.load(args.frozen_vit_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # Lightning checkpoints store CANDLE under the `net.` prefix.
    if any(k.startswith("net.") for k in state_dict.keys()):
        state_dict = {
            (k[4:] if k.startswith("net.") else k): v
            for k, v in state_dict.items()
            if not k.startswith("hvi_consistency.")
        }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[cache] missing_keys={missing}")
    if unexpected:
        print(f"[cache] unexpected_keys={unexpected}")
    model.eval()
    model.requires_grad_(False)
    return model


def save_image_tensor(path, tensor):
    arr = torch.clamp(tensor, 0.0, 1.0).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def build_loader(input_dir, target_dir, normals_dir, dino_dir, args, split):
    if split == "train":
        patch_size = args.train_patch_size if int(args.train_patch_size) > 0 else None
        patch_height = args.train_patch_height if int(args.train_patch_height) > 0 else None
        patch_width = args.train_patch_width if int(args.train_patch_width) > 0 else None
        crop_mode = args.train_crop_mode
        crop_x = args.train_crop_x
        crop_y = args.train_crop_y
    else:
        patch_size = args.test_patch_size if int(args.test_patch_size) > 0 else None
        patch_height = args.test_patch_height if int(args.test_patch_height) > 0 else None
        patch_width = args.test_patch_width if int(args.test_patch_width) > 0 else None
        crop_mode = args.test_crop_mode
        crop_x = args.test_crop_x
        crop_y = args.test_crop_y
    ds = ALNDatasetGeom(
        input_folder=input_dir,
        geom_folder=normals_dir or input_dir,
        target_folder=target_dir,
        dino_folder=dino_dir,
        dino_suffix=args.dino_suffix,
        dino_dim=args.dino_dim,
        patch_size=patch_size,
        patch_height=patch_height,
        patch_width=patch_width,
        crop_mode=crop_mode,
        crop_x=crop_x,
        crop_y=crop_y,
        enable_aug=False,
        allow_fallback_normal=True,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


def export_split(split, loader, model, cache_root, device, save_vit_png, repeat_idx=0, repeat_count=1):
    split_root = Path(cache_root) / split
    input_dir = split_root / "input"
    gt_dir = split_root / "gt"
    vit_dir = split_root / "vit"
    vit_png_dir = split_root / "vit_png"
    input_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    vit_dir.mkdir(parents=True, exist_ok=True)
    if save_vit_png:
        vit_png_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [("sample_id", "input_png", "vit_npy", "gt_png")]

    desc = f"cache-{split}"
    if repeat_count > 1:
        desc += f"[{repeat_idx + 1}/{repeat_count}]"
    for batch in tqdm(loader, desc=desc):
        ids, inp, dino, gt = batch
        sample_ids = ids[0] if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], (list, tuple)) else ids
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]
        inp = inp.to(device, non_blocking=True)
        dino = dino.to(device, non_blocking=True)
        with torch.no_grad():
            vit_out = torch.clamp(model(inp, dino), 0.0, 1.0)
        for i, sample_id in enumerate(sample_ids):
            if repeat_count > 1:
                sample_id = f"{sample_id}_rep{repeat_idx:02d}"
            input_path = input_dir / f"{sample_id}_input.png"
            gt_path = gt_dir / f"{sample_id}_gt.png"
            vit_path = vit_dir / f"{sample_id}_vit.npy"
            save_image_tensor(input_path, inp[i].detach().cpu())
            save_image_tensor(gt_path, gt[i].detach().cpu())
            np.save(vit_path, vit_out[i].detach().cpu().numpy().astype(np.float32))
            if save_vit_png:
                save_image_tensor(vit_png_dir / f"{sample_id}_vit.png", vit_out[i].detach().cpu())
            manifest_rows.append((sample_id, input_path.name, vit_path.name, gt_path.name))

    with open(split_root / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows(manifest_rows)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_candle(args).to(device)

    train_loader = build_loader(args.train_input_dir, args.train_target_dir, args.train_normals_dir, args.train_dino_dir, args, split="train")
    test_loader = build_loader(args.test_input_dir, args.test_target_dir, args.test_normals_dir, args.test_dino_dir, args, split="test")

    repeat_count = max(1, int(args.train_cache_repeats))
    for repeat_idx in range(repeat_count):
        export_split("train", train_loader, model, args.cache_root, device, bool(args.save_vit_png), repeat_idx=repeat_idx, repeat_count=repeat_count)
    export_split("test", test_loader, model, args.cache_root, device, bool(args.save_vit_png), repeat_idx=0, repeat_count=1)
    print(f"Cached VIT outputs saved to: {args.cache_root}")


if __name__ == "__main__":
    main()
