import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LowFreqChromaBiasNet, CANDLE
from options import options as opt
from utils.aln_dataset import ALNDatasetGeom


def _apply_flip(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "h":
        return torch.flip(x, dims=[-1])
    if mode == "v":
        return torch.flip(x, dims=[-2])
    if mode == "hv":
        return torch.flip(x, dims=[-2, -1])
    raise ValueError(f"Unknown flip mode: {mode}")


def _invert_flip(x: torch.Tensor, mode: str) -> torch.Tensor:
    # Flip ops are self-inverse.
    return _apply_flip(x, mode)


def _tta_modes() -> List[str]:
    return ["none", "h", "v", "hv"]


class CANDLEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_lowfreq_bias_baseline = bool(opt.use_lowfreq_bias_baseline)
        if self.use_lowfreq_bias_baseline:
            self.net = LowFreqChromaBiasNet(
                hidden=opt.lfb_hidden,
                kernel_size=opt.lfb_kernel_size,
                sigma=opt.lfb_sigma,
            )
        else:
            self.net = CANDLE(
                decoder=True,
                dino_dim=opt.dino_dim,
                query_dim=opt.query_dim,
                use_psf_dr=bool(opt.use_psf_dr),
                dr_heads=opt.dr_heads,
                dr_dropout=opt.dr_dropout,
                dr_alpha_init=opt.dr_alpha_init,
                psf_gate_hidden=opt.psf_gate_hidden,
                use_sffb_decoder=bool(opt.use_sffb_decoder),
                use_bfacg_decoder=bool(opt.use_bfacg_decoder),
                bfacg_hidden=opt.bfacg_hidden,
                bfacg_res_scale_init=opt.bfacg_res_scale_init,
                bfacg_variant=opt.bfacg_variant,
                bfacg_neutral_k=opt.bfacg_neutral_k,
                use_clp_decoder=bool(opt.use_clp_decoder),
                clp_latent_ch=opt.clp_latent_ch,
                clp_res_scale_init=opt.clp_res_scale_init,
                clp_conf_bias_init=opt.clp_conf_bias_init,
                use_ica7=bool(opt.use_ica7),
                use_abc_ica=bool(opt.use_abc_ica),
                ica_aux_ch=opt.ica_aux_ch,
                ica_hidden=opt.ica_hidden,
                ica_res_scale_init=opt.ica_res_scale_init,
                abc_hist_ckpt_path=opt.abc_hist_ckpt,
                abc_lab_ckpt_path=opt.abc_lab_ckpt,
                abc_aux_embed_dim=opt.abc_aux_embed_dim,
                abc_detach_aux_weight=bool(opt.abc_detach_aux_weight),
                use_hvi_bottleneck=bool(opt.use_hvi_bottleneck),
                sir_attn_dim=opt.sir_attn_dim,
                sir_heads=opt.sir_heads,
                sir_dropout=opt.sir_dropout,
                sir_lambda_init=opt.sir_lambda_init,
                sir_use_blur_ill=bool(opt.sir_use_blur_ill),
                hvi_eps=opt.hvi_eps,
            )

    def forward(self, x, dino_tokens, input_hist=None, input_lab_hist=None):
        if self.use_lowfreq_bias_baseline:
            return self.net(x)
        return self.net(x, dino_tokens, input_hist=input_hist, input_lab_hist=input_lab_hist)


def _build_model(device: torch.device) -> CANDLEModel:
    model = CANDLEModel()
    ckpt = torch.load(opt.pretrained_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if not model.use_lowfreq_bias_baseline:
        if "net.dino_to_geom.weight" in state_dict and "net.dino_to_query.weight" not in state_dict:
            state_dict["net.dino_to_query.weight"] = state_dict.pop("net.dino_to_geom.weight")
        has_query_norm = any(k.startswith("net.dino_to_query_norm.") for k in state_dict.keys())
        if not has_query_norm:
            model.net.dino_to_query_norm = nn.Identity()

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[load_state_dict] missing_keys={missing_keys}")
    if unexpected_keys:
        print(f"[load_state_dict] unexpected_keys={unexpected_keys}")

    model.eval()
    model.to(device)
    return model


def _infer_tta(
    model: CANDLEModel,
    input_img: torch.Tensor,
    dino_tokens: torch.Tensor,
    input_hist: torch.Tensor = None,
    input_lab_hist: torch.Tensor = None,
) -> torch.Tensor:
    preds: List[torch.Tensor] = []
    for mode in _tta_modes():
        x_aug = _apply_flip(input_img, mode)
        dino_aug = _apply_flip(dino_tokens, mode)
        out_aug = model(x_aug, dino_aug, input_hist=input_hist, input_lab_hist=input_lab_hist)
        out = _invert_flip(out_aug, mode)
        preds.append(out)
    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    print("Options")
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"TTA modes: {_tta_modes()}")

    valset = ALNDatasetGeom(
        input_folder=opt.test_input_dir,
        geom_folder=opt.test_normals_dir,
        target_folder=opt.test_target_dir,
        crop_mode="none",
        dino_folder=opt.test_dino_dir,
        dino_suffix=opt.dino_suffix,
        dino_dim=opt.dino_dim,
        dino_crop_manifest=opt.test_dino_crop_manifest,
        dino_ref_width=opt.dino_ref_width,
        dino_ref_height=opt.dino_ref_height,
        enable_aug=False,
        return_aux_hist=bool(opt.use_abc_ica and not opt.use_lowfreq_bias_baseline),
    )
    valloader = DataLoader(
        valset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers,
    )

    model = _build_model(device)
    os.makedirs(opt.output_path, exist_ok=True)
    times: List[float] = []

    for batch_idx, batch in enumerate(tqdm(valloader)):
        with torch.no_grad():
            if len(batch) == 8:
                ([name, _], input_img, dino_tokens, _, input_hist, _, input_lab_hist, _) = batch
            elif len(batch) == 4:
                ([name, _], input_img, dino_tokens, _) = batch
                input_hist = None
                input_lab_hist = None
            else:
                raise ValueError(f"Unexpected batch format with length={len(batch)}")

            input_img = input_img.to(device, non_blocking=True)
            dino_tokens = dino_tokens.to(device, non_blocking=True)
            if input_hist is not None:
                input_hist = input_hist.to(device, non_blocking=True)
            if input_lab_hist is not None:
                input_lab_hist = input_lab_hist.to(device, non_blocking=True)

            start = time.time()
            output = _infer_tta(
                model,
                input_img,
                dino_tokens,
                input_hist=input_hist,
                input_lab_hist=input_lab_hist,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)

            output = output.permute(0, 2, 3, 1).cpu().numpy()
            output = np.clip(output, 0, 1)

            if isinstance(name, (list, tuple)):
                batch_names = [str(n) for n in name]
            else:
                batch_names = [str(name)]

            for i in range(output.shape[0]):
                sample_name = batch_names[i] if i < len(batch_names) else f"sample_{batch_idx}_{i}"
                output_img = Image.fromarray((output[i] * 255).astype(np.uint8)).convert("RGB")
                output_img.save(os.path.join(opt.output_path, f"{sample_name}.png"))

    if times:
        print(f"Average inference time per image [s]: {float(np.mean(times)):.6f}")
    else:
        print("No samples processed.")


if __name__ == "__main__":
    main()
