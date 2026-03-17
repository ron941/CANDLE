import subprocess
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.aln_dataset import ALNDatasetGeom
from model import CANDLE, LowFreqChromaBiasNet
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import time
from PIL import Image
from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class CANDLEModel(pl.LightningModule):
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
        self.lpips_loss = LPIPS(net="vgg").requires_grad_(False)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self, x, dino_tokens, input_hist=None, input_lab_hist=None):
        if self.use_lowfreq_bias_baseline:
            return self.net(x)
        return self.net(x, dino_tokens, input_hist=input_hist, input_lab_hist=input_lab_hist)

def main():
    print("Options")
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)

    # Careful: the input and target folders are the same just to avoid creating a new DataLoader class, but there is no target images.
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
    
    valloader = DataLoader(valset, batch_size=opt.batch_size, pin_memory=True, shuffle=False,
                             drop_last=False, num_workers=opt.num_workers)
    model = CANDLEModel()
    ckpt = torch.load(opt.pretrained_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if not model.use_lowfreq_bias_baseline:
        if "net.dino_to_geom.weight" in state_dict and "net.dino_to_query.weight" not in state_dict:
            print("[Compat] Remapping net.dino_to_geom.weight -> net.dino_to_query.weight")
            state_dict["net.dino_to_query.weight"] = state_dict.pop("net.dino_to_geom.weight")

        has_query_norm = any(k.startswith("net.dino_to_query_norm.") for k in state_dict.keys())
        if not has_query_norm:
            print("[Compat] Checkpoint has no net.dino_to_query_norm.*; switching to Identity for inference.")
            model.net.dino_to_query_norm = nn.Identity()

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if any(k.startswith("net.hvi_bottleneck.") for k in missing_keys):
        print("[Compat] new HVI branch not found, initialized randomly.")
    if any(k.startswith("net.prompt") for k in missing_keys):
        print("[Compat] decoder prompt changed to HVI modulation; old prompt-bank weights are ignored.")
    if missing_keys:
        print(f"[load_state_dict] missing_keys={missing_keys}")
    if unexpected_keys:
        print(f"[load_state_dict] unexpected_keys={unexpected_keys}")
    model.eval()
    model.cuda()

    times = []

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    compare_dir = os.path.join(opt.output_path, "comparisons")
    os.makedirs(compare_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(valloader)):
        with torch.no_grad():
            if len(batch) == 8:
                ([name, _], input_img, dino_tokens, gt_img, input_hist, _, input_lab_hist, _) = batch
            elif len(batch) == 4:
                ([name, _], input_img, dino_tokens, gt_img) = batch
                input_hist = None
                input_lab_hist = None
            else:
                raise ValueError(f"Unexpected batch format with length={len(batch)}")

            input_img = input_img.cuda()
            dino_tokens = dino_tokens.cuda()
            if input_hist is not None:
                input_hist = input_hist.cuda()
            if input_lab_hist is not None:
                input_lab_hist = input_lab_hist.cuda()

            start = time.time()

            output = model(input_img, dino_tokens, input_hist=input_hist, input_lab_hist=input_lab_hist)

            times.append(time.time()-start)

            output = output.permute(0, 2, 3, 1).cpu().numpy()
            output = np.clip(output, 0, 1)
            input_np = input_img.permute(0, 2, 3, 1).detach().cpu().numpy()
            input_np = np.clip(input_np, 0, 1)
            gt_np = gt_img.permute(0, 2, 3, 1).detach().cpu().numpy()
            gt_np = np.clip(gt_np, 0, 1)

            if isinstance(name, (list, tuple)):
                batch_names = [str(n) for n in name]
            else:
                batch_names = [str(name)]

            for i in range(output.shape[0]):
                sample_name = batch_names[i] if i < len(batch_names) else f"sample_{batch_idx}_{i}"
                output_img = Image.fromarray((output[i] * 255).astype(np.uint8)).convert('RGB')
                output_img.save(f'{opt.output_path}/{sample_name}.png')

                # Align for visualization when model auto-crops to multiples of 8.
                min_h = min(input_np[i].shape[0], output[i].shape[0], gt_np[i].shape[0])
                min_w = min(input_np[i].shape[1], output[i].shape[1], gt_np[i].shape[1])
                in_vis = input_np[i][:min_h, :min_w]
                out_vis = output[i][:min_h, :min_w]
                gt_vis = gt_np[i][:min_h, :min_w]
                compare_strip = np.concatenate([in_vis, out_vis, gt_vis], axis=1)
                compare_img = Image.fromarray((compare_strip * 255).astype(np.uint8)).convert('RGB')
                compare_img.save(os.path.join(compare_dir, f"{sample_name}_compare.png"))
    
    print(f"Average inference time: {np.mean(times)}")

if __name__ == '__main__':
    main()
