import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from lpips import LPIPS
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from model import CANDLE
from refine_nafnet import RefineNAFNet
from utils.aln_dataset import ALNDatasetGeom
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def strip_prefix_state_dict(state_dict, prefix):
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


def load_ckpt_state(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def build_candle(args):
    return CANDLE(
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


class JointRefineModel(pl.LightningModule):
    def __init__(self, args, best_psnr_path=None, best_ckpt_dir=None):
        super().__init__()
        self.args = args
        self.coarse = build_candle(args)
        self.refine = RefineNAFNet(
            img_channel=3,
            width=args.refine_width,
            middle_blk_num=args.refine_middle_blocks,
            enc_blk_nums=tuple(args.refine_enc_blocks),
            dec_blk_nums=tuple(args.refine_dec_blocks),
            use_global_residual=not bool(args.refine_no_global_residual),
        )
        pixel_loss_type = str(args.pixel_loss_type).lower()
        if pixel_loss_type == "l1":
            self.pixel_loss = nn.L1Loss()
        elif pixel_loss_type == "mse":
            self.pixel_loss = nn.MSELoss()
        else:
            self.pixel_loss = CharbonnierLoss()
        self.lpips_loss = LPIPS(net="vgg").eval()
        self.lpips_loss.requires_grad_(False)
        self.ssim_metric = SSIM(data_range=1.0)
        self.loss_pixel_weight = float(args.loss_pixel_weight)
        self.loss_lpips_weight = float(args.loss_lpips_weight)
        self.loss_ssim_weight = float(args.loss_ssim_weight)
        self.best_psnr = -1.0
        self.best_epoch = -1
        self.best_psnr_path = best_psnr_path
        self.best_ckpt_dir = best_ckpt_dir
        self._load_init_weights()

    def _load_init_weights(self):
        coarse_state = strip_prefix_state_dict(load_ckpt_state(self.args.coarse_ckpt), "net.")
        coarse_state = {k: v for k, v in coarse_state.items() if not k.startswith("hvi_consistency.")}
        missing, unexpected = self.coarse.load_state_dict(coarse_state, strict=False)
        if missing:
            print(f"[joint][coarse-init] missing_keys={missing}")
        if unexpected:
            print(f"[joint][coarse-init] unexpected_keys={unexpected}")

        refine_state = load_ckpt_state(self.args.refine_ckpt)
        refine_state = strip_prefix_state_dict(refine_state, "net.")
        missing, unexpected = self.refine.load_state_dict(refine_state, strict=False)
        if missing:
            print(f"[joint][refine-init] missing_keys={missing}")
        if unexpected:
            print(f"[joint][refine-init] unexpected_keys={unexpected}")

    def setup(self, stage=None):
        self.lpips_loss = self.lpips_loss.to(self.device)
        self.ssim_metric = self.ssim_metric.to(self.device)

    def _unpack_batch(self, batch):
        if len(batch) == 4:
            ([sample_name, sample_idx], input_img, dino_img, target_img) = batch
            return {
                "id": sample_name,
                "sample_idx": sample_idx,
                "input": input_img,
                "dino": dino_img,
                "target": target_img,
            }
        raise ValueError(f"Unexpected batch format with length={len(batch)}")

    def forward(self, inp, dino):
        coarse = self.coarse(inp, dino)
        refined = self.refine(coarse)
        return coarse, refined

    def _compute_loss(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        loss_pixel = self.pixel_loss(pred, target)
        loss_lpips = self.lpips_loss(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean()
        loss_ssim = (1.0 - self.ssim_metric(pred, target)).mean()
        total = (
            self.loss_pixel_weight * loss_pixel
            + self.loss_lpips_weight * loss_lpips
            + self.loss_ssim_weight * loss_ssim
        )
        return pred, loss_pixel, loss_lpips, loss_ssim, total

    def training_step(self, batch, batch_idx):
        batch_data = self._unpack_batch(batch)
        coarse, refined = self(batch_data["input"], batch_data["dino"])
        refined, loss_pixel, loss_lpips, loss_ssim, total = self._compute_loss(refined, batch_data["target"])
        self.log("loss_pixel", loss_pixel.detach(), sync_dist=True)
        self.log("loss_lpips", loss_lpips.detach(), sync_dist=True)
        self.log("loss_ssim", loss_ssim.detach(), sync_dist=True)
        self.log("total_loss", total.detach(), sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        batch_data = self._unpack_batch(batch)
        with torch.no_grad():
            coarse, refined = self(batch_data["input"], batch_data["dino"])
            refined, loss_pixel, loss_lpips, loss_ssim, total = self._compute_loss(refined, batch_data["target"])
        psnr_i, ssim_i, _ = compute_psnr_ssim(refined, batch_data["target"])
        self.log("val_loss", total.detach(), sync_dist=True)
        self.log("val_lpips", loss_lpips.detach(), sync_dist=True)
        self.log("val_psnr", psnr_i, sync_dist=True)
        self.log("val_ssim", ssim_i, sync_dist=True)
        if batch_idx == 0:
            self._save_val_vis(batch_data, coarse, refined)
        return {"psnr": psnr_i, "ssim": ssim_i}

    def _save_val_vis(self, batch_data, coarse, refined):
        if not self.best_ckpt_dir:
            return
        vis_dir = Path(self.best_ckpt_dir).parent / "val_vis" / f"epoch_{self.current_epoch:03d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        max_vis = min(int(self.args.val_vis_count), refined.shape[0])
        for i in range(max_vis):
            sample_id = batch_data["id"][i]
            save_image(torch.clamp(batch_data["input"][i], 0.0, 1.0), vis_dir / f"{sample_id}_input.png")
            save_image(torch.clamp(coarse[i], 0.0, 1.0), vis_dir / f"{sample_id}_coarse.png")
            save_image(torch.clamp(refined[i], 0.0, 1.0), vis_dir / f"{sample_id}_refined.png")
            save_image(torch.clamp(batch_data["target"][i], 0.0, 1.0), vis_dir / f"{sample_id}_gt.png")

    def on_validation_epoch_end(self):
        avg_psnr = self.trainer.callback_metrics.get("val_psnr", torch.tensor(0.0))
        avg_ssim = self.trainer.callback_metrics.get("val_ssim", torch.tensor(0.0))
        if isinstance(avg_psnr, torch.Tensor):
            avg_psnr = avg_psnr.item()
        if isinstance(avg_ssim, torch.Tensor):
            avg_ssim = avg_ssim.item()
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.best_epoch = self.current_epoch
            if self.best_psnr_path is not None:
                with open(self.best_psnr_path, "w") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"Best PSNR: {self.best_psnr:.4f}\n")
                    f.write(f"Best SSIM: {avg_ssim:.4f}\n")
                    f.write(f"Epoch: {self.best_epoch}\n")
                    f.write(f"Time: {timestamp}\n")
        print(f"\n[JointRefine][Epoch {self.current_epoch}] PSNR={avg_psnr:.4f} SSIM={avg_ssim:.4f} Best={self.best_psnr:.4f}@{self.best_epoch}")

    def _build_optimizer(self):
        coarse_lr = float(self.args.coarse_lr) if float(self.args.coarse_lr) > 0 else float(self.args.lr)
        refine_lr = float(self.args.refine_lr) if float(self.args.refine_lr) > 0 else float(self.args.lr)
        betas = (float(self.args.adam_beta1), float(self.args.adam_beta2))
        wd = float(self.args.weight_decay)
        momentum = float(self.args.sgd_momentum)
        params = [
            {"params": self.coarse.parameters(), "lr": coarse_lr},
            {"params": self.refine.parameters(), "lr": refine_lr},
        ]
        opt_type = self.args.optimizer_type.lower()
        if opt_type == "adamw":
            return optim.AdamW(params, betas=betas, weight_decay=wd)
        if opt_type == "sgd":
            return optim.SGD(params, momentum=momentum, weight_decay=wd, nesterov=bool(self.args.sgd_nesterov))
        return optim.Adam(params, betas=betas, weight_decay=wd)

    def _build_scheduler(self, optimizer):
        scheduler_type = self.args.scheduler_type.lower()
        if scheduler_type == "fixed":
            return None
        t_max = int(self.args.scheduler_tmax_epochs) if int(self.args.scheduler_tmax_epochs) > 0 else int(self.args.epochs)
        t_max = max(1, t_max)
        if scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer=optimizer, T_max=t_max, eta_min=float(self.args.scheduler_min_lr))
        if scheduler_type == "warmup_cosine":
            warmup_epochs = max(0, int(self.args.scheduler_warmup_epochs))
            return LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=warmup_epochs, max_epochs=t_max)
        if scheduler_type == "plateau":
            return ReduceLROnPlateau(
                optimizer=optimizer,
                mode=str(self.args.plateau_mode).lower(),
                factor=float(self.args.plateau_factor),
                patience=int(self.args.plateau_patience),
                min_lr=float(self.args.scheduler_min_lr),
                threshold=float(self.args.plateau_threshold),
                threshold_mode=str(self.args.plateau_threshold_mode).lower(),
            )
        raise ValueError(f"Unsupported scheduler_type: {self.args.scheduler_type}")

    def configure_optimizers(self):
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        if self.args.scheduler_type.lower() == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": str(self.args.plateau_monitor),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return [optimizer], [scheduler]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--coarse_lr", type=float, default=2e-6)
    parser.add_argument("--refine_lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--patch_height", type=int, default=0)
    parser.add_argument("--patch_width", type=int, default=0)
    parser.add_argument("--train_crop_mode", type=str, default="random", choices=["random", "fixed", "none"])
    parser.add_argument("--train_crop_x", type=int, default=-1)
    parser.add_argument("--train_crop_y", type=int, default=-1)
    parser.add_argument("--val_patch_height", type=int, default=0)
    parser.add_argument("--val_patch_width", type=int, default=0)
    parser.add_argument("--val_crop_mode", type=str, default="none", choices=["random", "fixed", "none"])
    parser.add_argument("--val_crop_x", type=int, default=-1)
    parser.add_argument("--val_crop_y", type=int, default=-1)
    parser.add_argument("--use_data_aug", type=int, default=1)
    parser.add_argument("--aug_hflip_prob", type=float, default=0.5)
    parser.add_argument("--aug_vflip_prob", type=float, default=0.5)
    parser.add_argument("--aug_rotate90", type=int, default=1)
    parser.add_argument("--disable_aug_if_fullres", type=int, default=1)
    parser.add_argument("--wblogger", type=str, default="none")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--resume_weights_only", type=int, default=0)
    parser.add_argument("--save_last_ckpt", type=int, default=1)
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--sgd_nesterov", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_type", type=str, default="fixed", choices=["fixed", "cosine", "warmup_cosine", "plateau"])
    parser.add_argument("--scheduler_warmup_epochs", type=int, default=0)
    parser.add_argument("--scheduler_tmax_epochs", type=int, default=0)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--plateau_monitor", type=str, default="val_psnr")
    parser.add_argument("--plateau_mode", type=str, default="max", choices=["min", "max"])
    parser.add_argument("--plateau_factor", type=float, default=0.85)
    parser.add_argument("--plateau_patience", type=int, default=5)
    parser.add_argument("--plateau_threshold", type=float, default=1e-4)
    parser.add_argument("--plateau_threshold_mode", type=str, default="rel", choices=["rel", "abs"])
    parser.add_argument("--coarse_ckpt", type=str, required=True)
    parser.add_argument("--refine_ckpt", type=str, required=True)
    parser.add_argument("--pixel_loss_type", type=str, default="l1", choices=["l1", "mse", "charbonnier"])
    parser.add_argument("--loss_pixel_weight", type=float, default=1.0)
    parser.add_argument("--loss_lpips_weight", type=float, default=0.0)
    parser.add_argument("--loss_ssim_weight", type=float, default=0.7)
    parser.add_argument("--val_vis_count", type=int, default=4)
    parser.add_argument("--train_input_dir", type=str, required=True)
    parser.add_argument("--train_normals_dir", type=str, required=True)
    parser.add_argument("--train_target_dir", type=str, required=True)
    parser.add_argument("--train_dino_dir", type=str, required=True)
    parser.add_argument("--test_input_dir", type=str, required=True)
    parser.add_argument("--test_normals_dir", type=str, required=True)
    parser.add_argument("--test_target_dir", type=str, required=True)
    parser.add_argument("--test_dino_dir", type=str, required=True)
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
    parser.add_argument("--refine_width", type=int, default=48)
    parser.add_argument("--refine_middle_blocks", type=int, default=6)
    parser.add_argument("--refine_enc_blocks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--refine_dec_blocks", type=int, nargs="+", default=[1, 1, 2, 2])
    parser.add_argument("--refine_no_global_residual", type=int, default=1)
    return parser.parse_args()


def build_loader(args, split, is_train):
    patch_h = args.patch_height if args.patch_height > 0 else None
    patch_w = args.patch_width if args.patch_width > 0 else None
    if not is_train:
        patch_h = args.val_patch_height if args.val_patch_height > 0 else None
        patch_w = args.val_patch_width if args.val_patch_width > 0 else None
    ds = ALNDatasetGeom(
        input_folder=args.train_input_dir if is_train else args.test_input_dir,
        geom_folder=(args.train_normals_dir if is_train else args.test_normals_dir) or (args.train_input_dir if is_train else args.test_input_dir),
        target_folder=args.train_target_dir if is_train else args.test_target_dir,
        dino_folder=args.train_dino_dir if is_train else args.test_dino_dir,
        dino_suffix=args.dino_suffix,
        dino_dim=args.dino_dim,
        patch_size=args.patch_size if is_train else None,
        patch_height=patch_h,
        patch_width=patch_w,
        crop_mode=args.train_crop_mode if is_train else args.val_crop_mode,
        crop_x=args.train_crop_x if is_train else args.val_crop_x,
        crop_y=args.train_crop_y if is_train else args.val_crop_y,
        enable_aug=bool(args.use_data_aug) if is_train else False,
        aug_hflip_prob=args.aug_hflip_prob,
        aug_vflip_prob=args.aug_vflip_prob,
        aug_rotate90=bool(args.aug_rotate90),
        disable_aug_if_fullres=bool(args.disable_aug_if_fullres),
        allow_fallback_normal=True,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size if is_train else args.val_batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def main():
    args = parse_args()
    external_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if external_visible_devices:
        print(f"Using externally provided CUDA_VISIBLE_DEVICES={external_visible_devices}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        print(f"CUDA_VISIBLE_DEVICES not preset; fallback to args.cuda={args.cuda}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log_path = os.path.join(args.ckpt_dir, f"training_log_{timestamp}.txt")
    best_psnr_path = os.path.join(args.ckpt_dir, "best_psnr.txt")
    best_ckpt_dir = os.path.join(args.ckpt_dir, "best")

    logger = TensorBoardLogger(save_dir="logs/") if str(args.wblogger).lower() in ("none", "") else WandbLogger(project=args.wblogger, name="CANDLE-Refine-Joint")
    checkpoint_callback = ModelCheckpoint(
        dirpath=best_ckpt_dir,
        filename="best_model",
        monitor="val_psnr",
        mode="max",
        save_top_k=1,
        save_last=bool(args.save_last_ckpt),
        verbose=True,
    )

    trainloader = build_loader(args, "train", is_train=True)
    testloader = build_loader(args, "test", is_train=False)
    model = JointRefineModel(args, best_psnr_path=best_psnr_path, best_ckpt_dir=best_ckpt_dir)

    log_file = open(training_log_path, "w", buffering=1)
    original_stdout = sys.stdout

    class TeeOutput:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = TeeOutput(sys.stdout, log_file)
    print(f"\n{'='*60}\nJoint Refine Training Started at {timestamp}\nCheckpoint Dir: {args.ckpt_dir}\nTraining Log: {training_log_path}\nBest PSNR Log: {best_psnr_path}\n{'='*60}\n")

    trainer = pl.Trainer(
        precision="16-mixed",
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        accumulate_grad_batches=max(1, int(args.accumulate_grad_batches)),
        logger=logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        strategy="ddp_find_unused_parameters_true" if int(args.num_gpus) > 1 else "auto",
    )

    ckpt_path = None
    if args.resume_from:
        if bool(args.resume_weights_only):
            print(f"Loading joint weights only from: {args.resume_from}")
            ckpt = torch.load(args.resume_from, map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[joint][weights-only] missing_keys={missing}")
            if unexpected:
                print(f"[joint][weights-only] unexpected_keys={unexpected}")
        else:
            ckpt_path = args.resume_from
            print(f"Resuming full joint state from: {ckpt_path}")

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader, ckpt_path=ckpt_path)
    print(f"\n{'='*60}\nJoint Refine Training Completed!\nBest PSNR: {model.best_psnr:.4f}\nBest Epoch: {model.best_epoch}\nBest Model saved to: {best_ckpt_dir}/best_model.ckpt\n{'='*60}\n")
    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
