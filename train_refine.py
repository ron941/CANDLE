import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from refine_nafnet import RefineNAFNet
from utils.refine_dataset import RefineDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class RefineModel(pl.LightningModule):
    def __init__(self, args, best_psnr_path=None, best_ckpt_dir=None):
        super().__init__()
        self.args = args
        self.net = RefineNAFNet(
            img_channel=3,
            width=args.refine_width,
            middle_blk_num=args.refine_middle_blocks,
            enc_blk_nums=tuple(args.refine_enc_blocks),
            dec_blk_nums=tuple(args.refine_dec_blocks),
            use_global_residual=not bool(args.refine_no_global_residual),
        )
        self.char_loss = CharbonnierLoss()
        self.ssim_metric = SSIM(data_range=1.0)
        self.loss_char_weight = float(args.loss_char_weight)
        self.loss_ssim_weight = float(args.loss_ssim_weight)
        self.best_psnr = -1.0
        self.best_epoch = -1
        self.best_psnr_path = best_psnr_path
        self.best_ckpt_dir = best_ckpt_dir

    def setup(self, stage=None):
        self.ssim_metric = self.ssim_metric.to(self.device)

    def forward(self, vit):
        return self.net(vit)

    def _compute_loss(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        loss_char = self.char_loss(pred, target)
        loss_ssim = (1.0 - self.ssim_metric(pred, target)).mean()
        total = self.loss_char_weight * loss_char + self.loss_ssim_weight * loss_ssim
        return pred, loss_char, loss_ssim, total

    def training_step(self, batch, batch_idx):
        pred, loss_char, loss_ssim, total = self._compute_loss(self(batch["vit"]), batch["gt"])
        self.log("loss_char", loss_char.detach(), sync_dist=True)
        self.log("loss_ssim", loss_ssim.detach(), sync_dist=True)
        self.log("total_loss", total.detach(), sync_dist=True)
        return total

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pred, loss_char, loss_ssim, total = self._compute_loss(self(batch["vit"]), batch["gt"])
        psnr_i, ssim_i, _ = compute_psnr_ssim(pred, batch["gt"])
        if not torch.isfinite(total):
            total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        self.log("val_loss", total.detach(), sync_dist=True)
        self.log("val_psnr", psnr_i, sync_dist=True)
        self.log("val_ssim", ssim_i, sync_dist=True)
        if batch_idx == 0:
            self._save_val_vis(batch, pred)
        return {"psnr": psnr_i, "ssim": ssim_i}

    def _save_val_vis(self, batch, pred):
        if not self.best_ckpt_dir:
            return
        vis_dir = Path(self.best_ckpt_dir).parent / "val_vis" / f"epoch_{self.current_epoch:03d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        max_vis = min(int(self.args.val_vis_count), pred.shape[0])
        for i in range(max_vis):
            sample_id = batch["id"][i]
            save_image(torch.clamp(batch["input"][i], 0.0, 1.0), vis_dir / f"{sample_id}_input.png")
            save_image(torch.clamp(batch["vit"][i], 0.0, 1.0), vis_dir / f"{sample_id}_vit.png")
            save_image(torch.clamp(pred[i], 0.0, 1.0), vis_dir / f"{sample_id}_refined.png")
            save_image(torch.clamp(batch["gt"][i], 0.0, 1.0), vis_dir / f"{sample_id}_gt.png")

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
        print(f"\n[Refine][Epoch {self.current_epoch}] PSNR={avg_psnr:.4f} SSIM={avg_ssim:.4f} Best={self.best_psnr:.4f}@{self.best_epoch}")

    def _build_optimizer(self, params):
        betas = (float(self.args.adam_beta1), float(self.args.adam_beta2))
        wd = float(self.args.weight_decay)
        if self.args.optimizer_type.lower() == "adamw":
            return optim.AdamW(params, lr=self.args.lr, betas=betas, weight_decay=wd)
        return optim.Adam(params, lr=self.args.lr, betas=betas, weight_decay=wd)

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
        raise ValueError(f"Unsupported scheduler_type: {self.args.scheduler_type}")

    def configure_optimizers(self):
        optimizer = self._build_optimizer(self.net.parameters())
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return [optimizer], [scheduler]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
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
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_type", type=str, default="fixed", choices=["fixed", "cosine", "warmup_cosine"])
    parser.add_argument("--scheduler_warmup_epochs", type=int, default=0)
    parser.add_argument("--scheduler_tmax_epochs", type=int, default=0)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--refine_cache_root", type=str, required=True)
    parser.add_argument("--refine_width", type=int, default=48)
    parser.add_argument("--refine_middle_blocks", type=int, default=6)
    parser.add_argument("--refine_enc_blocks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--refine_dec_blocks", type=int, nargs="+", default=[1, 1, 2, 2])
    parser.add_argument("--refine_no_global_residual", type=int, default=1)
    parser.add_argument("--loss_char_weight", type=float, default=1.0)
    parser.add_argument("--loss_ssim_weight", type=float, default=0.7)
    parser.add_argument("--val_vis_count", type=int, default=4)
    return parser.parse_args()


def build_loader(args, split, is_train):
    patch_h = args.patch_height if args.patch_height > 0 else 0
    patch_w = args.patch_width if args.patch_width > 0 else 0
    if not is_train:
        patch_h = args.val_patch_height if args.val_patch_height > 0 else 0
        patch_w = args.val_patch_width if args.val_patch_width > 0 else 0
    ds = RefineDataset(
        root=args.refine_cache_root,
        split=split,
        patch_size=args.patch_size if is_train else 0,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log_path = os.path.join(args.ckpt_dir, f"training_log_{timestamp}.txt")
    best_psnr_path = os.path.join(args.ckpt_dir, "best_psnr.txt")
    best_ckpt_dir = os.path.join(args.ckpt_dir, "best")

    logger = TensorBoardLogger(save_dir="logs/") if str(args.wblogger).lower() in ("none", "") else WandbLogger(project=args.wblogger, name="CANDLE-Refine")
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
    model = RefineModel(args, best_psnr_path=best_psnr_path, best_ckpt_dir=best_ckpt_dir)

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
    print(f"\n{'='*60}\nRefine Training Started at {timestamp}\nCheckpoint Dir: {args.ckpt_dir}\nTraining Log: {training_log_path}\nBest PSNR Log: {best_psnr_path}\n{'='*60}\n")

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
            print(f"Loading refine weights only from: {args.resume_from}")
            ckpt = torch.load(args.resume_from, map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[refine][weights-only] missing_keys={missing}")
            if unexpected:
                print(f"[refine][weights-only] unexpected_keys={unexpected}")
        else:
            ckpt_path = args.resume_from
            print(f"Resuming full refine state from: {ckpt_path}")

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader, ckpt_path=ckpt_path)
    print(f"\n{'='*60}\n✅ Refine Training Completed!\nBest PSNR: {model.best_psnr:.4f}\nBest Epoch: {model.best_epoch}\nBest Model saved to: {best_ckpt_dir}/best_model.ckpt\n{'='*60}\n")
    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
