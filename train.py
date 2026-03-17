import subprocess
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

from utils.aln_dataset import ALNDatasetGeom
from model import CANDLE, DynamicAnalyticalHVI, LowFreqChromaBiasNet
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.val_utils import compute_psnr_ssim
from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


def L2_histo(x, y):
    bins = x.size(1)
    r = torch.arange(bins, device=x.device)
    s, t = torch.meshgrid(r, r, indexing='ij')
    tt = (t >= s).to(dtype=x.dtype, device=x.device)
    cdf_x = torch.matmul(x, tt)
    cdf_y = torch.matmul(y, tt)
    return torch.sum(torch.square(cdf_x - cdf_y), dim=1)


class CANDLEModel(pl.LightningModule):
    def __init__(self, best_psnr_path=None, best_ckpt_dir=None):
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
        self.use_abc_ica = bool(opt.use_abc_ica)
        if self.use_lowfreq_bias_baseline and self.use_abc_ica:
            raise ValueError("use_lowfreq_bias_baseline and use_abc_ica cannot be enabled together.")
        self.automatic_optimization = not self.use_abc_ica
        self.l1_loss  = nn.L1Loss()
        self.lpips_loss = LPIPS(net="vgg").eval()
        self.lpips_loss.requires_grad_(False)
        self.ssim_loss = SSIM(data_range=1.0)
        self.lpips_lambda = float(opt.lpips_lambda)
        self.ssim_lambda = float(opt.ssim_lambda)
        self.abc_hist_loss_w = float(opt.abc_hist_loss_w)
        self.abc_lab_loss_w = float(opt.abc_lab_loss_w)
        self.lfb_bias_loss_w = float(opt.lfb_bias_loss_w)
        self.lfb_recon_loss_w = float(opt.lfb_recon_loss_w)
        self.hvi_consistency_weight = float(opt.hvi_consistency_weight)
        self.hvi_consistency = DynamicAnalyticalHVI(eps=opt.hvi_eps)
        self.hvi_consistency.requires_grad_(False)
        self.best_psnr = -1
        self.best_psnr_path = best_psnr_path
        self.best_ckpt_dir = best_ckpt_dir
        self.best_epoch = -1
        
    def setup(self, stage=None):
        self.lpips_loss = self.lpips_loss.to(self.device)
        self.hvi_consistency = self.hvi_consistency.to(self.device)
    
    def forward(self, x, dino_tokens, input_hist=None, input_lab_hist=None):
        if self.use_lowfreq_bias_baseline:
            return self.net(x)
        return self.net(x, dino_tokens, input_hist=input_hist, input_lab_hist=input_lab_hist)

    def _unpack_batch(self, batch):
        if len(batch) == 4:
            ([clean_name, de_id], degrad_patch, dino_patch, clean_patch) = batch
            return {
                "id": [clean_name, de_id],
                "degrad": degrad_patch,
                "dino": dino_patch,
                "clean": clean_patch,
                "input_hist": None,
                "target_hist": None,
                "input_lab_hist": None,
                "target_lab_hist": None,
            }
        if len(batch) == 8:
            (
                [clean_name, de_id],
                degrad_patch,
                dino_patch,
                clean_patch,
                input_hist,
                target_hist,
                input_lab_hist,
                target_lab_hist,
            ) = batch
            return {
                "id": [clean_name, de_id],
                "degrad": degrad_patch,
                "dino": dino_patch,
                "clean": clean_patch,
                "input_hist": input_hist,
                "target_hist": target_hist,
                "input_lab_hist": input_lab_hist,
                "target_lab_hist": target_lab_hist,
            }
        raise ValueError(f"Unexpected batch format with length={len(batch)}")

    def _compute_recon_losses(self, restored, clean_patch, batch_idx):
        l1_loss = self.l1_loss(restored, clean_patch).mean()
        lpips_loss = self.lpips_loss(restored * 2.0 - 1.0, clean_patch * 2.0 - 1.0).mean()
        ssim_loss = (1 - self.ssim_loss(restored, clean_patch)).mean()
        hvi_consistency_loss = torch.tensor(0.0, device=restored.device, dtype=restored.dtype)

        if self.hvi_consistency_weight > 0.0:
            h_r, v_r, i_r, _ = self.hvi_consistency.forward_hvit(restored)
            with torch.no_grad():
                h_t, v_t, i_t, _ = self.hvi_consistency.forward_hvit(clean_patch)
            hvi_consistency_loss = (
                self.l1_loss(h_r, h_t) +
                self.l1_loss(v_r, v_t) +
                self.l1_loss(i_r, i_t)
            ) / 3.0
            hvi_consistency_loss = hvi_consistency_loss.mean()

        if not torch.isfinite(l1_loss):
            print(f"[Warn][train] non-finite l1_loss at batch {batch_idx}, set to 0.")
            l1_loss = torch.nan_to_num(l1_loss, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(lpips_loss):
            print(f"[Warn][train] non-finite lpips_loss at batch {batch_idx}, set to 0.")
            lpips_loss = torch.nan_to_num(lpips_loss, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(ssim_loss):
            print(f"[Warn][train] non-finite ssim_loss at batch {batch_idx}, set to 0.")
            ssim_loss = torch.nan_to_num(ssim_loss, nan=0.0, posinf=0.0, neginf=0.0)

        total_loss = (
            l1_loss
            + self.lpips_lambda * lpips_loss
            + self.ssim_lambda * ssim_loss
            + self.hvi_consistency_weight * hvi_consistency_loss
        )
        if not torch.isfinite(total_loss):
            print(f"[Warn][train] non-finite total_loss at batch {batch_idx}, forcing to 0.")
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return l1_loss, lpips_loss, ssim_loss, hvi_consistency_loss, total_loss
    
    def training_step(self, batch, batch_idx):
        batch_data = self._unpack_batch(batch)
        degrad_patch = batch_data["degrad"]
        dino_patch = batch_data["dino"]
        clean_patch = torch.clamp(batch_data["clean"], 0.0, 1.0)

        if self.use_lowfreq_bias_baseline:
            restored, s_pred = self.net(degrad_patch, return_bias=True)
            restored = torch.clamp(restored, 0.0, 1.0)
            s_gt = self.net.build_target_bias(degrad_patch, clean_patch)
            s_gt = torch.clamp(s_gt, min=-self.net.max_bias, max=self.net.max_bias)
            bias_loss = self.l1_loss(s_pred, s_gt).mean()
            # Keep recon term as a monitor only; backprop through Lab->RGB conversion is numerically unstable.
            recon_loss = self.l1_loss(restored.detach(), clean_patch).mean()
            total_loss = self.lfb_bias_loss_w * bias_loss + self.lfb_recon_loss_w * recon_loss
            if not torch.isfinite(total_loss):
                total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
            self.log("lfb_bias_loss", bias_loss.detach(), sync_dist=True)
            self.log("lfb_recon_loss", recon_loss.detach(), sync_dist=True)
            self.log("total_loss", total_loss.detach(), sync_dist=True)
            return total_loss

        if self.use_abc_ica:
            input_hist = batch_data["input_hist"]
            target_hist = batch_data["target_hist"]
            input_lab_hist = batch_data["input_lab_hist"]
            target_lab_hist = batch_data["target_lab_hist"]
            if input_hist is None or target_hist is None or input_lab_hist is None or target_lab_hist is None:
                raise ValueError("ABC-ICA training requires input/target RGB+LAB histograms in the batch.")

            opt_hist, opt_lab, opt_main = self.optimizers()

            # 1) Hist branch update
            opt_hist.zero_grad()
            pred_hist, hist_raw_weights = self.net.abc_aux_branch.forward_hist(input_hist)
            hist_loss = (
                L2_histo(pred_hist[:, 0], target_hist[:, 0]) +
                L2_histo(pred_hist[:, 1], target_hist[:, 1]) +
                L2_histo(pred_hist[:, 2], target_hist[:, 2])
            ).mean()
            hist_loss_w = self.abc_hist_loss_w * hist_loss
            self.manual_backward(hist_loss_w)
            opt_hist.step()

            # 2) Lab branch update
            opt_lab.zero_grad()
            pred_lab, lab_raw_weights = self.net.abc_aux_branch.forward_lab(input_lab_hist)
            lab_loss = (
                L2_histo(pred_lab[:, 0], target_lab_hist[:, 0]) +
                L2_histo(pred_lab[:, 1], target_lab_hist[:, 1]) +
                L2_histo(pred_lab[:, 2], target_lab_hist[:, 2])
            ).mean()
            lab_loss_w = self.abc_lab_loss_w * lab_loss
            self.manual_backward(lab_loss_w)
            opt_lab.step()

            # 3) Main branch update (aux weights are detached in CANDLE by default)
            if self.net.abc_detach_aux_weight:
                hist_for_main = hist_raw_weights
                lab_for_main = lab_raw_weights
            else:
                _, _, hist_for_main, lab_for_main = self.net.forward_abc_aux(input_hist, input_lab_hist)

            opt_main.zero_grad()
            restored = self.net(
                degrad_patch,
                dino_patch,
                input_hist=input_hist,
                input_lab_hist=input_lab_hist,
                hist_raw_weights=hist_for_main,
                lab_raw_weights=lab_for_main,
            )
            restored = torch.clamp(restored, 0.0, 1.0)
            l1_loss, lpips_loss, ssim_loss, hvi_consistency_loss, total_loss = self._compute_recon_losses(
                restored, clean_patch, batch_idx
            )
            self.manual_backward(total_loss)
            opt_main.step()

            self.log("abc_hist_loss", hist_loss.detach(), sync_dist=True, prog_bar=False)
            self.log("abc_lab_loss", lab_loss.detach(), sync_dist=True, prog_bar=False)
        else:
            restored = self.net(degrad_patch, dino_patch)
            restored = torch.clamp(restored, 0.0, 1.0)
            l1_loss, lpips_loss, ssim_loss, hvi_consistency_loss, total_loss = self._compute_recon_losses(
                restored, clean_patch, batch_idx
            )

        # shared logs
        self.log("l1_loss", l1_loss.detach(), sync_dist=True)
        self.log("lpips_loss", lpips_loss.detach(), sync_dist=True)
        self.log("ssim_loss", ssim_loss.detach(), sync_dist=True)
        self.log("hvi_consistency_loss", hvi_consistency_loss.detach(), sync_dist=True)
        self.log("total_loss", total_loss.detach(), sync_dist=True)

        return total_loss

    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        batch_data = self._unpack_batch(batch)
        [clean_name, de_id] = batch_data["id"]
        degrad_patch = batch_data["degrad"]
        dino_patch = batch_data["dino"]
        target_patch = batch_data["clean"]
        input_hist = batch_data["input_hist"]
        input_lab_hist = batch_data["input_lab_hist"]

        with torch.no_grad():
            if self.use_lowfreq_bias_baseline:
                restored = self.net(degrad_patch)
            elif self.use_abc_ica:
                restored = self.net(
                    degrad_patch,
                    dino_patch,
                    input_hist=input_hist,
                    input_lab_hist=input_lab_hist,
                )
            else:
                restored = self.net(degrad_patch, dino_patch)
            restored = torch.clamp(restored, 0.0, 1.0)
            target_patch = torch.clamp(target_patch, 0.0, 1.0)

        if not torch.isfinite(restored).all():
            sample_id = clean_name[0] if isinstance(clean_name, list) and len(clean_name) > 0 else str(clean_name)
            print(f"[Warn][val] non-finite restored tensor at batch {batch_idx}, sample={sample_id}")
            restored = torch.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)

        psnr_i, ssim_i, _ = compute_psnr_ssim(restored, target_patch)
        if not np.isfinite(psnr_i):
            print(f"[Warn][val] non-finite PSNR at batch {batch_idx}, fallback to 0.")
            psnr_i = 0.0
        if not np.isfinite(ssim_i):
            print(f"[Warn][val] non-finite SSIM at batch {batch_idx}, fallback to 0.")
            ssim_i = 0.0

        self.log("val_psnr", psnr_i, sync_dist=True)
        self.log("val_ssim", ssim_i, sync_dist=True)
        
        return {"psnr": psnr_i, "ssim": ssim_i}
    
    def on_validation_epoch_end(self):
        """在每个验证 epoch 结束后调用"""
        # 获取所有验证指标的平均值
        avg_psnr = self.trainer.callback_metrics.get("val_psnr", torch.tensor(0.0))
        avg_ssim = self.trainer.callback_metrics.get("val_ssim", torch.tensor(0.0))
        
        if isinstance(avg_psnr, torch.Tensor):
            avg_psnr = avg_psnr.item()
        if isinstance(avg_ssim, torch.Tensor):
            avg_ssim = avg_ssim.item()
        
        current_epoch = self.current_epoch
        
        # ✅ 打印当前 epoch 的验证指标
        print(f"\n📊 Epoch {current_epoch} Validation Results:")
        print(f"   PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
        
        # 追踪 best PSNR
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.best_epoch = current_epoch
            
            # 仅更新 best 指标记录；checkpoint 由 ModelCheckpoint 统一管理
            if self.best_ckpt_dir is not None:
                # 保存 best_psnr.txt
                if self.best_psnr_path is not None:
                    with open(self.best_psnr_path, 'w') as f:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"Best PSNR: {self.best_psnr:.4f}\n")
                        f.write(f"Best SSIM: {avg_ssim:.4f}\n")
                        f.write(f"Epoch: {self.best_epoch}\n")
                        f.write(f"Time: {timestamp}\n\n")
                
                print(f"   ✅ New Best PSNR: {self.best_psnr:.4f} (saved)")
        else:
            print(f"   Best PSNR: {self.best_psnr:.4f} at Epoch {self.best_epoch}")



    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)

    def _build_optimizer(self, params, lr=None):
        lr = float(opt.lr if lr is None else lr)
        betas = (float(opt.adam_beta1), float(opt.adam_beta2))
        wd = float(opt.weight_decay)
        if str(opt.optimizer_type).lower() == "adamw":
            return optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
        return optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)

    def _build_scheduler(self, optimizer):
        scheduler_type = str(opt.scheduler_type).lower()
        if scheduler_type == "fixed":
            return None
        t_max = int(opt.scheduler_tmax_epochs) if int(opt.scheduler_tmax_epochs) > 0 else int(opt.epochs)
        t_max = max(1, t_max)
        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                optimizer=optimizer,
                T_max=t_max,
                eta_min=float(opt.scheduler_min_lr),
            )
        if scheduler_type == "warmup_cosine":
            warmup_epochs = max(0, int(opt.scheduler_warmup_epochs))
            return LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=t_max,
            )
        raise ValueError(f"Unsupported scheduler_type: {opt.scheduler_type}")
    
    def configure_optimizers(self):
        if self.use_lowfreq_bias_baseline:
            optimizer = self._build_optimizer(self.net.parameters())
            scheduler = self._build_scheduler(optimizer)
            if scheduler is None:
                return optimizer
            return [optimizer], [scheduler]

        if self.use_abc_ica:
            hist_module, lab_module = self.net.get_abc_aux_modules()
            if hist_module is None or lab_module is None:
                raise RuntimeError("ABC auxiliary modules are not initialized.")
            hist_params = list(hist_module.parameters())
            lab_params = list(lab_module.parameters())
            aux_param_ids = {id(p) for p in hist_params + lab_params}
            main_params = [p for p in self.net.parameters() if id(p) not in aux_param_ids]

            opt_hist = self._build_optimizer(hist_params)
            opt_lab = self._build_optimizer(lab_params)
            opt_main = self._build_optimizer(main_params)
            scheduler = self._build_scheduler(opt_main)
            if scheduler is None:
                return [opt_hist, opt_lab, opt_main]
            return [opt_hist, opt_lab, opt_main], [scheduler]

        optimizer = self._build_optimizer(self.net.parameters())
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return [optimizer], [scheduler]

def main():
    print("Options")
    print(opt)

    external_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if external_visible_devices:
        print(f"Using externally provided CUDA_VISIBLE_DEVICES={external_visible_devices}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
        print(f"CUDA_VISIBLE_DEVICES not preset; fallback to opt.cuda={opt.cuda}")
    
    # ✅ 创建检查点目录并准备日志
    os.makedirs(opt.ckpt_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log_path = os.path.join(opt.ckpt_dir, f"training_log_{timestamp}.txt")
    best_psnr_path = os.path.join(opt.ckpt_dir, "best_psnr.txt")
    best_ckpt_dir = os.path.join(opt.ckpt_dir, "best")

    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="CANDLE-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    # ✅ 只保存 best checkpoint，不保存每个 epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath=best_ckpt_dir,
        filename="best_model",
        monitor="val_psnr",
        mode="max",
        save_top_k=1,
        save_last=bool(opt.save_last_ckpt),
        verbose=True
    )

    trainset = ALNDatasetGeom(input_folder=opt.train_input_dir,
                               geom_folder=opt.train_normals_dir,
                               target_folder=opt.train_target_dir,
                               resize_width_to=opt.resize_width,
                               patch_size=opt.patch_size,
                               patch_height=opt.patch_height,
                               patch_width=opt.patch_width,
                               crop_mode=opt.train_crop_mode,
                               crop_x=opt.train_crop_x,
                               crop_y=opt.train_crop_y,
                               dino_folder=opt.train_dino_dir,
                               dino_suffix=opt.dino_suffix,
                               dino_dim=opt.dino_dim,
                               dino_crop_manifest=opt.train_dino_crop_manifest,
                               dino_ref_width=opt.dino_ref_width,
                               dino_ref_height=opt.dino_ref_height,
                               enable_aug=bool(opt.use_data_aug),
                               aug_hflip_prob=opt.aug_hflip_prob,
                               aug_vflip_prob=opt.aug_vflip_prob,
                               aug_rotate90=bool(opt.aug_rotate90),
                               disable_aug_if_fullres=bool(opt.disable_aug_if_fullres),
                               allow_fallback_normal=True,
                               return_aux_hist=bool(opt.use_abc_ica and not opt.use_lowfreq_bias_baseline))

    val_patch_h = int(opt.val_patch_height) if int(opt.val_patch_height) > 0 else None
    val_patch_w = int(opt.val_patch_width) if int(opt.val_patch_width) > 0 else None
    testset = ALNDatasetGeom(input_folder=opt.test_input_dir,
                             geom_folder=opt.test_normals_dir,
                             target_folder=opt.test_target_dir,
                             patch_height=val_patch_h,
                             patch_width=val_patch_w,
                             crop_mode=opt.val_crop_mode,
                             crop_x=opt.val_crop_x,
                             crop_y=opt.val_crop_y,
                             dino_folder=opt.test_dino_dir,
                             dino_suffix=opt.dino_suffix,
                             dino_dim=opt.dino_dim,
                             dino_crop_manifest=opt.test_dino_crop_manifest,
                             dino_ref_width=opt.dino_ref_width,
                             dino_ref_height=opt.dino_ref_height,
                             enable_aug=False,
                             allow_fallback_normal=False,
                             return_aux_hist=bool(opt.use_abc_ica and not opt.use_lowfreq_bias_baseline))
    
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    testloader = DataLoader(testset, batch_size=opt.val_batch_size, pin_memory=True, shuffle=False, num_workers=opt.num_workers)
    
    # ✅ 传递 best PSNR 追踪参数
    model = CANDLEModel(best_psnr_path=best_psnr_path, best_ckpt_dir=best_ckpt_dir)
    
    # ✅ 将训练日志重定向到文件
    import sys
    log_file = open(training_log_path, 'w', buffering=1)
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
    
    print(f"\n{'='*60}")
    print(f"Training Started at {timestamp}")
    print(f"Checkpoint Dir: {opt.ckpt_dir}")
    print(f"Training Log: {training_log_path}")
    print(f"Best PSNR Log: {best_psnr_path}")
    print(f"{'='*60}\n")
    
    precision_mode = 32 if bool(opt.use_lowfreq_bias_baseline) else "16-mixed"
    trainer = pl.Trainer(
        precision=precision_mode,
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        accumulate_grad_batches=max(1, int(opt.accumulate_grad_batches)),
        gradient_clip_val=1.0 if bool(opt.use_lowfreq_bias_baseline) else 0.0,
        gradient_clip_algorithm="norm" if bool(opt.use_lowfreq_bias_baseline) else None,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,   # ✅ 每 1 个 epoch 做一次 validation（全量测试）
    )
    
    # Resume behavior
    ckpt_path = None
    if opt.resume_from:
        if bool(opt.resume_weights_only):
            print(f"Loading weights only from: {opt.resume_from}")
            ckpt = torch.load(opt.resume_from, map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[weights-only] missing_keys={missing}")
            if unexpected:
                print(f"[weights-only] unexpected_keys={unexpected}")
        else:
            ckpt_path = opt.resume_from
            print(f"Resuming full training state from: {ckpt_path}")

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader, ckpt_path=ckpt_path)
    
    # ✅ 训练完成后的总结
    print(f"\n{'='*60}")
    print(f"✅ Training Completed!")
    print(f"Best PSNR: {model.best_psnr:.4f}")
    print(f"Best Epoch: {model.best_epoch}")
    print(f"Best Model saved to: {best_ckpt_dir}/best_model.ckpt")
    print(f"Best PSNR log: {best_psnr_path}")
    print(f"Training log: {training_log_path}")
    print(f"{'='*60}\n")
    
    # ✅ 恢复标准输出
    sys.stdout = original_stdout
    log_file.close()



if __name__ == '__main__':
    main()
