import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn.functional as F

def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = torch.nan_to_num(recoverd.detach(), nan=0.0, posinf=1.0, neginf=0.0)
    clean = torch.nan_to_num(clean.detach(), nan=0.0, posinf=1.0, neginf=0.0)
    recoverd = np.clip(recoverd.cpu().numpy(), 0, 1)
    clean = np.clip(clean.cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0.0
    ssim = 0.0
    valid_count = 0

    for i in range(recoverd.shape[0]):
        try:
            psnr_i = peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
            ssim_i = structural_similarity(clean[i], recoverd[i], data_range=1, channel_axis=2)
        except Exception:
            continue
        if not np.isfinite(psnr_i) or not np.isfinite(ssim_i):
            continue
        psnr += float(psnr_i)
        ssim += float(ssim_i)
        valid_count += 1

    if valid_count == 0:
        return 0.0, 0.0, recoverd.shape[0]
    return psnr / valid_count, ssim / valid_count, valid_count

def pad_to_multiple_of_14(img: torch.Tensor) -> torch.Tensor:
    """
    Pads a tensor [B, C, H, W] so that H and W are multiples of 14.
    """
    B, C, H, W = img.shape
    pad_h = (14 - H % 14) % 14
    pad_w = (14 - W % 14) % 14

    # Pad at the bottom and right
    padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')  # or 'constant'
    return padded_img
