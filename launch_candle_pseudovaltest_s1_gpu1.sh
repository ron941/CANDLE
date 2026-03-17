#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJ=/raid/ron/ALN_final/candle-main-ori-dino-4stage-HVI-edge-final-test-refine
DATA=/raid/ron/ALN_final/dataset/CL3AN_resize2064x1376_nonormal/stage2_ft_official_20260313_valtest1x5crop1024x768_pluspairwise_plusnew_plus12_plusntirevaltestpseudo_20260317
INIT=/raid/ron/ALN_final/candle-main-ori-dino-4stage-HVI-edge-final/train_ckpt_prog75_stage2_ps768_gpu0/best/best_model.ckpt

echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import os
import torch

print("env CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch.cuda.current_device =", torch.cuda.current_device())
    print("torch.cuda.get_device_name(0) =", torch.cuda.get_device_name(0))
PY

python "$PROJ/train.py" \
  --cuda 0 --num_gpus 1 \
  --epochs 16 --batch_size 10 --accumulate_grad_batches 2 --val_batch_size 1 \
  --lr 1e-4 \
  --optimizer_type adam --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.0 \
  --scheduler_type fixed \
  --patch_size 0 --patch_height 256 --patch_width 256 \
  --train_crop_mode random \
  --use_data_aug 1 --aug_hflip_prob 0.5 --aug_vflip_prob 0.5 --aug_rotate90 1 \
  --val_crop_mode none --val_patch_height 0 --val_patch_width 0 \
  --lpips_lambda 0.0 --ssim_lambda 0.7 \
  --query_dim 32 --use_psf_dr 1 --dr_heads 4 --dr_dropout 0.0 --dr_alpha_init 0.0 --psf_gate_hidden 64 \
  --use_sffb_decoder 0 --use_bfacg_decoder 1 --bfacg_variant v1 --bfacg_hidden 64 --bfacg_res_scale_init 0.0 \
  --use_clp_decoder 0 --use_hvi_bottleneck 0 --hvi_consistency_weight 0.0 \
  --save_last_ckpt 1 \
  --resume_from "$INIT" --resume_weights_only 1 \
  --train_input_dir "$DATA/train/input" \
  --train_normals_dir "$DATA/train/input" \
  --train_target_dir "$DATA/train/gt" \
  --train_dino_dir "$DATA/train/dino_tokens32" \
  --test_input_dir "$DATA/test/input" \
  --test_normals_dir "$DATA/test/input" \
  --test_target_dir "$DATA/test/gt" \
  --test_dino_dir "$DATA/test/dino_tokens32" \
  --num_workers 16 \
  --wblogger none \
  --ckpt_dir "$PROJ/train_ckpt_expft_plus12pseudovaltest_s1_ps256_ep16_lr1e4_gpu1_from_prog75stage2best"
