#!/usr/bin/env bash
set -euo pipefail

PROJ=/raid/ron/ALN_final/candle-main-ori-dino-4stage-HVI-edge-final-test-refine
DATA=/raid/ron/ALN_final/dataset/CL3AN_resize2064x1376_nonormal/stage2_ft_official_20260313_valtest1x5crop1024x768_pluspairwise_plusnew_plus12_20260315
VIT_CKPT=/raid/ron/ALN_final/candle-main-ori-dino-4stage-HVI-edge-final-test/train_ckpt_expft_plus12_s3_ps768_ep10_lr2e5_augon_gpu3_from_s2last_bs1x4/best/best_model.ckpt
CACHE=/raid/ron/ALN_final/dataset/CL3AN_resize2064x1376_nonormal/refine_cache_plus12_from_s3best_20260316

# 1) Cache frozen VIT outputs
CUDA_VISIBLE_DEVICES=0 python $PROJ/cache_vit_outputs.py \
  --cuda 0 \
  --batch_size 1 --num_workers 8 \
  --train_patch_size 768 \
  --train_crop_mode random \
  --train_cache_repeats 4 \
  --test_patch_size 0 \
  --test_crop_mode none \
  --frozen_vit_ckpt $VIT_CKPT \
  --cache_root $CACHE \
  --train_input_dir $DATA/train/input \
  --train_normals_dir $DATA/train/input \
  --train_target_dir $DATA/train/gt \
  --train_dino_dir $DATA/train/dino_tokens32 \
  --test_input_dir $DATA/test/input \
  --test_normals_dir $DATA/test/input \
  --test_target_dir $DATA/test/gt \
  --test_dino_dir $DATA/test/dino_tokens32

# 2) Refine Stage 1
CUDA_VISIBLE_DEVICES=0 python $PROJ/train_refine.py \
  --cuda 0 --num_gpus 1 \
  --epochs 12 --batch_size 8 --accumulate_grad_batches 1 --val_batch_size 1 \
  --lr 1e-4 \
  --optimizer_type adam --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.0 \
  --scheduler_type fixed \
  --patch_size 256 --patch_height 0 --patch_width 0 \
  --train_crop_mode random \
  --use_data_aug 1 --aug_hflip_prob 0.5 --aug_vflip_prob 0.5 --aug_rotate90 1 \
  --val_crop_mode none --val_patch_height 0 --val_patch_width 0 \
  --refine_cache_root $CACHE \
  --refine_width 48 --refine_middle_blocks 6 \
  --refine_enc_blocks 1 2 4 8 \
  --refine_dec_blocks 1 1 2 2 \
  --refine_no_global_residual 1 \
  --loss_char_weight 1.0 --loss_ssim_weight 0.7 \
  --wblogger none \
  --ckpt_dir $PROJ/train_ckpt_refine_plus12_r1_ps256_ep12_lr1e4_gpu0

# 3) Refine Stage 2
R1_BEST=$PROJ/train_ckpt_refine_plus12_r1_ps256_ep12_lr1e4_gpu0/best/best_model.ckpt
CUDA_VISIBLE_DEVICES=0 python $PROJ/train_refine.py \
  --cuda 0 --num_gpus 1 \
  --epochs 10 --batch_size 2 --accumulate_grad_batches 2 --val_batch_size 1 \
  --lr 5e-5 \
  --optimizer_type adam --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.0 \
  --scheduler_type cosine \
  --patch_size 512 --patch_height 0 --patch_width 0 \
  --train_crop_mode random \
  --use_data_aug 1 --aug_hflip_prob 0.5 --aug_vflip_prob 0.5 --aug_rotate90 1 \
  --val_crop_mode none --val_patch_height 0 --val_patch_width 0 \
  --refine_cache_root $CACHE \
  --refine_width 48 --refine_middle_blocks 6 \
  --refine_enc_blocks 1 2 4 8 \
  --refine_dec_blocks 1 1 2 2 \
  --refine_no_global_residual 1 \
  --loss_char_weight 1.0 --loss_ssim_weight 0.7 \
  --resume_from $R1_BEST --resume_weights_only 1 \
  --wblogger none \
  --ckpt_dir $PROJ/train_ckpt_refine_plus12_r2_ps512_ep10_lr5e5_gpu0_from_r1best

# 4) Refine Stage 3
R2_BEST=$PROJ/train_ckpt_refine_plus12_r2_ps512_ep10_lr5e5_gpu0_from_r1best/best/best_model.ckpt
CUDA_VISIBLE_DEVICES=0 python $PROJ/train_refine.py \
  --cuda 0 --num_gpus 1 \
  --epochs 8 --batch_size 1 --accumulate_grad_batches 4 --val_batch_size 1 \
  --lr 2e-5 \
  --optimizer_type adam --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.0 \
  --scheduler_type cosine \
  --patch_size 768 --patch_height 0 --patch_width 0 \
  --train_crop_mode random \
  --use_data_aug 1 --aug_hflip_prob 0.5 --aug_vflip_prob 0.5 --aug_rotate90 1 \
  --val_crop_mode none --val_patch_height 0 --val_patch_width 0 \
  --refine_cache_root $CACHE \
  --refine_width 48 --refine_middle_blocks 6 \
  --refine_enc_blocks 1 2 4 8 \
  --refine_dec_blocks 1 1 2 2 \
  --refine_no_global_residual 1 \
  --loss_char_weight 1.0 --loss_ssim_weight 0.7 \
  --resume_from $R2_BEST --resume_weights_only 1 \
  --wblogger none \
  --ckpt_dir $PROJ/train_ckpt_refine_plus12_r3_ps768_ep8_lr2e5_gpu0_from_r2best
