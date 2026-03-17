#!/usr/bin/env bash
set -euo pipefail

PROJ=/raid/ron/ALN_final/candle-main-ori-dino-4stage-HVI-edge-final
DATA=/raid/ron/ALN_final/dataset/CL3AN_resize2064x1376_nonormal/stage2_ft_official_20260313
INIT=/raid/ron/ALN_final/candle-main-ori-dino-4stage-HVI-edge-final/train_ckpt_prog75_stage2_ps768_gpu0/best/best_model.ckpt

S1_DIR=$PROJ/train_ckpt_stage2formal_s1_ps256_ep12_lr1e4_gpu0
S2_DIR=$PROJ/train_ckpt_stage2formal_s2_ps512_ep10_lr5e5_gpu0_from_s1
S3_DIR=$PROJ/train_ckpt_stage2formal_s3_ps768_ep8_lr2e5_gpu0_from_s2

CUDA_VISIBLE_DEVICES=0 python $PROJ/train.py \
  --cuda 0 --num_gpus 1 \
  --epochs 12 --batch_size 2 --accumulate_grad_batches 2 --val_batch_size 1 \
  --lr 1e-4 \
  --optimizer_type adam --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.0 \
  --scheduler_type fixed \
  --patch_size 256 --patch_height 0 --patch_width 0 \
  --train_crop_mode random \
  --use_data_aug 0 --aug_hflip_prob 0.0 --aug_vflip_prob 0.0 --aug_rotate90 0 \
  --val_crop_mode none --val_patch_height 0 --val_patch_width 0 \
  --lpips_lambda 0.0 --ssim_lambda 0.7 \
  --query_dim 32 --use_psf_dr 1 --dr_heads 4 --dr_dropout 0.0 --dr_alpha_init 0.0 --psf_gate_hidden 64 \
  --use_sffb_decoder 0 --use_bfacg_decoder 1 --bfacg_variant v1 --bfacg_hidden 64 --bfacg_res_scale_init 0.0 \
  --use_clp_decoder 0 --use_hvi_bottleneck 0 --hvi_consistency_weight 0.0 \
  --save_last_ckpt 1 \
  --resume_from $INIT --resume_weights_only 1 \
  --train_input_dir $DATA/train/input \
  --train_normals_dir $DATA/train/input \
  --train_target_dir $DATA/train/gt \
  --train_dino_dir $DATA/train/dino_tokens32 \
  --test_input_dir $DATA/test/input \
  --test_normals_dir $DATA/test/input \
  --test_target_dir $DATA/test/gt \
  --test_dino_dir $DATA/test/dino_tokens32 \
  --num_workers 16 \
  --wblogger none \
  --ckpt_dir $S1_DIR

S1_BEST=$S1_DIR/best/best_model.ckpt

CUDA_VISIBLE_DEVICES=0 python $PROJ/train.py \
  --cuda 0 --num_gpus 1 \
  --epochs 10 --batch_size 2 --accumulate_grad_batches 2 --val_batch_size 1 \
  --lr 5e-5 \
  --optimizer_type adam --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.0 \
  --scheduler_type cosine \
  --patch_size 512 --patch_height 0 --patch_width 0 \
  --train_crop_mode random \
  --use_data_aug 0 --aug_hflip_prob 0.0 --aug_vflip_prob 0.0 --aug_rotate90 0 \
  --val_crop_mode none --val_patch_height 0 --val_patch_width 0 \
  --lpips_lambda 0.0 --ssim_lambda 0.7 \
  --query_dim 32 --use_psf_dr 1 --dr_heads 4 --dr_dropout 0.0 --dr_alpha_init 0.0 --psf_gate_hidden 64 \
  --use_sffb_decoder 0 --use_bfacg_decoder 1 --bfacg_variant v1 --bfacg_hidden 64 --bfacg_res_scale_init 0.0 \
  --use_clp_decoder 0 --use_hvi_bottleneck 0 --hvi_consistency_weight 0.0 \
  --save_last_ckpt 1 \
  --resume_from $S1_BEST --resume_weights_only 1 \
  --train_input_dir $DATA/train/input \
  --train_normals_dir $DATA/train/input \
  --train_target_dir $DATA/train/gt \
  --train_dino_dir $DATA/train/dino_tokens32 \
  --test_input_dir $DATA/test/input \
  --test_normals_dir $DATA/test/input \
  --test_target_dir $DATA/test/gt \
  --test_dino_dir $DATA/test/dino_tokens32 \
  --num_workers 16 \
  --wblogger none \
  --ckpt_dir $S2_DIR

S2_BEST=$S2_DIR/best/best_model.ckpt

CUDA_VISIBLE_DEVICES=0 python $PROJ/train.py \
  --cuda 0 --num_gpus 1 \
  --epochs 8 --batch_size 2 --accumulate_grad_batches 2 --val_batch_size 1 \
  --lr 2e-5 \
  --optimizer_type adam --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.0 \
  --scheduler_type cosine \
  --patch_size 768 --patch_height 0 --patch_width 0 \
  --train_crop_mode random \
  --use_data_aug 0 --aug_hflip_prob 0.0 --aug_vflip_prob 0.0 --aug_rotate90 0 \
  --val_crop_mode none --val_patch_height 0 --val_patch_width 0 \
  --lpips_lambda 0.0 --ssim_lambda 0.7 \
  --query_dim 32 --use_psf_dr 1 --dr_heads 4 --dr_dropout 0.0 --dr_alpha_init 0.0 --psf_gate_hidden 64 \
  --use_sffb_decoder 0 --use_bfacg_decoder 1 --bfacg_variant v1 --bfacg_hidden 64 --bfacg_res_scale_init 0.0 \
  --use_clp_decoder 0 --use_hvi_bottleneck 0 --hvi_consistency_weight 0.0 \
  --save_last_ckpt 1 \
  --resume_from $S2_BEST --resume_weights_only 1 \
  --train_input_dir $DATA/train/input \
  --train_normals_dir $DATA/train/input \
  --train_target_dir $DATA/train/gt \
  --train_dino_dir $DATA/train/dino_tokens32 \
  --test_input_dir $DATA/test/input \
  --test_normals_dir $DATA/test/input \
  --test_target_dir $DATA/test/gt \
  --test_dino_dir $DATA/test/dino_tokens32 \
  --num_workers 16 \
  --wblogger none \
  --ckpt_dir $S3_DIR
