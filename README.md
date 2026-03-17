# CANDLE

Color Ambient Normalization with DINO Layer Enhancement

## Overview

CANDLE is a color ambient normalization framework built on multi-stage DINO feature guidance and a coarse-to-refine restoration pipeline.

This repository contains the training and inference code for:
- backbone training
- refine-stage training
- joint coarse + refine training
- inference and test-time augmentation
- DINO token extraction

Core method elements:
- multi-stage DINO layer fusion
- query-guided feature conditioning
- coarse-to-refine restoration
- ambient light normalization

## Environment

Install dependencies with:

```bash
pip install -r requirements_aln_final.txt
```

Recommended environment:
- Python 3.10
- PyTorch with CUDA
- NVIDIA GPU for training and inference

## Repository Structure

```text
CANDLE/
├── model.py
├── refine_nafnet.py
├── train.py
├── train_refine.py
├── train_refine_joint.py
├── test.py
├── test_tta.py
├── extract_dino_tokens32.py
├── options.py
├── utils/
└── NAFNet/
```

## Main Entry Points

- `train.py`: backbone training
- `train_refine.py`: refine-only training
- `train_refine_joint.py`: joint coarse + refine training
- `test.py`: inference
- `test_tta.py`: inference with test-time augmentation
- `extract_dino_tokens32.py`: DINO token extraction

## Key Files

- `model.py`: CANDLE backbone
- `refine_nafnet.py`: refinement network
- `utils/aln_dataset.py`: dataset loader
- `utils/val_utils.py`: validation metrics
- `utils/schedulers.py`: scheduler helpers

## Data Preparation

Training and inference expect:
- input images
- ground-truth target images
- pre-extracted DINO token files

The DINO features can be prepared with:

```bash
python extract_dino_tokens32.py \
  --input_dir /path/to/images \
  --out_dir /path/to/dino_tokens32 \
  --target_w 1024 \
  --target_h 768 \
  --batch_size 8
```

## Training

Example entry points:

```bash
python train.py ...
python train_refine.py ...
python train_refine_joint.py ...
```

The final joint training pipeline is implemented in `train_refine_joint.py`.

## Inference

Run inference with:

```bash
python test.py ...
```

For test-time augmentation:

```bash
python test_tta.py ...
```

## Supported Optimization Options

- The codebase supports pixel-oriented fine-tuning with `L1`, `MSE`, and `Charbonnier` losses.
- The joint pipeline also supports `fixed`, `cosine`, `warmup_cosine`, and `plateau` learning-rate schedules.

## Repository Scope

This repository is prepared as a code-only package for training and inference.

It intentionally does not include:
- datasets
- training checkpoints
- experiment logs
- visualization outputs

These artifacts are ignored through `.gitignore` and should be managed separately from the source repository.

## Notes

- Update dataset paths and checkpoint paths in your training or inference commands before running.
- Some helper shell scripts are included as references and may need local path adjustment.
