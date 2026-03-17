# CANDLE

Color Ambient Normalization with DINO Layer Enhancement

## Overview

This repository contains the training and inference code for the CANDLE framework.

Core method elements:
- multi-stage DINO layer fusion
- query-guided feature conditioning
- coarse-to-refine restoration
- ambient light normalization

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

## Notes

- The final joint training pipeline is implemented in `train_refine_joint.py`.
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
