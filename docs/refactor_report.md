# Refactor Report

## Modified Files

- Added audit and report docs: `docs/refactor_audit.md`, `docs/refactor_report.md`.
- Added centralized config utilities: `utils/config.py`.
- Added reproducibility utilities: `utils/seed.py`.
- Added shared model construction: `models/build.py`.
- Added training engine package: `engine/trainer.py`, `engine/checkpoint.py`, `engine/schedulers.py`.
- Added dataset checking tool: `tools/check_dataset.py`.
- Refactored `train.py`, `test.py`, `evaluate.py`, `profile_model.py`.
- Updated `data/dataset.py`, `utils/logger.py`, `utils/metrics.py`, `utils/timer.py`.
- Updated all core configs and added `configs/dwconv9_horse2zebra.yaml`.
- Added scripts for DWConv9, both-direction testing, and profiling.
- Updated `README.md`, `docs/autodl_guide.md`, and `docs/reproduction_notes.md`.

## Structure Improvements

- Moved model construction out of entry scripts into `models/build.py`.
- Moved CycleGAN optimization logic into `CycleGANTrainer`.
- Moved checkpoint save/load and scheduler policy into `engine/`.
- Centralized YAML loading, validation, default filling, CLI overrides, DataLoader kwargs, and resolved-config output in `utils/config.py`.
- Kept entry scripts focused on CLI, config, orchestration, and reporting.

## Efficiency Improvements

- Added RTX 4080 SUPER-oriented defaults: `num_workers=8`, `pin_memory=true`, `persistent_workers=true`, `prefetch_factor=4`.
- Added CUDA `non_blocking=True` tensor transfers.
- Added configurable cuDNN benchmark/deterministic flags.
- Added optional AMP with `use_amp: false` by default.
- Added warmup and CUDA synchronization to profile timing, with average and standard deviation output.
- Added test-time elapsed time and average time per image.
- Avoided unbounded tensor/loss accumulation by aggregating Python floats only.
- Kept sample saving frequency configurable.

## Preserved Thesis Semantics

- Baseline remains standard CycleGAN with `ResnetGenerator`, 9 residual blocks, LSGAN, cycle loss, identity loss, and 70x70 PatchGAN.
- Lightweight remains CycleGAN with depthwise separable generator convolutions and 6 residual blocks.
- Discriminator, losses, data augmentation, optimizer, LR schedule, batch size, image buffer, and default loss weights remain aligned with the thesis.
- `batch_size=1` remains the official reproduction setting.
- AMP is optional and disabled by default.

## Remaining Work

- Install PyTorch and dependencies before runtime validation; the current local environment lacks `torch`.
- Place datasets under `datasets/horse2zebra` and `datasets/apple2orange`.
- Run CPU/CUDA profile after dependencies are installed.
- Run `bash scripts/train_debug.sh` before full 200-epoch training.
- Add a project-approved CMMD implementation if exact CMMD reporting is required.

## Current Local Debug

```bash
python -m py_compile train.py test.py evaluate.py profile_model.py tools/check_dataset.py
python profile_model.py --model_type baseline --device cpu --repeat 2 --warmup 1
python profile_model.py --model_type lightweight --device cpu --repeat 2 --warmup 1
bash scripts/train_debug.sh
```

## AutoDL Training

```bash
cd /root/autodl-tmp
git clone https://github.com/SunMingYang04/cyclegan-lightweight-thesis.git
cd cyclegan-lightweight-thesis
conda create -n cyclegan_thesis python=3.9 -y
conda activate cyclegan_thesis
pip install -r requirements.txt
chmod +x scripts/*.sh
bash scripts/train_debug.sh
bash scripts/train_baseline_horse2zebra.sh
bash scripts/train_lightweight_horse2zebra.sh
```

## Validation Status

- Python syntax check: passed.
- Script path consistency check: passed.
- Dataset check: skipped because `datasets/horse2zebra` is not present.
- Model import/profile checks: blocked by missing local dependency `torch`.

## Git Recommendation

Do not push until PyTorch is installed and profile/debug checks pass. After validation in a proper environment:

```bash
git add .
git commit -m "refactor: improve CycleGAN project structure and GPU efficiency"
```
