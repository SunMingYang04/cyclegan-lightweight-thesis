# Refactor Audit

## 1. Current Project Structure

```text
cyclegan-lightweight-thesis/
  README.md
  requirements.txt
  .gitignore
  train.py
  test.py
  evaluate.py
  profile_model.py
  configs/
    debug_horse2zebra.yaml
    baseline_horse2zebra.yaml
    lightweight_horse2zebra.yaml
    baseline_apple2orange.yaml
    lightweight_apple2orange.yaml
    ablation_no_identity.yaml
    ablation_no_cycle.yaml
  data/
    dataset.py
    transforms.py
  models/
    generators.py
    discriminators.py
    losses.py
    image_pool.py
  utils/
    logger.py
    visualizer.py
    count_params.py
    timer.py
    metrics.py
  scripts/
    train_*.sh
    test_*.sh
    eval_*.sh
  docs/
    autodl_guide.md
    experiment_log.md
    reproduction_notes.md
```

## 2. Core File Responsibilities

- `train.py`: currently contains config loading, seed setup, model construction, optimizer/scheduler construction, full CycleGAN training loop, checkpoint saving, loss CSV writing, and sample image saving.
- `test.py`: loads one generator checkpoint, runs one direction (`AtoB` or `BtoA`), and saves input/fake images.
- `evaluate.py`: computes FID/KID, optional SSIM/LPIPS through `input_dir`, optional CMMD placeholder, and writes JSON/CSV.
- `profile_model.py`: builds a generator/discriminator pair and measures generator inference time with random input.
- `data/dataset.py`: implements unpaired A/B domain dataset and sorted image listing.
- `data/transforms.py`: implements train/test image transforms.
- `models/generators.py`: defines standard and lightweight generators, including residual blocks and depthwise separable convolutions.
- `models/discriminators.py`: defines 70x70 PatchGAN.
- `models/losses.py`: defines LSGAN loss and L1 helper losses.
- `models/image_pool.py`: defines CycleGAN image history buffer.
- `utils/logger.py`: configures stdout/file logging.
- `utils/visualizer.py`: saves normalized tensors as image grids or single images.
- `utils/count_params.py`: counts parameters.
- `utils/timer.py`: measures forward inference time with CUDA synchronization.
- `utils/metrics.py`: wraps FID/KID/SSIM/LPIPS/CMMD placeholder and metric file output.
- `configs/*.yaml`: stores dataset, model, optimization, and experiment options.
- `scripts/*.sh`: stores convenience commands for training, testing, and evaluation.

## 3. Code Structure Issues

- `train.py` has too many responsibilities: config parsing, model building, checkpointing, schedulers, training state, logging, and optimization are coupled in one file.
- Model construction is duplicated in `train.py`, `test.py`, and `profile_model.py`.
- Config parsing is duplicated and lacks centralized defaults, validation, CLI overrides, and resolved-config output.
- Checkpoint save/resume logic is incomplete: optimizer states are saved, but scheduler states are not saved; resume advances schedulers manually instead of restoring scheduler state.
- Path management is mostly relative and simple, but there is no single resolved experiment directory contract.
- Logging format exists, but metadata such as Git commit, Python/PyTorch versions, CUDA/GPU information, and resolved config are not consistently recorded.
- `scripts` and `README` are mostly aligned, but required scripts for `dwconv9`, `test both`, and profile commands are missing.

## 4. Runtime Efficiency Issues

- DataLoader does not expose optimized defaults for the target RTX 4080 SUPER machine: `persistent_workers`, `prefetch_factor`, and config-driven `pin_memory` are missing.
- CUDA tensor transfers do not use `non_blocking=True`.
- `torch.backends.cudnn.benchmark` is enabled, but deterministic/benchmark settings are not configurable.
- AMP is not supported. It should be optional and default to disabled to preserve thesis semantics.
- Sample saving is configurable by step, but training still saves at least one epoch grid. This is acceptable; no unbounded image cache is present.
- Loss aggregation uses Python floats via `.item()`, which avoids graph accumulation and is safe.
- Test uses `model.eval()` and `torch.no_grad()`, but supports only one direction and lacks timing summary.
- Metric computation streams SSIM/LPIPS pair-by-pair and does not load all images at once, which is good. It lacks `max_images`.
- `profile_model.py` uses CUDA synchronization through `utils/timer.py`, but only reports average time and does not report standard deviation.

## 5. Training Stability Issues

- LR decay formula matches the intended fixed-then-linear policy for normal 200/100 setup.
- ImagePool behavior matches original CycleGAN logic.
- `use_identity` and `use_cycle` switches are implemented.
- Checkpoint content lacks scheduler states and AMP scaler state, so exact resume is incomplete.
- Random seed is configurable, but cuDNN deterministic/benchmark controls are not centralized.
- `use_amp` is not implemented.
- Baseline/lightweight semantics are preserved, but no `dwconv9` model type exists yet.

## 6. Maintainability Issues

- Naming is mostly clear, but config fields are not validated or normalized.
- No `utils/config.py` exists; defaults are spread across entry scripts.
- No `models/build.py` exists; model initialization is not centralized.
- No `engine/` package exists; training engine is not reusable or testable.
- Dataset checking capability is missing.
- Error messages are reasonable in dataset/test, but checkpoint existence checks should be clearer.

## 7. GitHub and AutoDL Deployment Issues

- `requirements.txt` includes required core dependencies.
- `.gitignore` safely ignores datasets, checkpoints, results, weight files, logs, cache, and wandb.
- Existing shell scripts are executable.
- AutoDL instructions exist and support `/root/autodl-tmp`.
- Missing scripts required by the new spec: `train_dwconv9_horse2zebra.sh`, `test_both_lightweight_horse2zebra.sh`, `profile_baseline.sh`, `profile_lightweight.sh`.
- The project can be cloned and configured, but training requires datasets to be placed under `datasets/` first.

## 8. Refactor Priorities

1. Add `utils/config.py` for defaults, validation, path normalization, CLI overrides, and `config_resolved.yaml`.
2. Add `models/build.py` for shared model construction and weight initialization.
3. Add `engine/` with `CycleGANTrainer`, checkpoint helpers, and scheduler helpers.
4. Optimize DataLoader configuration and CUDA transfers without changing thesis semantics.
5. Add `utils/seed.py` and dataset checking.
6. Extend `test.py`, `evaluate.py`, and `profile_model.py` to meet CLI and reporting requirements.
7. Update configs, scripts, README, and AutoDL docs.
