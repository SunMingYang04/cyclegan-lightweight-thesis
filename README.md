# CycleGAN Lightweight Thesis Reproduction

PyTorch reproduction project for the undergraduate thesis **《基于CycleGAN跨域图像转换的研究和实现》**.

Remote repository preset:

```bash
https://github.com/SunMingYang04/cyclegan-lightweight-thesis.git
```

## Project Overview

This repository implements standard CycleGAN and a lightweight generator variant for unpaired image-to-image translation. The main dataset is `horse2zebra`, with `apple2orange` as a supplementary dataset.

## Thesis Tasks

- Baseline: standard CycleGAN.
- Translation tasks: `horse2zebra` and `apple2orange`.
- Generator: encoder, residual transformer, decoder.
- Discriminator: 70x70 PatchGAN.
- Losses: LSGAN adversarial loss, cycle consistency loss, identity loss.
- Training: batch size 1, Adam, 200 epochs, first 100 fixed LR, last 100 linearly decayed LR.

## Model Structure

Baseline generator:

```text
ReflectionPad2d(3)
Conv 7x7
Downsample Conv stride=2 x 2
ResnetBlock x 9
Upsample x 2
ReflectionPad2d(3)
Conv 7x7
Tanh
```

Lightweight generator keeps the same topology but replaces major convolutions with depthwise separable convolutions and reduces residual blocks from 9 to 6.

## Baseline vs Lightweight

| Model | Conv Type | Residual Blocks | Expected Single-G Params |
| --- | --- | ---: | ---: |
| Baseline | Standard Conv | 9 | about 11.78M |
| Lightweight | Depthwise Separable Conv | 6 | about 1.58M |

## Installation

```bash
conda create -n cyclegan_thesis python=3.9 -y
conda activate cyclegan_thesis
pip install -r requirements.txt
```

If `clean-fid`, `lpips`, or `torchmetrics` installation fails on a specific server image, first ensure the matching PyTorch/CUDA wheel is installed, then install the remaining packages individually.

## Dataset Layout

```text
datasets/
  horse2zebra/
    trainA/
    trainB/
    testA/
    testB/
  apple2orange/
    trainA/
    trainB/
    testA/
    testB/
```

`datasets`, `checkpoints`, and `results` are ignored by Git by default.

## Local Debug

```bash
bash scripts/train_debug.sh
```

## AutoDL

```bash
cd /root/autodl-tmp
git clone https://github.com/SunMingYang04/cyclegan-lightweight-thesis.git
cd cyclegan-lightweight-thesis
conda create -n cyclegan_thesis python=3.9 -y
conda activate cyclegan_thesis
pip install -r requirements.txt
bash scripts/train_debug.sh
bash scripts/train_baseline_horse2zebra.sh
bash scripts/train_lightweight_horse2zebra.sh
```

Resume training:

```bash
python train.py --config configs/baseline_horse2zebra.yaml --name baseline_horse2zebra --resume checkpoints/baseline_horse2zebra/latest.pth
```

## Testing

```bash
bash scripts/test_baseline_horse2zebra.sh
bash scripts/test_lightweight_horse2zebra.sh
```

Manual example:

```bash
python test.py --config configs/baseline_horse2zebra.yaml --name baseline_horse2zebra --checkpoint checkpoints/baseline_horse2zebra/latest_G_A.pth --direction AtoB --num_test 50
```

Outputs:

```text
results/baseline_horse2zebra/test_AtoB/input/
results/baseline_horse2zebra/test_AtoB/fake/
```

## Evaluation

```bash
bash scripts/eval_baseline_horse2zebra.sh
bash scripts/eval_lightweight_horse2zebra.sh
```

The evaluator writes both `metrics.json` and `metrics.csv`.

FID/KID compare `fake_dir` against `real_dir`. SSIM/LPIPS are computed only when `--input_dir` is provided. Without `--input_dir`, SSIM and LPIPS are written as `null`.

CMMD is currently a reserved interface returning `null`. Add a project-approved dependency and implementation in `utils/metrics.py` if exact CMMD reporting is required.

## Model Profiling

```bash
python profile_model.py --model_type baseline --device cuda
python profile_model.py --model_type lightweight --device cuda
```

Outputs:

```text
results/profile_baseline.json
results/profile_lightweight.json
```

## Expected Thesis Results

These values are targets for comparison and are not hard-coded.

| Metric | Baseline | Lightweight |
| --- | ---: | ---: |
| Single generator params | about 11.78M | about 1.58M |
| Single image inference time | about 15.31 ms | about 5.66 ms |
| horse2zebra A->B FID | about 72.22 | about 89.26 |
| horse2zebra B->A FID | about 132.64 | about 135.85 |

## Notes

- Do not commit datasets, checkpoints, results, or large model weights.
- For long-term checkpoint storage, use AutoDL storage, GitHub Releases, or Git LFS.
- This project is for thesis reproduction and learning. Exact numeric reproducibility is not guaranteed across GPU models, dependency versions, or random seeds.

## Git Setup

```bash
git init
git add .
git commit -m "init: create CycleGAN lightweight thesis reproduction project"
git branch -M main
git remote add origin https://github.com/SunMingYang04/cyclegan-lightweight-thesis.git
```

Push manually when ready:

```bash
git push -u origin main
```

