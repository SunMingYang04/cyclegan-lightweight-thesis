# CycleGAN Lightweight Thesis Reproduction

PyTorch project for reproducing the undergraduate thesis **《基于CycleGAN跨域图像转换的研究和实现》**.

Repository:

```bash
https://github.com/SunMingYang04/cyclegan-lightweight-thesis.git
```

## Reproduction Goal

This project reproduces standard CycleGAN and a lightweight CycleGAN generator on unpaired image translation tasks:

- Main dataset: `horse2zebra`
- Supplementary dataset: `apple2orange`
- Metrics: parameters, inference time, FID, KID, SSIM, LPIPS, optional CMMD placeholder

## Baseline vs Lightweight

| Model | Generator | Residual Blocks | Convolution | Discriminator | Losses |
| --- | --- | ---: | --- | --- | --- |
| Baseline | `ResnetGenerator` | 9 | standard convolution | 70x70 PatchGAN | LSGAN + cycle + identity |
| Lightweight | `LightweightGenerator` | 6 | depthwise separable convolution | 70x70 PatchGAN | LSGAN + cycle + identity |
| DWConv9 | `LightweightGenerator` | 9 | depthwise separable convolution | 70x70 PatchGAN | ablation only |

The official thesis reproduction setting keeps `batch_size=1`, `lr=0.0002`, Adam betas `(0.5, 0.999)`, 200 epochs, first 100 epochs fixed LR, last 100 epochs linear decay, image buffer size 50, `lambda_cycle=10`, and `lambda_identity=5`.

`batch_size=2` or `batch_size=4` can be used for quick engineering tests, but formal thesis comparisons should use `batch_size=1`.

AMP is available through config/CLI, but defaults to `use_amp: false` because mixed precision may introduce small numeric differences.

## Hardware Notes

For RTX 4080 SUPER 32GB with large host memory, recommended DataLoader defaults are:

```yaml
num_workers: 8
pin_memory: true
persistent_workers: true
prefetch_factor: 4
cudnn_benchmark: true
cudnn_deterministic: false
use_amp: false
```

Set `cudnn_deterministic: true` only when strict reproducibility is more important than speed.

## Installation

```bash
conda create -n cyclegan_thesis python=3.9 -y
conda activate cyclegan_thesis
pip install -r requirements.txt
```

If `clean-fid` or `lpips` installation fails, install the correct PyTorch/CUDA wheel first, then retry:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install clean-fid lpips
```

## Dataset Layout

```text
datasets/horse2zebra/
  trainA/
  trainB/
  testA/
  testB/
```

`apple2orange` uses the same folder format.

Datasets, checkpoints, results, and large logs are ignored by Git.

## Local Debug

```bash
python profile_model.py --model_type baseline --device cpu --repeat 2 --warmup 1
python profile_model.py --model_type lightweight --device cpu --repeat 2 --warmup 1
python tools/check_dataset.py --dataroot datasets/horse2zebra
bash scripts/train_debug.sh
```

If the dataset has not been placed yet, skip the dataset and training commands.

## CUDA Profile

```bash
python profile_model.py --model_type baseline --device cuda
python profile_model.py --model_type lightweight --device cuda
python profile_model.py --model_type dwconv9 --device cuda
```

Profile results depend on GPU model, driver, CUDA, PyTorch version, and current system load. Depthwise separable convolutions reduce parameters but may not scale linearly in GPU latency.

## Formal Training

```bash
bash scripts/train_baseline_horse2zebra.sh
bash scripts/train_lightweight_horse2zebra.sh
bash scripts/train_dwconv9_horse2zebra.sh
bash scripts/train_no_identity.sh
bash scripts/train_no_cycle.sh
```

Resume:

```bash
python train.py --config configs/baseline_horse2zebra.yaml --name baseline_horse2zebra --resume checkpoints/baseline_horse2zebra/latest.pth
```

Each experiment writes:

```text
results/{name}/config_resolved.yaml
results/{name}/log.txt
results/{name}/losses.csv
results/{name}/samples/
checkpoints/{name}/latest.pth
checkpoints/{name}/latest_G_A.pth
checkpoints/{name}/latest_G_B.pth
```

## Testing

```bash
bash scripts/test_baseline_horse2zebra.sh
bash scripts/test_lightweight_horse2zebra.sh
bash scripts/test_both_lightweight_horse2zebra.sh
```

Manual examples:

```bash
python test.py --config configs/baseline_horse2zebra.yaml --name baseline_horse2zebra --checkpoint checkpoints/baseline_horse2zebra/latest_G_A.pth --direction AtoB
python test.py --config configs/lightweight_horse2zebra.yaml --name lightweight_horse2zebra --checkpoint checkpoints/lightweight_horse2zebra/latest_G_A.pth --direction both --num_test 50
```

Outputs:

```text
results/{name}/test_AtoB/input/
results/{name}/test_AtoB/fake/
results/{name}/test_BtoA/input/
results/{name}/test_BtoA/fake/
```

## Evaluation

```bash
bash scripts/eval_baseline_horse2zebra.sh
bash scripts/eval_lightweight_horse2zebra.sh
```

Manual example:

```bash
python evaluate.py \
  --real_dir datasets/horse2zebra/testB \
  --fake_dir results/lightweight_horse2zebra/test_AtoB/fake \
  --input_dir results/lightweight_horse2zebra/test_AtoB/input \
  --output results/lightweight_horse2zebra/metrics_AtoB.json \
  --max_images 500
```

FID/KID compare `fake_dir` with `real_dir`. SSIM/LPIPS require `--input_dir`; if it is omitted, they are saved as `null`. FID/KID values may vary with dependency versions, preprocessing, and sample count.

CMMD is currently a reserved interface returning `null`; add an approved implementation in `utils/metrics.py` if required.

## AutoDL Workflow

```bash
cd /root/autodl-tmp
git clone https://github.com/SunMingYang04/cyclegan-lightweight-thesis.git
cd cyclegan-lightweight-thesis
conda create -n cyclegan_thesis python=3.9 -y
conda activate cyclegan_thesis
pip install -r requirements.txt
chmod +x scripts/*.sh
bash scripts/train_debug.sh
```

Then run formal experiments:

```bash
bash scripts/train_baseline_horse2zebra.sh
bash scripts/train_lightweight_horse2zebra.sh
```

## Common Errors

- `CUDA unavailable`: check driver, PyTorch CUDA wheel, and `nvidia-smi`.
- `dataset path not found`: place data under `datasets/{dataset_name}/trainA`, `trainB`, `testA`, `testB`.
- `checkpoint not found`: run training first or check `checkpoints/{name}/latest_G_A.pth`.
- Out of memory: keep formal `batch_size=1`; reduce `num_workers` if host memory pressure appears.
- `clean-fid` install failed: install PyTorch first, then `pip install clean-fid`.
- `lpips` install failed: install PyTorch/torchvision first, then `pip install lpips`.
- GitHub push permission issue: ensure the remote belongs to your account and authenticate with token or SSH.
- AutoDL data path issue: copy or symlink dataset into `/root/autodl-tmp/cyclegan-lightweight-thesis/datasets/`.

## Git Notes

Do not upload datasets, checkpoints, generated images, large logs, or weight files. For long-term checkpoint storage, use AutoDL storage, GitHub Releases, or Git LFS.

```bash
git add .
git commit -m "refactor: improve CycleGAN project structure and GPU efficiency"
git push -u origin main
```

Do not push until local validation is acceptable.
