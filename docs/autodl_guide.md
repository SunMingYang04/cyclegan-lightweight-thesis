# AutoDL Guide

```bash
cd /root/autodl-tmp
git clone https://github.com/SunMingYang04/cyclegan-lightweight-thesis.git
cd cyclegan-lightweight-thesis
conda create -n cyclegan_thesis python=3.9 -y
conda activate cyclegan_thesis
pip install -r requirements.txt
```

Place datasets under `datasets/horse2zebra` and `datasets/apple2orange`.

Debug run:

```bash
bash scripts/train_debug.sh
```

Full runs:

```bash
bash scripts/train_baseline_horse2zebra.sh
bash scripts/train_lightweight_horse2zebra.sh
```

Resume:

```bash
python train.py --config configs/baseline_horse2zebra.yaml --name baseline_horse2zebra --resume checkpoints/baseline_horse2zebra/latest.pth
```

