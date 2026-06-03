#!/usr/bin/env bash
set -e
python test.py --config configs/baseline_horse2zebra.yaml --name baseline_horse2zebra --checkpoint checkpoints/baseline_horse2zebra/latest_G_A.pth --direction AtoB

