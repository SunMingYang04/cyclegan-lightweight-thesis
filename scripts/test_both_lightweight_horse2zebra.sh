#!/usr/bin/env bash
set -e
python test.py --config configs/lightweight_horse2zebra.yaml --name lightweight_horse2zebra --checkpoint checkpoints/lightweight_horse2zebra/latest_G_A.pth --direction both

