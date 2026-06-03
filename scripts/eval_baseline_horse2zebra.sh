#!/usr/bin/env bash
set -e
python evaluate.py --real_dir datasets/horse2zebra/testB --fake_dir results/baseline_horse2zebra/test_AtoB/fake --input_dir datasets/horse2zebra/testA --output results/baseline_horse2zebra/metrics_AtoB.json

