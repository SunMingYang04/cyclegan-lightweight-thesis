#!/usr/bin/env bash
set -e
python evaluate.py --real_dir datasets/horse2zebra/testB --fake_dir results/lightweight_horse2zebra/test_AtoB/fake --input_dir results/lightweight_horse2zebra/test_AtoB/input --output results/lightweight_horse2zebra/metrics_AtoB.json
