#!/usr/bin/env bash
# Validate every component before launching real training.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

echo "== unit tests =="
python -m pytest tests/ -q

echo "== real-SAM component smoke (input 256) =="
python experiments/smoke_real.py --input_size 256

echo "== end-to-end: 1-epoch synthetic train -> prompt-free eval =="
python experiments/train_ham_medsam.py --config configs/ham_medsam_abdomen.yaml \
  --data synthetic --epochs 1 --batch_size 2 --input_size 256 \
  --loss dice+ce+momentum --output_dir /tmp/hamsmoke
python experiments/eval_prompt_free.py --checkpoint /tmp/hamsmoke/best.ckpt \
  --data synthetic --input_size 256 --output_dir /tmp/hamsmoke_pf

echo "ALL SMOKE CHECKS PASSED. Ready for real training (see README 'Real training on H100')."
