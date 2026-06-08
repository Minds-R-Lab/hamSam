#!/usr/bin/env bash
# ==========================================================================
# REAL OOD PROBE: evaluate the already-trained premise checkpoints (ISIC2017)
# on PH2 -- a different-source dermoscopy set with NO image overlap. No
# retraining; just box-prompted inference + aggregate vs the matched control.
# Prereq: scripts/run_premise.sh has produced runs/premise/{ham,variantA}_s*/best.ckpt
#         and PH2 has been converted (data/convert_ph2.py).
# Usage:  PH2=data/processed/ph2 bash scripts/run_ph2_ood.sh
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PH2=${PH2:-data/processed/ph2}
SEEDS=${SEEDS:-"42 43 44"}
OUT=${OUT:-runs/premise}

[ -d "$PH2/test/images" ] || { echo "PH2 not found at $PH2/test/images -- run data/convert_ph2.py first"; exit 1; }

for name in ham variantA; do
  for s in $SEEDS; do
    dir="$OUT/${name}_s${s}"
    [ -f "$dir/best.ckpt" ] || { echo "[miss] $dir/best.ckpt (run run_premise.sh first)"; continue; }
    if [ -f "$dir/ph2/test_per_organ.json" ]; then echo "[skip] $dir/ph2"; else
      python experiments/eval_test.py --checkpoint "$dir/best.ckpt" --data "$PH2" \
        --prompt_mode box --output_dir "$dir/ph2"
    fi
  done
done

echo; echo "==========  Ham vs control: in-dist (ISIC2017) vs REAL OOD (PH2)  =========="
python experiments/collect_premise.py --root "$OUT" --seeds "$SEEDS" --ood_sub ph2
