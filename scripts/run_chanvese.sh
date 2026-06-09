#!/usr/bin/env bash
# ==========================================================================
# DEEP CHAN-VESE (physics-native, prompt-free) vs capacity-matched control.
#   Same U-Net + learned data term for both; the only difference is whether the
#   phase-field gradient flow is in the loop. Trained prompt-free on ISIC2017,
#   evaluated on the clean ISIC2017 test split. 3 seeds.
# Resumable. Override via env: SEEDS="42" EPOCHS=40 bash scripts/run_chanvese.sh
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DATA17=${DATA17:-$HOME/HamVision_V2/data/ISIC2017}
EPOCHS=${EPOCHS:-60}
STEPS=${STEPS:-15}
INPUT=${INPUT:-256}
SEEDS=${SEEDS:-"42 43 44"}
OUT=${OUT:-runs/cv}

for s in $SEEDS; do
  for tag in phys ctrl; do
    dir="$OUT/${tag}_s${s}"
    flag=""; [ "$tag" = "ctrl" ] && flag="--no_physics"
    if [ -f "$dir/test_per_organ.json" ]; then echo "[skip] $dir"; continue; fi
    echo "[train] $tag seed=$s"
    python experiments/train_chanvese.py --data "$DATA17" --output_dir "$dir" \
      --epochs "$EPOCHS" --steps "$STEPS" --input_size "$INPUT" --seed "$s" $flag
  done
done

echo; echo "==========  DEEP CHAN-VESE: physics vs control  =========="
python experiments/collect_chanvese.py --root "$OUT" --seeds "$SEEDS"
