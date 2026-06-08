#!/usr/bin/env bash
# ==========================================================================
# DATA-EFFICIENCY SWEEP  (does the Hamiltonian prior help most when labels are scarce?)
#   Ham vs capacity-matched variant-A, box-prompted, trained on a deterministic
#   fraction of ISIC2017 labels, evaluated on the clean ISIC2017 test split.
#   Same subset per (frac, seed) for both encoders. 3 seeds -> mean +/- std.
#   frac=1.0 reuses the premise full-data results if present (no retrain).
# Resumable. Override via env: FRACS="0.1 0.5 1.0" SEEDS="42" bash scripts/run_dataeff.sh
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DATA17=${DATA17:-$HOME/HamVision_V2/data/ISIC2017}
CONFIG=${CONFIG:-configs/ham_medsam_vmdata.yaml}
EPOCHS=${EPOCHS:-40}
FRACS=${FRACS:-"0.05 0.1 0.25 0.5 1.0"}
SEEDS=${SEEDS:-"42 43 44"}
OUT=${OUT:-runs/dataeff}
PREMISE=${PREMISE:-runs/premise}

run () {                       # $1=name  $2=ablation
  local name=$1 abl=$2
  for f in $FRACS; do
    for s in $SEEDS; do
      local dir="$OUT/${name}_f${f}_s${s}"
      # reuse full-data result from the premise run if available
      if [ "$f" = "1.0" ] && [ -f "$PREMISE/${name}_s${s}/indist/test_per_organ.json" ] \
         && [ ! -f "$dir/test_per_organ.json" ]; then
        mkdir -p "$dir"
        cp "$PREMISE/${name}_s${s}/indist/test_per_organ.json" "$dir/test_per_organ.json"
        echo "[reuse] $dir <- premise full-data"; continue
      fi
      if [ -f "$dir/best.ckpt" ]; then echo "[skip] train $dir"; else
        echo "[train] $name frac=$f seed=$s"
        python experiments/train_ham_medsam.py --config "$CONFIG" --data "$DATA17" \
          --encoder ham --bottleneck deepest --ablation "$abl" --loss dice+ce \
          --epochs "$EPOCHS" --early_stop_patience 15 --seed "$s" \
          --train_frac "$f" --output_dir "$dir"
      fi
      if [ -f "$dir/test_per_organ.json" ]; then echo "[skip] eval $dir"; else
        python experiments/eval_test.py --checkpoint "$dir/best.ckpt" --data "$DATA17" \
          --prompt_mode box --output_dir "$dir"
      fi
    done
  done
}

run ham      none
run variantA A

echo; echo "================  DATA-EFFICIENCY CURVE  ================"
python experiments/collect_dataeff.py --root "$OUT" --fracs "$FRACS" --seeds "$SEEDS"
