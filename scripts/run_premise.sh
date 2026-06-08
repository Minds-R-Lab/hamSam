#!/usr/bin/env bash
# ==========================================================================
# DECISIVE PREMISE TEST
# Does the Hamiltonian *mechanism* beat a capacity-matched ConvNeXt control?
#   - variant-A is matched to ~99% of Ham's parameter count, so any gap is
#     attributable to mechanism, not capacity.
#   - box-prompted (clean spine, no prompt-free confound)
#   - trained on ISIC2017, evaluated IN-DIST (ISIC2017) and OOD (ISIC2018)
#   - 3 seeds each -> mean +/- std (single-seed gaps < ~0.5pp are noise)
# Resumable: re-running skips any (train/eval) whose output already exists.
# Override anything via env, e.g.:  SEEDS="42 43" EPOCHS=30 bash scripts/run_premise.sh
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DATA17=${DATA17:-$HOME/HamVision_V2/data/ISIC2017}
DATA18=${DATA18:-$HOME/HamVision_V2/data/ISIC2018}
CONFIG=${CONFIG:-configs/ham_medsam_vmdata.yaml}
EPOCHS=${EPOCHS:-40}
SEEDS=${SEEDS:-"42 43 44"}
OUT=${OUT:-runs/premise}

echo "ISIC2017=$DATA17"
echo "ISIC2018=$DATA18  (OOD)"
echo "seeds=$SEEDS  epochs=$EPOCHS  out=$OUT"

run () {                       # $1=name  $2=ablation
  local name=$1 abl=$2
  for s in $SEEDS; do
    local dir="$OUT/${name}_s${s}"
    if [ -f "$dir/best.ckpt" ]; then
      echo "[skip] train $dir"
    else
      echo "[train] $name seed=$s (ablation=$abl)"
      python experiments/train_ham_medsam.py --config "$CONFIG" --data "$DATA17" \
        --encoder ham --bottleneck deepest --ablation "$abl" --loss dice+ce \
        --epochs "$EPOCHS" --early_stop_patience 15 --seed "$s" --output_dir "$dir"
    fi
    if [ -f "$dir/indist/test_per_organ.json" ]; then echo "[skip] eval in-dist $dir"; else
      python experiments/eval_test.py --checkpoint "$dir/best.ckpt" --data "$DATA17" \
        --prompt_mode box --output_dir "$dir/indist"
    fi
    if [ -f "$dir/ood/test_per_organ.json" ]; then echo "[skip] eval OOD $dir"; else
      python experiments/eval_test.py --checkpoint "$dir/best.ckpt" --data "$DATA18" \
        --prompt_mode box --output_dir "$dir/ood"
    fi
  done
}

run ham      none
run variantA A

echo; echo "================  RESULTS  ================"
python experiments/collect_premise.py --root "$OUT" --seeds "$SEEDS"
