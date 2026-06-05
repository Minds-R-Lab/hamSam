#!/usr/bin/env bash
# Launch the Ham-MedSAM comparison matrix and collect a summary table.
# Each cell: train -> per-organ test eval. Override anything via env vars.
#
#   DATA=data/processed/flare22 CKPT=checkpoints/sam_vit_b_01ec64.pth \
#   SEEDS="42 43 44" bash scripts/run_ablations.sh
#
#   DRY_RUN=1 bash scripts/run_ablations.sh     # just print the matrix
#   (re-running RESUMES: cells with results are skipped; FORCE=1 redoes them)
set -euo pipefail

DATA=${DATA:-data/processed/flare22}
CONFIG=${CONFIG:-configs/ham_medsam_abdomen.yaml}
SEEDS=${SEEDS:-42}
ENCODERS=${ENCODERS:-"ham baseline"}
LOSSES=${LOSSES:-"dice+ce dice+ce+hausdorff dice+ce+momentum"}
CKPT=${CKPT:-}                       # optional MedSAM/SAM ViT-B .pth (needs INPUT=1024)
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-8}
INPUT=${INPUT:-1024}
ROOT=${ROOT:-outputs/ablation}
PROMPT_MODES=${PROMPT_MODES:-"box"}  # add "auto" to also score prompt-free
DRY_RUN=${DRY_RUN:-0}

ckpt_arg=""; [ -n "$CKPT" ] && ckpt_arg="--sam_checkpoint $CKPT"
echo "matrix: encoders=[$ENCODERS] losses=[$LOSSES] seeds=[$SEEDS] -> $ROOT  (ckpt='${CKPT:-none}')"

for enc in $ENCODERS; do
  bn=deepest; [ "$enc" = baseline ] && bn=none
  for loss in $LOSSES; do
    # the plain-ConvNeXt baseline emits no momentum -> momentum loss is a no-op there
    if [ "$enc" = baseline ] && echo "$loss" | grep -q momentum; then
      echo "[skip] baseline+momentum (baseline has no momentum signal)"; continue
    fi
    for seed in $SEEDS; do
      out="$ROOT/${enc}_${loss//+/-}_seed${seed}"
      echo "=== enc=$enc bottleneck=$bn loss=$loss seed=$seed -> $out ==="
      [ "$DRY_RUN" = 1 ] && continue
      first_pm=$(echo $PROMPT_MODES | awk '{print $1}')
      if [ -f "$out/eval_${first_pm}/test_per_organ.json" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "[skip] already complete (FORCE=1 to redo): $out"; continue
      fi
      python experiments/train_ham_medsam.py --config "$CONFIG" --data "$DATA" \
        --encoder "$enc" --bottleneck "$bn" --loss "$loss" --seed "$seed" \
        --epochs "$EPOCHS" --batch_size "$BATCH" --input_size "$INPUT" \
        --device "$DEVICE" --output_dir "$out" $ckpt_arg
      for pm in $PROMPT_MODES; do
        python experiments/eval_test.py --checkpoint "$out/best.ckpt" --data "$DATA" \
          --input_size "$INPUT" --device "$DEVICE" --prompt_mode "$pm" \
          --output_dir "$out/eval_$pm"
      done
    done
  done
done

[ "$DRY_RUN" = 1 ] && { echo "(dry run: nothing executed)"; exit 0; }
python experiments/collect_results.py --root "$ROOT" --out "$ROOT/summary" --per_organ
echo "Summary -> $ROOT/summary/summary.md"
