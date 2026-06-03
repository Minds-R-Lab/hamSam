#!/usr/bin/env bash
# One-time environment setup for the H100. Run from the repo root.
set -euo pipefail

echo "[1/3] Python deps"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[2/3] segment-anything (real SAM prompt encoder + mask decoder)"
# Official package (needs a matching torch/torchvision). Either works:
pip install git+https://github.com/facebookresearch/segment-anything.git \
  || pip install segment-anything-py
# Ham-MedSAM only uses segment_anything.modeling; if torchvision is broken the
# code stubs the unused import automatically (src/sam_utils.ensure_sam_importable).

echo "[3/3] sanity import"
python - <<'PY'
import torch
from src.sam_utils import ensure_sam_importable
ensure_sam_importable()
from segment_anything.modeling import PromptEncoder, MaskDecoder  # noqa
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("segment_anything.modeling import OK")
PY
echo "Done. Next: bash scripts/download_checkpoints.sh  (optional, for pretrained MedSAM/SAM)"
echo "Then:  bash scripts/run_smoke.sh"
