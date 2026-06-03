#!/usr/bin/env bash
# Download a ViT-B checkpoint for the SAM prompt encoder + mask decoder.
# Ham-MedSAM works WITHOUT this (random-init SAM from scratch), but pretrained
# MedSAM/SAM weights are needed for meaningful accuracy.
set -euo pipefail
mkdir -p checkpoints

# Official SAM ViT-B (loads via sam_model_registry["vit_b"]).
if [ ! -f checkpoints/sam_vit_b_01ec64.pth ]; then
  echo "Downloading SAM ViT-B ..."
  wget -O checkpoints/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

# MedSAM ViT-B (same architecture, medical fine-tune). Distributed via the
# MedSAM repo (bowang-lab/MedSAM). Place medsam_vit_b.pth in checkpoints/ and
# point the config at it; it loads with the same vit_b registry.
echo "For MedSAM weights: see https://github.com/bowang-lab/MedSAM (medsam_vit_b.pth)"
echo "Set model.sam_checkpoint in the YAML to the .pth you want (input_size must be 1024)."
