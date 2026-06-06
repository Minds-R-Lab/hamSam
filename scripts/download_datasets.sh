#!/usr/bin/env bash
# Download abdomen-CT datasets and convert to the training layout.
# FLARE22 (labeled) is OPEN on Zenodo and is fully automated below.
# BTCV is GATED (Synapse account + data-use agreement) -> manual auth.
set -euo pipefail
mkdir -p data/raw data/processed
pip install -q nibabel zenodo_get >/dev/null 2>&1 || pip install nibabel zenodo_get

echo "==================== FLARE22 (open, Zenodo 7860267) ===================="
if [ -z "$(find data/raw/flare22 -name 'FLARE22_Tr_*_0000.nii.gz' 2>/dev/null | head -1)" ]; then
  mkdir -p data/raw/flare22
  ( cd data/raw/flare22 && zenodo_get 7860267 )
  find data/raw/flare22 -name '*.zip' -exec unzip -n -d data/raw/flare22 {} \;
fi
echo "Converting FLARE22 (auto-detects the images/labels dirs) ..."
python data/prepare_data.py --dataset flare22 --root data/raw/flare22 \
    --out data/processed/flare22
echo "FLARE22 ready at data/processed/flare22"

echo
echo "==================== BTCV (gated, Synapse syn3193805) =================="
echo "BTCV needs a free Synapse account + accepted data-use agreement:"
echo "  1) Accept the DUA at https://www.synapse.org/Synapse:syn3193805"
echo "  2) Create a personal access token, then:"
cat <<'CMD'
  pip install synapseclient
  export SYNAPSE_AUTH_TOKEN=<your_token>
  synapse get -r syn3193805 --downloadLocation data/raw/btcv
  # then (autodetects RawData/Training/{img,label}):
  python data/prepare_data.py --dataset btcv --root data/raw/btcv \
      --out data/processed/btcv
CMD
echo
echo "==================== MSD-Lung (Medical Decathlon Task06) ==============="
echo "Open via http://medicaldecathlon.com (Task06_Lung.tar; also commonly on the"
echo "MONAI AWS mirror). After extracting to data/raw/msd_lung/Task06_Lung:"
cat <<'CMD'
  python data/prepare_data.py --dataset msd_lung \
      --images_dir data/raw/msd_lung/Task06_Lung/imagesTr \
      --labels_dir data/raw/msd_lung/Task06_Lung/labelsTr \
      --out data/processed/msd_lung
CMD

echo
echo "==================== BraTS (use Medical Decathlon Task01) =============="
echo "VM-MedSAM uses brain-tumour MRI (FLAIR). The simplest open source is MSD"
echo "Task01_BrainTumour (4-D image; we take FLAIR=modality 0, whole tumour)."
echo "Extract to data/raw/brats/Task01_BrainTumour, then:"
cat <<'CMD'
  python data/prepare_data.py --dataset brats \
      --images_dir data/raw/brats/Task01_BrainTumour/imagesTr \
      --labels_dir data/raw/brats/Task01_BrainTumour/labelsTr \
      --out data/processed/brats
CMD

echo
echo "============ Kvasir-SEG (polyp) -- EASIEST single-target, one zip ======"
if [ ! -d data/raw/kvasir-seg/Kvasir-SEG ]; then
  mkdir -p data/raw/kvasir-seg
  wget -O data/raw/kvasir-seg/kvasir-seg.zip https://datasets.simula.no/downloads/kvasir-seg.zip
  unzip -n -d data/raw/kvasir-seg data/raw/kvasir-seg/kvasir-seg.zip
fi
python data/prepare_data.py --dataset kvasir_seg \
    --images_dir data/raw/kvasir-seg/Kvasir-SEG/images \
    --labels_dir data/raw/kvasir-seg/Kvasir-SEG \
    --out data/processed/kvasir_seg
echo "Kvasir-SEG ready at data/processed/kvasir_seg (single-target polyp)."

echo
echo "============ 2-D single-target sets (ideal for prompt-free) ==========="
echo "Download each, arrange as <root>/images + mask dir(s), then convert:"
cat <<'CMD'
  # CVC-ClinicDB (polyp): images/ + masks/ (same stem)
  python data/prepare_data.py --dataset cvc_clinicdb --images_dir data/raw/cvc/images --labels_dir data/raw/cvc --out data/processed/cvc_clinicdb
  # BUSI (breast tumour): <x>.png + <x>_mask.png
  python data/prepare_data.py --dataset busi --images_dir data/raw/busi/images --labels_dir data/raw/busi --out data/processed/busi
  # DRIVE (retinal vessels): id-paired, .gif masks
  python data/prepare_data.py --dataset drive --images_dir data/raw/drive/images --labels_dir data/raw/drive --out data/processed/drive
  # Montgomery (lung): leftMask/ + rightMask/ OR-merged
  python data/prepare_data.py --dataset montgomery --images_dir data/raw/montgomery/CXR_png --labels_dir data/raw/montgomery/ManualMask --out data/processed/montgomery
CMD
echo "(CVC-ClinicDB, BUSI, BraTS, MSD-Lung are single-target -> the fair prompt-free testbeds.)"

echo
echo "Train on FLARE22 alone now:"
echo "  python experiments/train_ham_medsam.py --config configs/ham_medsam_abdomen.yaml \\"
echo "      --encoder ham --loss dice+ce+momentum --seed 42 --output_dir outputs/ham/seed_42 --device cuda"
echo "(Use configs/ham_medsam_vmdata.yaml only once ALL eight datasets are prepared.)"
echo
echo "NOTE: verify each release's label order vs its docs; override --label_map_json if needed."
