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
echo "Train on FLARE22 alone now:"
echo "  python experiments/train_ham_medsam.py --config configs/ham_medsam_abdomen.yaml \\"
echo "      --encoder ham --loss dice+ce+momentum --seed 42 --output_dir outputs/ham/seed_42 --device cuda"
echo "(Use configs/ham_medsam_vmdata.yaml only once ALL eight datasets are prepared.)"
echo
echo "NOTE: verify each release's label order vs its docs; override --label_map_json if needed."
