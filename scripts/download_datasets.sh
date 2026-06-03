#!/usr/bin/env bash
# Download the abdomen-CT datasets, then convert to the training layout.
# FLARE22 (labeled) is OPEN on Zenodo; BTCV is GATED (Synapse account + DUA).
set -euo pipefail
mkdir -p data/raw data/processed

echo "==================== FLARE22 (open, Zenodo 7860267) ===================="
# 50 labeled abdomen CT scans + 13-organ labels. ~Several GB.
pip install zenodo_get >/dev/null 2>&1 || true
if [ ! -d data/raw/flare22 ]; then
  mkdir -p data/raw/flare22
  ( cd data/raw/flare22 && zenodo_get 7860267 )   # downloads all record files
  # Unzip any archives the record ships (image/label folders inside).
  find data/raw/flare22 -name '*.zip' -exec unzip -n -d data/raw/flare22 {} \;
fi
echo "After download, locate the image and label dirs (names vary by release),"
echo "then convert (13-organ -> per-organ binary 2D slices, patient-level split):"
cat <<'CMD'
  python data/prepare_data.py --dataset flare22 \
      --images_dir data/raw/flare22/images \
      --labels_dir data/raw/flare22/labels \
      --out        data/processed/flare22
CMD

echo
echo "==================== BTCV (gated, Synapse syn3193805) =================="
echo "1) Create a free Synapse account and ACCEPT the data-use agreement at:"
echo "     https://www.synapse.org/Synapse:syn3193805"
echo "2) Create a personal access token, then:"
cat <<'CMD'
  pip install synapseclient
  export SYNAPSE_AUTH_TOKEN=<your_token>
  # Download the Abdomen RawData (images + labels) into data/raw/btcv:
  synapse get -r syn3193805 --downloadLocation data/raw/btcv
  # (or download Abdomen/RawData.zip from the website and unzip it there)

  python data/prepare_data.py --dataset btcv \
      --images_dir data/raw/btcv/RawData/Training/img \
      --labels_dir data/raw/btcv/RawData/Training/label \
      --out        data/processed/btcv
CMD
echo
echo "NOTE: verify each release's label order against its docs; override with"
echo "      --label_map_json if it differs from the built-in convention."
