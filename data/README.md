# Data preparation

Ham-MedSAM uses the **same eight datasets as VM-MedSAM** (Li et al. 2026,
Table 1) so the comparison is apples-to-apples. All are trained as binary,
box-promptable segmentation (the SAM/MedSAM paradigm: one (image, box, binary
mask) sample per structure instance). VM-MedSAM trains jointly across all eight
(`configs/ham_medsam_vmdata.yaml`).

## Datasets (VM-MedSAM Table 1)

| Dataset | Modality | Target | Size (paper) | Source |
|---|---|---|---|---|
| BTCV | Abdominal CT | 13 organs | 5,000 slices | Synapse multi-atlas labeling |
| FLARE22 | Abdominal CT | 13 organs | 5,000 slices | https://flare22.grand-challenge.org |
| MSD Lung | Chest CT | lung cancer | 630 slices | http://medicaldecathlon.com |
| BraTS | MRI (FLAIR) | brain tumor | 4,840 slices | http://braintumorsegmentation.org |
| CVC-ClinicDB | Colonoscopy | polyp | 612 images | polyp.grand-challenge.org/CVCClinicDB |
| BUSI | Ultrasound | breast tumor | 1,312 images | Breast Ultrasound Images dataset |
| DRIVE | Fundus photo | retinal vessel | 20 images | https://drive.grand-challenge.org |
| Montgomery | Chest X-ray | lung | 138 images | Montgomery County X-ray set |

Patient-level splits are used for 3D volumes (CT/MRI) to avoid slice leakage.

## Optional zero-shot probes (NOT in VM-MedSAM)

Used only by `experiments/eval_zero_shot.py` to test transfer to genuinely
unseen domains: **ISIC 2018** (dermoscopy — a modality absent from the eight),
plus **TN3K** (thyroid US) and **ACDC** (cardiac MRI) as unseen-organ probes.

## Preprocessing (`sam_preprocess.py`)

1. Resize longest side to 1024 (keep aspect ratio).
2. Pad to 1024×1024 (zeros).
3. Replicate single channel to 3 for grayscale modalities.
4. Normalise: `none` (default, [0,1]) or `imagenet`. For encoders trained from
   scratch, dataset statistics often beat ImageNet stats.

## Getting the abdomen-CT data (BTCV + FLARE22) -- IMPLEMENTED

`scripts/download_datasets.sh` documents and (for the open one) automates it:

* **FLARE22** -- OPEN on Zenodo (record 7860267): 50 labeled abdomen CT + 13
  organ labels. Downloaded with `zenodo_get`, no account needed.
* **BTCV** -- GATED (Synapse syn3193805): free account + data-use agreement
  required, then `synapseclient` with a personal access token.

Convert the raw NIfTI to the training layout (per-organ binary 2D slices,
abdomen soft-tissue window [-160, 240] HU, patient-level split):
```
python data/prepare_data.py --dataset flare22 \
    --images_dir data/raw/flare22/images --labels_dir data/raw/flare22/labels \
    --out data/processed/flare22
```
13-organ label conventions are built in (`FLARE22_LABELS`, `BTCV_LABELS`);
**verify against your download's docs** and override with `--label_map_json`
if the release uses a different order. All eight converters are implemented: BTCV/FLARE22 (abdomen CT, per-organ),
MSD-Lung (CT lung window), BraTS (MRI-FLAIR whole-tumour), and the four 2D sets
CVC-ClinicDB/BUSI/DRIVE/Montgomery (single-target image pairs). The single-target
ones (CVC, BUSI, MSD-Lung, BraTS) are the fair testbeds for prompt-free inference.

## Layout produced by `prepare_data.py`

```
data/processed/<dataset>/{train,val,test}/images/<pid>_z<zzz>_org<L>_<name>.npy  # HxW [0,1]
data/processed/<dataset>/{train,val,test}/masks/ <pid>_z<zzz>_org<L>_<name>.npy  # HxW {0,1}
```
Multi-organ CT volumes are exploded into per-organ binary masks (one sample =
one organ on one slice), the SAM/MedSAM box-promptable paradigm. Splits are
patient-level to avoid slice leakage.
