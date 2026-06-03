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

## Layout produced by `prepare_data.py`

```
data/processed/<dataset>/{train,val,test}/images/<id>.npy   # HxW or HxWx3
data/processed/<dataset>/{train,val,test}/masks/<id>.npy    # HxW {0,1}
```
Converters require challenge access and are stubbed with per-dataset
instructions; multi-organ CT volumes are exploded into per-organ binary masks.
