# Data preparation

This folder will hold the dataset download / preprocessing scripts. The
extension reuses HamVision's `prepare_data.py` pipeline as a starting
point; the only new datasets are abdomen CT (Li et al.'s benchmark)
and a SAM-style 1024 x 1024 resize step.

## Datasets

| Dataset | Modality | Used for | Public? | Notes |
|---|---|---|---|---|
| FLARE22 | Abdomen CT, 12 organs | Phase 0-3 | yes | https://flare22.grand-challenge.org |
| BTCV | Abdomen CT, 13 organs | Phase 0-3 (alt) | yes | Synapse multi-organ benchmark |
| LIDC-IDRI | Lung CT | Phase 0 reproduction | yes | substitute for Li et al.'s lung-cancer set |
| BraTS 2021 | Brain MRI | Phase 0 reproduction | yes | substitute for Li et al.'s brain-tumour set |
| ISIC 2018 | Dermoscopy | Phase 5 zero-shot | yes | already in HamVision suite |
| TN3K | Thyroid US | Phase 5 zero-shot | yes | already in HamVision suite |
| ACDC | Cardiac MRI | Phase 5 zero-shot | yes | already in HamVision suite |

## Preprocessing

SAM/MedSAM operate on 1024 x 1024 RGB images. The preprocessing
pipeline is:

1. Resize the longer side to 1024 (keep aspect ratio).
2. Pad the shorter side to 1024 with zeros.
3. Replicate the single channel to 3 channels for grayscale modalities.
4. Normalise to ImageNet statistics (mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]) **only** if the SAM mask decoder
   expects it; for medical encoders trained from scratch, dataset
   statistics often work better.

The HamVision `prepare_data.py` already handles steps 1-3; the
SAM-specific normalisation is added in `data/sam_preprocess.py` (to be
written).
