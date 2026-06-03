# Summary of Li et al. (2026), *VM-MedSAM*

**Citation.** Zimao Li, Hongyan Zhao, Fan Yin, Lu Zheng, Jun Tie, Peng
Zhao, Shengzhou Xu. *Vision mamba augmented segment anything model for
medical image segmentation.* Medical Physics, 53:e70492, 2026.
DOI 10.1002/mp.70492.

## What they did

VM-MedSAM is a SAM-variant for medical image segmentation. The
architecture follows MedSAM (image encoder + prompt encoder + mask
decoder) but with two changes:

1. **Image encoder swap.** The heavy ViT image encoder of MedSAM is
   replaced by a ResVision Mamba+ (RVM+) Mamba-based encoder
   (introduced in the same group's prior work). The encoder is trained
   from scratch; no MedSAM pretrained weights are used.
2. **Boundary-aware loss.** A Hausdorff-distance term is added to the
   training loss to compensate for blurred boundaries in compressed
   medical images.

The prompt encoder is **frozen** at MedSAM's weights. The mask decoder
is the standard SAM cross-attention.

### Resolution path

```
input image      3 x 1024 x 1024
   |  7x7 conv stride 2
   v
shallow feat    32 x 512 x 512
   |  RVM+ stage 1
   v
                64 x 256 x 256
   |  RVM+ stage 2
   v
                128 x 128 x 128
   |  RVM+ stage 3
   v
image embedding 256 x 64 x 64       --> SAM mask decoder
```

The image embedding shape (256 x 64 x 64) is the same as MedSAM's, so
the prompt encoder + mask decoder can be reused without modification.

### Training / evaluation data

CORRECTION (June 2026 code review): the published paper (Table 1) evaluates
on EIGHT public datasets across FIVE modalities, not on CT/MRI alone:

* BTCV (abdominal CT, 13 organs, 5,000 slices)
* FLARE22 (abdominal CT, 13 organs, 5,000 slices)
* MSD Lung (chest CT, lung cancer, 630 slices)
* BraTS (brain MRI FLAIR, 4,840 slices)
* CVC-ClinicDB (colonoscopy, polyp, 612 images)
* BUSI (ultrasound, breast tumour, 1,312 images)
* DRIVE (fundus photography, retinal vessel, 20 images)
* Montgomery (chest X-ray, lung, 138 images)

Patient-level splits are used for the 3D volumes to avoid slice leakage.
Training is JOINT across all datasets (not per-domain).

### Reported gains over MedSAM

* parameters: −65.11%
* training speed: ×3.82
* model size: −85.41%
* abdominal organs: slight Dice improvement (0–1 pp depending on organ)
* lung cancer / brain tumour: large Dice improvements (several pp)

## What they did *not* do

1. **Eliminate the box prompt.** Like every SAM variant, VM-MedSAM
   still requires a user-supplied bounding box at inference. This is
   the central usability barrier for clinical batch processing.
2. **Supervise the encoder directly.** Their Hausdorff loss acts on
   the predicted mask, not on any intermediate encoder representation.
   The encoder learns boundary information only indirectly via the
   long path through the prompt encoder + mask decoder.
3. **Expose interpretable intermediate signals.** RVM+ emits one
   generic feature tensor. There is no built-in saliency, no built-in
   spatial-derivative signal, no in-network attribution.
4. **Demonstrate zero-shot cross-modality transfer.** CORRECTION: VM-MedSAM
   *does* evaluate across five modalities (CT, MRI, ultrasound, X-ray,
   colonoscopy/fundus; Table 1), so the earlier "CT/MRI only" framing was
   wrong. What it does NOT do is *zero-shot* transfer: every reported number
   comes from a model JOINTLY trained on that domain. Train-on-CT /
   test-on-unseen-modality (no fine-tuning) is the gap our Extension 6
   actually fills. OCT, histology and blood-cell microscopy are still
   untested.
5. **Statistical significance reporting.** Single-seed numbers are
   reported; no cross-seed standard deviations.

## Why HamVision is a natural extension

| VM-MedSAM weakness | HamVision capability that helps |
|---|---|
| Prompt dependency | energy map H is a free saliency signal |
| Hausdorff loss is post-hoc | momentum p is a band-pass differentiator at the encoder |
| Single feature stream | bottleneck produces (q, p, H) by design |
| No zero-shot transfer (joint training) | HamVision 3-seed pipeline + cross-modality protocol already implemented |
| Single-seed numbers | 3-seed pipeline already implemented in HamVision |

The detailed extension plan is in `../PLAN.md`.

## Related work mentioned in the paper

The VM-MedSAM paper cites:

* **SAM** (Kirillov et al. 2023) -- original segment anything model.
* **MedSAM** (Ma et al. 2024) -- medical fine-tune of SAM.
* **Med-SA** (Wu et al.) -- adapter-based 2D-to-3D adaptation.
* **3DSAM-Adapter** (Gong et al.) -- adapter variant.
* **Mamba** (Gu & Dao 2023) -- selective state-space sequence model.
* **VMamba** (Liu et al. 2024) -- 2D vision Mamba.
* **Vim** -- bidirectional Mamba blocks.
* **RVM / RVM+** (Liao et al.) -- residual Mamba with lightweight scan.

The Hamiltonian-as-architecture line (Greydanus 2019, Mamba-3, etc.) is
*not* discussed -- which is the niche our extension fills.
