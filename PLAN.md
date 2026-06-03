# Ham-MedSAM extension plan

A research plan for extending Li et al. (Med. Phys. 2026), *Vision mamba
augmented segment anything model for medical image segmentation*
(VM-MedSAM, DOI 10.1002/mp.70492), with the Hamiltonian-bottleneck ideas
from HamVision (HamSeg + HamCls).

Status: planning document. Code stubs are in `src/`. The first execution
milestone is reproducing the VM-MedSAM numbers as a baseline; everything
else builds on that baseline.

---

## 0. Code-review corrections (June 2026)

This plan and the copied primitives were reviewed against the two source PDFs
(now in `two_papers/`, git-ignored). Substantive findings, all addressed in
the code:

1. **Scan-direction bug (fixed).** `HamiltonianSS2D._to_lines` reversed the
   *channel* axis instead of the *spatial* axis for the d=1/d=3 sweeps, so the
   "four-direction parallel scan (left, right, up, down)" of the HamVision
   paper was realised as only two genuine directions. Fixed in
   `src/hamiltonian.py` (flip dim 2); guarded by `tests/test_hamiltonian.py`.

2. **Resolution-aware clamps (fixed).** The +/-50 value clamp and [-5,0] decay
   clamp were tuned for a 28x28 bottleneck (L=28); the scan magnitude grows
   ~sqrt(L) and reaches ~44 at L=512. `value_clamp` is now per-stage and the
   encoder widens it for coarser placements. Relevant only for
   `bottleneck='all'`.

3. **Momentum-loss target (resolved).** The plan's prose ("high momentum where
   the boundary is") contradicts its own formula (the raw distance transform
   is ZERO on the boundary). `MomentumBoundaryLoss` implements the stated
   *intent* via a boundary-proximity map by default; `target='distance'`
   keeps the literal formula. Report which target was used.

4. **VM-MedSAM modality claim (corrected).** VM-MedSAM is NOT "CT/MRI only":
   its Table 1 covers eight datasets across five modalities (CT, MRI,
   ultrasound, X-ray, colonoscopy/fundus), jointly trained. Extension 6's real
   novelty is *zero-shot* transfer (train on one domain, test on an unseen one
   with no fine-tuning), not "evaluating more modalities". Sections 1 and 3
   below are corrected accordingly.

5. **Two HamSeg number sets.** HamVision reports a plain 3-seed table
   (ISIC2017 88.37 / ISIC2018 89.29 / TN3K 85.95 / ACDC 93.81 / MMOTU 87.10)
   and a higher TTA+ensemble table (Tables 2-3: 89.28 / 90.32 / 87.39 / ... /
   88.17). The zero-shot reference constants use the Tables 2-3 (deployment)
   numbers; state this when comparing.

6. **Momentum loss redesigned (recommendation adopted).** Rather than match
   raw |p| to a fixed template (which fights the paper's measured
   interior>boundary>exterior structure of |p|), the default
   `MomentumBoundaryLoss` mode is `projection`: a learned 1x1 conv reads a
   boundary logit out of p, supervised (BCE + soft-Dice) against a soft
   boundary band. Still encoder-level (gradients flow into p); the three
   template variants remain available for ablation. The loss head's params
   must be added to the optimiser (the training script does this).

7. **Datasets switched to VM-MedSAM's exact eight** (Table 1: BTCV, FLARE22,
   MSD Lung, BraTS, CVC-ClinicDB, BUSI, DRIVE, Montgomery), trained jointly,
   binary box-promptable. ISIC/TN3K/ACDC are kept only as optional unseen-domain
   zero-shot probes. See `configs/ham_medsam_vmdata.yaml` and `data/README.md`.

8. **SAM backend = MedSAM ViT-B (default).** SAM 3/3.1 (Nov 2025/Mar 2026) is
   open-vocabulary text/concept segmentation -- a different paradigm; SAM2/
   MedSAM2 need multi-scale FPN features in the decoder. Both are documented
   extension points (`src/sam_utils.py`) that raise on use; ViT-B keeps the
   VM-MedSAM comparison clean.

---

## 1. What Li et al. (2026) did, and where it stops short

VM-MedSAM keeps the MedSAM architecture (image encoder + prompt encoder +
mask decoder) but swaps the heavy ViT image encoder for a ResVision
Mamba+ (RVM+) Mamba-based encoder. Reported gains over MedSAM:

* parameters −65.11%,
* training speed ×3.82,
* model size −85.41%,
* small Dice gains on 12 abdominal organs,
* larger Dice gains on lung cancer and brain tumour.

Their architecture nonetheless has four characteristic limitations that
HamVision's Hamiltonian bottleneck addresses head-on:

1. **Prompt dependency.** Like every SAM variant, VM-MedSAM still
   requires a user-supplied bounding box at inference time. This is the
   single largest barrier to clinical batch deployment.
2. **Boundary loss is post-hoc.** They add a Hausdorff-distance term on
   the output mask to compensate for blurred boundaries. Hausdorff is a
   *property of the predicted mask*; it never reaches the encoder.
3. **Single feature stream.** The RVM+ encoder emits a generic feature
   tensor. There is no built-in saliency, no built-in spatial-derivative
   signal, and no in-network interpretability.
4. **No zero-shot cross-modality transfer.** (Corrected from "narrow
   modality evaluation".) VM-MedSAM is jointly trained and evaluated on eight
   datasets spanning five modalities (CT, MRI, ultrasound, X-ray,
   colonoscopy/fundus). What is missing is *zero-shot* transfer to an unseen
   modality without fine-tuning -- the robustness setting our Extension 6
   targets.

## 2. What HamVision brings

The Hamiltonian bottleneck in HamVision produces three structured outputs
from a single damped-oscillator parallel scan:

| Output | Definition | What it gives Ham-MedSAM |
|---|---|---|
| **q** (position) | filtered feature representation | drop-in replacement for RVM+ feature stream |
| **p** (momentum) | band-pass spatial differentiator (Appendix A of HamVision) | encoder-level boundary signal that the Hausdorff loss tries to inject post-hoc |
| **H** (energy) | non-negative per-pixel saliency, `H = ½(|q|² + |p|²)` | free saliency map that can drive a prompt-free pipeline and replace Grad-CAM |

In addition, HamVision provides:

* BIBO-stable scan (`|exp(−νΔ)| < 1` by construction), so no LayerNorm
  is needed inside the scan and the encoder is even more compact than
  RVM+;
* validated cross-modality robustness on 14 benchmarks across nine
  imaging modalities (dermoscopy, ultrasound, MRI, OCT, histology,
  blood-cell microscopy, CT, X-ray, retinal fundus);
* a 3-seed pipeline with documented inference protocol (TTA + ensemble
  + default 0.5 threshold), so reproducibility infrastructure is
  already written.

## 3. Six concrete extensions (ranked)

Each extension lists the gap in VM-MedSAM, the HamVision component that
fills it, and the experiment that demonstrates the gain.

### Extension 1: Hamiltonian SS2D image encoder (highest priority)

* **Gap.** RVM+ exposes only one feature stream. The Hausdorff loss in
  §4 of VM-MedSAM is the price of not having a built-in derivative
  signal.
* **Fix.** Replace each RVM+ stage with a `HamiltonianBottleneck`
  (`src/hamiltonian.py`). The encoder now emits `(f, p, H_map)` per
  spatial location instead of a single tensor.
* **Output wiring.** `f` goes into the SAM-style mask decoder exactly
  where the RVM+ output went. `p` is exposed to the loss (extension 3).
  `H_map` is exposed to the prompt-free pipeline (extension 2).
* **Experiment.** Re-run VM-MedSAM's training script with the encoder
  swap, no other changes. Report Dice, 95%-Hausdorff, parameter count,
  inference FLOPs on the abdomen-CT split.
* **Hypothesis.** Dice ± 0.5 pp, Hausdorff falls by ≥ 5%, parameter
  count drops by another ~10% relative to VM-MedSAM (Hamiltonian
  bottleneck has no LayerNorm overhead).

### Extension 2: Prompt-free segmentation via energy-derived boxes

* **Gap.** VM-MedSAM still needs a user-drawn bounding box at inference.
* **Fix.** Use the energy map `H_map` to generate the box automatically.
  Algorithm (`src/prompt_free.py`):
  1. Compute `H_map` via the Hamiltonian bottleneck.
  2. Threshold at the top-20% energy quantile (per-image, not global).
  3. Take the bounding box of the largest connected component.
  4. Feed that box to SAM's frozen prompt encoder.
* **Experiment.** Two settings: (a) box-prompted (gold-standard
  human-drawn box, matches Li et al.); (b) prompt-free
  (auto-generated box from `H_map`). Report Dice and intersection-over-union
  of the auto-box against the GT box. Inference becomes a single forward
  pass with no user input.
* **Hypothesis.** Dice drops by ≤ 3 pp in the prompt-free setting on
  abdomen CT; on the high-contrast modalities in HamVision's suite (ISIC
  dermoscopy, TN3K thyroid US, ACDC cardiac MRI), the drop is ≤ 1 pp
  because the energy map is sharp.

### Extension 3: Momentum-supervised boundary loss replaces Hausdorff

* **Gap.** Hausdorff loss acts on the final mask -- way downstream of
  the encoder.
* **Fix.** Supervise the encoder momentum directly:

  $$\mathcal{L}_p = \big\| \mathrm{softplus}(|p|_{\mathrm{avg}}) - \mathrm{DT}(\partial y_{\mathrm{gt}}) \big\|_1$$

  where `|p|_avg` is the channel-averaged momentum magnitude and
  `DT(∂y_gt)` is the (clipped, normalised) distance transform of the
  GT boundary. The loss enforces high momentum exactly where the
  boundary is; the propagation through the rest of the network is
  implicit.
* **Implementation.** `src/losses.py::MomentumBoundaryLoss`.
* **Experiment.** Three-way ablation: (i) MedSAM original cross-entropy
  + Dice; (ii) Li et al.'s setting (i + Hausdorff-on-mask); (iii)
  ours (i + momentum-on-feature). Report Dice and 95%-Hausdorff. If
  ours wins on Hausdorff with no Dice regression, the loss is more
  principled.
* **Hypothesis.** Momentum-on-feature reduces 95%-Hausdorff by ~15%
  relative to Hausdorff-on-mask, because gradient flow is much shorter
  from the loss to the layer the loss is supervising.

### Extension 4: Phase-Space Spectral Pooling in the mask decoder

* **Gap.** The SAM mask decoder is a small transformer that does
  cross-attention between image and prompt embeddings. It is the
  bottleneck for "detailed features" that Li et al. explicitly flagged.
* **Fix.** Augment the decoder input with PSSP features computed on
  the complex bottleneck signal `z = q + i·p`:
  1. Form `z` along each row of the bottleneck output.
  2. Per-row FFT, retain `K = 12` low-frequency bins.
  3. Concatenate real, imaginary, and magnitude components of `Z` to
     the decoder's cross-attention key.
* **Implementation.** A new `PSSPDecoderAdapter` in
  `src/ham_medsam.py` that wraps the SAM mask decoder.
* **Experiment.** Ablate the four configurations: with vs. without
  PSSP × with vs. without momentum-supervised loss. Headline metric is
  small-organ Dice (gallbladder, adrenal, oesophagus), where boundary
  detail matters most.
* **Hypothesis.** PSSP gives ≥ 1 pp Dice on small organs without
  affecting large-organ Dice.

### Extension 5: Multi-class panel from per-channel energy

* **Gap.** SAM/MedSAM/VM-MedSAM produce one binary mask per prompt; for
  12 organs that is 12 forward passes.
* **Fix.** Before the SE attention collapses energy to a single
  channel, the per-channel energy tensor `H_c(i, j)` is itself
  multi-modal. Attach a small linear head `H_c → 12-class probabilities`
  and supervise against the GT one-hot label. All 12 masks come from
  one forward pass.
* **Implementation.** `MultiClassEnergyHead` in `src/ham_medsam.py`.
  Trained on top of a frozen Hamiltonian encoder so the head is
  cheap.
* **Experiment.** Compare (i) VM-MedSAM with 12 prompted forward passes
  versus (ii) Ham-MedSAM with a single multi-class forward pass.
  Report per-organ Dice plus inference wall-clock time per slice.
* **Hypothesis.** Wall-clock per slice drops by 6-10× with negligible
  Dice loss.

### Extension 6: Zero-shot cross-modality evaluation

* **Gap.** VM-MedSAM trains jointly on its five modalities; it never
  demonstrates *zero-shot* transfer to an unseen modality.
* **Fix.** Train Ham-MedSAM on the abdomen-CT split (matching their
  protocol exactly), then zero-shot it on:
  * **ISIC 2018** (dermoscopy, binary lesion segmentation),
  * **TN3K** (thyroid ultrasound, binary nodule segmentation),
  * **ACDC** (cardiac MRI, 4-class).
  Report Dice with and without per-domain fine-tuning.
* **Hypothesis.** Hamiltonian inductive bias generalises: the
  zero-shot Dice on dermoscopy and cardiac MRI is within 5 pp of
  HamSeg's fully supervised number, while VM-MedSAM's would not be
  (because its numbers come from joint training on each domain; a
  train-on-CT model has no zero-shot evidence in the paper).

## 4. Phased implementation plan

```
Phase 0 (week 1)   Reproduce VM-MedSAM baseline on abdomen CT
                    -- pin their training script, get their reported numbers
                    -- this is the apples-to-apples baseline for every
                       subsequent ablation
Phase 1 (weeks 2-3) Extension 1: encoder swap
                    -- drop in HamiltonianBottleneck in place of RVM+
                    -- match training schedule exactly
                    -- report Dice / Hausdorff / parameter / FLOP delta
Phase 2 (week 4)    Extension 3: momentum-supervised boundary loss
                    -- ablate Hausdorff-on-mask vs momentum-on-feature
                    -- this is the "small change, big interpretability win"
                       contribution
Phase 3 (weeks 5-6) Extension 2: prompt-free pipeline
                    -- implement H -> bbox extractor (no training needed
                       to start; just an inference-time module)
                    -- report Dice in both prompted and prompt-free modes
Phase 4 (week 7)    Extension 4: PSSP in mask decoder
                    -- ablate at small-organ subset only
                    -- relatively low-risk; can be tabled if time is tight
Phase 5 (week 8)    Extension 6: cross-modality zero-shot
                    -- the headline generalisation result
                    -- uses the same checkpoints from Phase 1, no new
                       training
Phase 6 (weeks 9-10) Extension 5: multi-class panel
                    -- inference-speed contribution
                    -- only needed if 12-organ wall-clock is a story we
                       want to push
Phase 7 (weeks 11-12) Writing, figures, response to reviewers
```

Phases 4-6 can be parallelised or partially dropped depending on
timeline pressure. Phases 0-3 are the spine of the paper.

## 5. Datasets

| Dataset | Modality | Source | Used for |
|---|---|---|---|
| FLARE22 / BTCV | Abdomen CT, 12 organs | public challenges | Phase 0-3 (matches Li et al.) |
| Lung cancer | CT | LIDC subset (Li et al. used 600+ images, source not specified) | Phase 0 reproduction only |
| Brain tumour | MRI | BraTS subset | Phase 0 reproduction only |
| ISIC 2018 | Dermoscopy | public | Phase 5 zero-shot |
| TN3K | Thyroid US | public | Phase 5 zero-shot |
| ACDC | Cardiac MRI | public | Phase 5 zero-shot |

The data preparation script in `data/prepare_data.py` (to be ported from
the HamVision release) downloads and normalises everything in the public
datasets list. The two from Li et al.'s paper that are not openly
available (lung-cancer, brain-tumour) are needed only for the Phase 0
baseline reproduction and can be substituted with LIDC-IDRI and BraTS
2021 if the original datasets are not accessible.

## 6. Evaluation protocol

Following the HamVision pipeline (which itself follows MedSAM and
VM-MedSAM):

* **Splits.** Official train/validation/test partitions from each
  challenge -- never custom cross-validation.
* **Seeds.** Three random seeds per benchmark, report mean ± std.
* **Metrics.** Dice (always), 95%-Hausdorff distance (for the boundary
  story), mIoU, sensitivity, specificity, parameter count, training
  speed (steps/sec), inference FLOPs per 1024 × 1024 slice.
* **Inference protocol.** Default 0.5 threshold; no per-dataset tuning.
  4-direction TTA + 3-seed probability ensemble is optional and reported
  separately if used (matches HamVision's "deployment-mode" disclosure).

## 7. Baselines

The comparison table contains, at minimum:

* SAM (no medical fine-tuning),
* MedSAM (Ma et al. 2024),
* Med-SA (Wu et al.),
* 3DSAM-Adapter (Gong et al.),
* VM-MedSAM (Li et al. 2026, the paper we are extending),
* HamSeg (HamVision, the baseline that does **not** use SAM),
* Ham-MedSAM (ours, with the six extensions).

The two most important comparisons are Ham-MedSAM vs VM-MedSAM (does the
encoder swap help?) and Ham-MedSAM vs HamSeg (does the SAM prompt
encoder + decoder add value over a pure Hamiltonian U-Net?). Either
direction has a story.

## 8. Compute budget

Phase 0-2 fit comfortably on a single A100 / H100:

* Encoder swap re-trains in roughly the same wall-clock time as
  VM-MedSAM's reported training, because HamiltonianBottleneck has
  similar FLOPs to RVM+ at the bottleneck resolution.
* Three-seed sweep on abdomen CT: ~ 36 GPU-hours.
* Momentum-loss ablation: another ~ 24 GPU-hours.
* Prompt-free pipeline: no extra training, only inference.
* Zero-shot transfer: no extra training, only inference.

Total budget for Phases 0-3 plus Phase 5: under 100 single-GPU hours.
Phase 4 adds another ~ 24 GPU-hours if pursued. Phase 6 (multi-class
head) is essentially a fine-tuning sprint on top of the Phase 1
checkpoint.

## 9. Risk analysis

* **Risk:** the Hamiltonian encoder is too narrow a primitive for SAM's
  decoder, which was trained against ViT features. **Mitigation:** keep
  a small ConvNeXt residual path inside the bottleneck (the
  `ablation='none'` default in `HamiltonianBottleneck`), so the decoder
  always sees feature dimensionality similar to ViT outputs.
* **Risk:** energy-derived bounding boxes are noisy on low-contrast
  modalities (e.g., abdominal CT soft-tissue boundaries). **Mitigation:**
  fall back to a smoothed energy map and a top-K connected-component
  pruning step; the prompt-free pipeline is itself an ablation, so the
  Dice number with the auto-box is the result, not a claim about
  matching the manual box exactly.
* **Risk:** momentum loss interferes with the segmentation head's
  gradients. **Mitigation:** weight the momentum loss with a small λ
  (start at 0.1, sweep on validation), and detach the segmentation
  branch's gradient from the momentum supervision via a stop-gradient
  on `p` if the two objectives conflict.

## 10. Suggested paper outline

Title: *Hamiltonian-Augmented MedSAM: Prompt-Free, Interpretable, and
Cross-Modal Medical Segmentation.*

1. Introduction (problem, three claims: encoder, prompt-free,
   cross-modal).
2. Related work (SAM-medical family, Mamba-medical family,
   Hamiltonian / SSM theory).
3. Method:
   * 3.1 Hamiltonian image encoder (extension 1).
   * 3.2 Energy-driven prompt-free pipeline (extension 2).
   * 3.3 Momentum-supervised boundary loss (extension 3).
   * 3.4 (Optional) PSSP-augmented mask decoder (extension 4).
4. Experiments:
   * 4.1 Abdomen CT (matched protocol vs VM-MedSAM).
   * 4.2 Lung cancer / brain tumour (matched protocol vs VM-MedSAM).
   * 4.3 Zero-shot cross-modality (ISIC, TN3K, ACDC).
   * 4.4 Ablations (encoder, loss, prompt-free, PSSP).
5. Discussion + limitations.
6. Conclusion.

Headline numbers to land:

* Parameter count ≤ VM-MedSAM with ≥ 1 pp Dice on abdomen CT.
* Dice gap ≤ 3 pp between box-prompted and prompt-free Ham-MedSAM.
* 95%-Hausdorff ≤ Hausdorff-on-mask training on the same model, with
  no Dice regression.
* Zero-shot Dice on ISIC, TN3K, ACDC within 5 pp of HamSeg's fully
  supervised number, and ≥ 10 pp above VM-MedSAM zero-shot.

## 11. Next concrete steps

1. Read `notes/li2026_vm_medsam.md` for the complete paper summary.
2. Get the VM-MedSAM official code or re-implement from the paper
   description.
3. Reproduce their abdomen-CT numbers (Phase 0).
4. Fill in the stub bodies in `src/ham_encoder.py`,
   `src/ham_medsam.py`, `src/losses.py`, `src/prompt_free.py`.
5. Run Phase 1 (encoder swap) and verify the Dice / parameter / FLOP
   target numbers in Section 3.
