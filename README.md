# hamSam — Hamiltonian-augmented MedSAM (Ham-MedSAM)

Extends **VM-MedSAM** (Li et al., *Vision mamba augmented segment anything
model for medical image segmentation*, Med. Phys. 53(5):e70492, 2026,
DOI 10.1002/mp.70492) with the **Hamiltonian bottleneck** of **HamVision /
HamSeg**. The image encoder is replaced by a Hamiltonian oscillator encoder
that emits, at every spatial location, three structured maps:

| output | meaning | used for |
|---|---|---|
| `feat` (q) | filtered features | SAM mask decoder input |
| `p` (momentum) | band-pass spatial derivative | encoder-level boundary loss |
| `H_map` (energy) | parameter-free saliency | prompt-free bounding box |

This gives three contributions over VM-MedSAM: an interpretable encoder, an
**encoder-level momentum boundary loss** (a learned head reads a boundary map
out of the momentum `p`; vs. VM-MedSAM's post-hoc Hausdorff on the mask), and
**prompt-free** inference (vs. a required user box).

> Status: research code. Every component -- including the REAL segment-anything
> prompt encoder + mask decoder -- runs end-to-end and is unit-tested on CPU
> (see "What is tested"). The real SAM decoder is built from scratch when no
> checkpoint is given, so the full stack is turnkey without any download. Real
> benchmark *numbers* still require a GPU, pretrained MedSAM/SAM ViT-B weights,
> and the datasets.

## Run it on the H100 (copy-paste)

```bash
# 0. Get the latest code. Downloaded data/checkpoints are git-ignored, so a
#    pull won't conflict with anything you've already fetched.
cd hamSam && git pull

# 1. One-time deps (envs cloned before nibabel was pinned need this).
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git \
  || pip install segment-anything-py

# 2. Verify every component on the GPU (real SAM decoder, all configs/losses).
bash scripts/run_smoke.sh

# 3. (Optional) pretrained ViT-B weights. Random-init works for wiring tests;
#    pretrained MedSAM/SAM is needed for meaningful accuracy.
bash scripts/download_checkpoints.sh          # -> checkpoints/sam_vit_b_01ec64.pth

# 4. Data: FLARE22 auto-downloads AND converts (open, Zenodo). BTCV prints its
#    gated Synapse steps.
bash scripts/download_datasets.sh             # -> data/processed/flare22/{train,val,test}

# 5. Train on FLARE22 (single-dataset config; use vmdata only once all 8 exist).
python experiments/train_ham_medsam.py --config configs/ham_medsam_abdomen.yaml \
    --encoder ham --bottleneck deepest --loss dice+ce+momentum --seed 42 \
    --output_dir outputs/ham/seed_42 --device cuda
```

To use pretrained weights, set `model.sam_checkpoint: checkpoints/sam_vit_b_01ec64.pth`
in `configs/ham_medsam_abdomen.yaml` (keep `input_size: 1024`). Before trusting
per-organ names, open one converted `org1_liver` mask and confirm the label
order matches your FLARE22 release (else pass `--label_map_json`).

## Layout
```
src/        hamiltonian.py (primitive, bug-fixed) · ham_encoder.py · ham_medsam.py
            losses.py · prompt_free.py · metrics.py · sam_utils.py
data/       sam_preprocess.py · datasets.py · prepare_data.py · README.md
experiments train_ham_medsam.py · eval_zero_shot.py · eval_prompt_free.py · _common.py
visualize/  visualize_energy_prompt.py
configs/    ham_medsam_abdomen.yaml · vm_medsam_baseline_abdomen.yaml · zero_shot.yaml
tests/      pytest suite (15 tests)
notes/PLAN  paper summary + research plan (see PLAN.md §0 for review corrections)
```

## Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For real runs also install SAM and download a MedSAM ViT-B checkpoint:
# pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Quick start
```bash
# CPU smoke test — no data or checkpoint needed (uses synthetic shapes + a
# fallback decoder); proves the pipeline end-to-end:
python experiments/train_ham_medsam.py --config configs/ham_medsam_abdomen.yaml \
    --data synthetic --epochs 1 --batch_size 2 --input_size 128 --output_dir /tmp/run

# Run the test suite:
pytest tests/ -q
```

### Turnkey setup + verification (H100)
```bash
bash scripts/setup_h100.sh           # deps + segment-anything + sanity import
bash scripts/run_smoke.sh            # unit tests + real-SAM component smoke + e2e
bash scripts/download_checkpoints.sh # optional: pretrained SAM/MedSAM ViT-B
```
`scripts/run_smoke.sh` runs `experiments/smoke_real.py`, which forwards AND
backwards every config (box / prompt-free / PSSP / multiclass / all encoder
placements) and every loss through the REAL SAM decoder, then a 1-epoch
synthetic train + prompt-free eval. If it prints "ALL SMOKE CHECKS PASSED",
the stack is ready.

### Real training
```bash
# Phase 0 baseline (pure-ConvNeXt encoder, VM-MedSAM-style):
python experiments/train_ham_medsam.py --config configs/vm_medsam_baseline_abdomen.yaml \
    --encoder baseline --loss dice+ce+hausdorff --seed 42 \
    --output_dir outputs/baseline/seed_42
# Phase 1 encoder swap + Phase 2 momentum loss:
python experiments/train_ham_medsam.py --config configs/ham_medsam_abdomen.yaml \
    --encoder ham --bottleneck deepest --loss dice+ce+momentum --seed 42 \
    --output_dir outputs/ham/seed_42
# Phase 3 prompt-free, Phase 5 zero-shot (no extra training):
python experiments/eval_prompt_free.py --checkpoint outputs/ham/seed_42/best.ckpt --data data/processed/flare22 --output_dir outputs/ham/seed_42/pf
python experiments/eval_zero_shot.py   --checkpoint outputs/ham/seed_42/best.ckpt --prompt_mode auto --output_dir outputs/ham/seed_42/zs
```
Set `model.sam_checkpoint` in the YAML to your MedSAM ViT-B `.pth`. Encoder
placement is configurable: `--bottleneck {deepest,all,none}` (`none` is the
baseline encoder).

**Datasets.** Uses VM-MedSAM's exact eight (BTCV, FLARE22, MSD Lung, BraTS,
CVC-ClinicDB, BUSI, DRIVE, Montgomery), trained jointly via
`configs/ham_medsam_vmdata.yaml`. ISIC/TN3K/ACDC are optional out-of-distribution
zero-shot probes only (not in VM-MedSAM). See `data/README.md`.

**Backend.** `model.backend=medsam_vitb` (SAM ViT-B) is the default and the
only one wired, because VM-MedSAM uses it and its 256x64x64 embedding matches
`HamEncoder`. `sam2`/`medsam2` (need multi-scale FPN features) and `sam3`
(text/concept paradigm) are documented extension points that raise on use.

## Evaluate + the comparison runs that make the paper

After a run finishes, get the per-organ breakdown on the held-out TEST split
(not just the pooled val number):
```bash
python experiments/eval_test.py --checkpoint outputs/ham/seed_42/best.ckpt \
    --data data/processed/flare22 --device cuda           # add --prompt_mode auto for prompt-free
```
The scientifically essential comparisons (run each at input 1024, ideally with
`model.sam_checkpoint` set to MedSAM/SAM ViT-B weights):
1. **Encoder ablation** -- Ham vs plain-ConvNeXt baseline, identical settings:
   `--encoder ham --bottleneck deepest`  vs  `--encoder baseline` (bottleneck=none).
2. **Loss ablation** -- `--loss dice+ce+hausdorff` (VM-MedSAM-style) vs
   `--loss dice+ce+momentum` (ours) vs `dice+ce`.
3. **Prompt-free** -- `eval_test.py --prompt_mode auto` vs `box`.
A random-init SAM decoder (no checkpoint) trains to a sane ~0.89 pooled val
Dice on FLARE22, but is a sanity baseline only -- set the checkpoint for
reportable numbers.

## What is tested (ran in CI/CPU)
- `hamiltonian.py`: 4-direction reversal, reshape round-trip, sqrt(L) stability
  to L=512, ablation shapes, differentiability.
- `ham_encoder.py`: 64×64 output at 1024 input, all three placements, backward.
- `losses.py`: Dice/Hausdorff/momentum (all three targets) + combined backward.
- `prompt_free.py`: box localisation + empty-energy fallback.
- `ham_medsam.py`: forward+backward for all 6 configs (box / prompt-free / PSSP
  / multiclass / every-stage / baseline).
- Real SAM path: `smoke_real.py` runs all 6 model configs + 4 losses through
  the actual segment-anything PromptEncoder + MaskDecoder (forward+backward);
  `tests/test_sam_real.py` adds 3 more (skipped only if the package is absent).
- Full pipeline: train (freeze→unfreeze decoder) → checkpoint → prompt-free /
  zero-shot eval.

## What is NOT verified here
The real `segment_anything` decoder path IS now exercised (from-scratch build,
`smoke_real.py` + `tests/test_sam_real.py`). Still unverified in this
environment: accuracy with *pretrained* MedSAM/SAM weights, GPU/AMP at full
1024 resolution, the dataset converters in `data/prepare_data.py` (need
challenge access), and any accuracy claim. The fallback conv decoder is only a
last resort when segment-anything is absent — never use it for reported metrics.

## Notable correction
`src/hamiltonian.py` fixes a bug in the upstream copy where two of the four
scan directions reversed the channel axis instead of the spatial axis — so the
"four-direction scan" was effectively two. See the file's CHANGELOG and
`PLAN.md` §0.

## Attribution
- Hamiltonian primitives derive from the HamVision/HamSeg release (with the fix
  above). Cite HamVision.
- Extends VM-MedSAM; cite Li et al. (2026), DOI 10.1002/mp.70492.
