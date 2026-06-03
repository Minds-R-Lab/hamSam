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

> Status: research code. The novel components and the full train/eval pipeline
> run end-to-end and are unit-tested on CPU (see "What is tested"). Real
> benchmark numbers require a GPU, a MedSAM ViT-B checkpoint, and the datasets.

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

### Real training (H100)
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

## What is tested (ran in CI/CPU)
- `hamiltonian.py`: 4-direction reversal, reshape round-trip, sqrt(L) stability
  to L=512, ablation shapes, differentiability.
- `ham_encoder.py`: 64×64 output at 1024 input, all three placements, backward.
- `losses.py`: Dice/Hausdorff/momentum (all three targets) + combined backward.
- `prompt_free.py`: box localisation + empty-energy fallback.
- `ham_medsam.py`: forward+backward for all 6 configs (box / prompt-free / PSSP
  / multiclass / every-stage / baseline).
- Full pipeline: train → checkpoint → prompt-free eval → zero-shot eval.

## What is NOT verified here
GPU/AMP paths, the real `segment_anything` decoder (a fallback stands in for
CPU tests — never use it for reported metrics), dataset converters in
`data/prepare_data.py` (need challenge access), and any accuracy claim. These
are code-complete but unrun in this environment.

## Notable correction
`src/hamiltonian.py` fixes a bug in the upstream copy where two of the four
scan directions reversed the channel axis instead of the spatial axis — so the
"four-direction scan" was effectively two. See the file's CHANGELOG and
`PLAN.md` §0.

## Attribution
- Hamiltonian primitives derive from the HamVision/HamSeg release (with the fix
  above). Cite HamVision.
- Extends VM-MedSAM; cite Li et al. (2026), DOI 10.1002/mp.70492.
