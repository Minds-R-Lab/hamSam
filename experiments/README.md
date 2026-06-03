# Experiments

This folder will hold the training and evaluation entry points. The
recipe follows the phased plan in `../PLAN.md` §4.

## Phase 0 - reproduce VM-MedSAM baseline

Goal: get the abdomen-CT Dice / Hausdorff numbers reported by Li et al.
(2026) on the local hardware, so subsequent extensions are compared on
an apples-to-apples baseline.

```
python experiments/train_vm_medsam_baseline.py \
    --config configs/vm_medsam_abdomen.yaml
```

## Phase 1 - Hamiltonian encoder swap

```
python experiments/train_ham_medsam.py \
    --config configs/ham_medsam_abdomen.yaml \
    --encoder ham   --loss dice+ce
```

## Phase 2 - momentum-supervised boundary loss

```
python experiments/train_ham_medsam.py \
    --config configs/ham_medsam_abdomen.yaml \
    --encoder ham   --loss dice+ce+momentum
```

## Phase 3 - prompt-free inference (no training)

```
python experiments/eval_prompt_free.py \
    --checkpoint outputs/ham_medsam/seed_42/best.ckpt \
    --data data/processed/flare22 --quantile 0.80
```

## Phase 5 - cross-modality zero-shot

```
python experiments/eval_zero_shot.py \
    --checkpoint outputs/ham_medsam/seed_42/best.ckpt \
    --datasets isic2018 tn3k acdc
```

## Suggested directory layout for runs

```
outputs/
├── vm_medsam_baseline/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
└── ham_medsam/
    ├── seed_42/
    ├── seed_43/
    └── seed_44/
```

Match the HamVision convention: every config file is JSON or YAML,
every seed gets its own subfolder, and `args.json` is dumped on training
start. This is the structure the `run_multi_seed.py` orchestrator in
HamVision already understands; port it over once Phase 1 is running.
