"""Training entry point for Ham-MedSAM.

Mirrors VM-MedSAM's recipe so Phase-0 (baseline) vs Phase-1+ (extensions) are
apples-to-apples: AdamW, cosine annealing, Dice+BCE(+Hausdorff/+momentum), box
prompts perturbed 0-20px, mask decoder frozen for the first N epochs then
unfrozen. The Hamiltonian encoder trains from scratch; SAM prompt encoder +
mask decoder load from a MedSAM checkpoint (frozen prompt encoder).

Examples
--------
# smoke test, no data/checkpoint needed (CPU):
python experiments/train_ham_medsam.py --config configs/ham_medsam_abdomen.yaml \
    --data synthetic --epochs 1 --batch_size 2 --input_size 128 --output_dir /tmp/run

# Phase 1 (encoder swap) on real data + MedSAM checkpoint:
python experiments/train_ham_medsam.py --config configs/ham_medsam_abdomen.yaml \
    --encoder ham --loss dice+ce --seed 42 --output_dir outputs/ham_medsam/seed_42
"""
import argparse
import json
import os
import sys

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ham_medsam import HamMedSAM        # noqa: E402
from src.losses import CombinedLoss          # noqa: E402
from src.metrics import dice_score           # noqa: E402
from experiments._common import (set_seed, build_loader, LOSS_FLAGS,  # noqa: E402
                                 amp_autocast, make_grad_scaler, pick_amp_dtype)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--encoder", default="ham", choices=["ham", "baseline", "rvm_plus"])
    p.add_argument("--bottleneck", default=None, choices=[None, "deepest", "all", "none"])
    p.add_argument("--ablation", default=None, choices=[None, "none", "A", "B"],
                   help="bottleneck ablation: A=ConvNeXt-only (capacity-matched "
                        "control), B=oscillator-only. Overrides config.")
    p.add_argument("--loss", default="dice+ce", choices=list(LOSS_FLAGS))
    p.add_argument("--momentum_signal", default="momentum",
                   choices=["momentum", "grad_energy", "combo"],
                   help="boundary signal for the momentum loss: raw momentum, "
                        "|grad energy| (boundary-peaked), or learned combo.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sam_checkpoint", default=None,
                   help="override model.sam_checkpoint (MedSAM/SAM ViT-B .pth; needs input 1024)")
    p.add_argument("--early_stop_patience", type=int, default=None,
                   help="stop if val Dice does not improve for N validations "
                        "(uniform across runs; 0/None=off). Overrides config.")
    p.add_argument("--energy_prompt", choices=["box", "dense", "learned"], default=None,
                   help="prompt-free prompt type: box (energy->bbox) or dense "
                        "(energy map as SAM mask prompt). Overrides config.")
    p.add_argument("--prompt_free", action="store_true",
                   help="train prompt-free: the model derives its box from its own "
                        "energy map each step (use only on SINGLE-TARGET data).")
    p.add_argument("--prompt_free_after", type=int, default=None,
                   help="warm-start: train box-prompted, then switch to prompt-free at "
                        "this epoch (single-target). Recovers box-level quality before "
                        "adapting to energy boxes.")
    p.add_argument("--data", default=None, help="override data_root; 'synthetic' for smoke test")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--input_size", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg, tcfg = cfg["model"], cfg["train"]
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    input_size = args.input_size or cfg["input_size"]
    epochs = args.epochs or tcfg["epochs"]
    batch_size = args.batch_size or tcfg["batch_size"]
    data_root = args.data or cfg.get("data_root") or cfg.get("data_roots")
    multiclass = cfg.get("multiclass", False)
    num_classes = cfg.get("num_classes", 1)

    # encoder flag -> bottleneck placement
    if args.sam_checkpoint is not None:
        mcfg["sam_checkpoint"] = args.sam_checkpoint   # persisted via saved cfg
    if args.prompt_free:
        mcfg["prompt_free"] = True
    if args.energy_prompt is not None:
        mcfg["energy_prompt"] = args.energy_prompt
    if args.ablation is not None:
        mcfg["ablation"] = args.ablation
    if args.prompt_free_after is not None:
        mcfg["prompt_free"] = False   # warm-start box-prompted, flip later
    bottleneck = args.bottleneck or mcfg["bottleneck"]
    if args.encoder in ("baseline", "rvm_plus"):
        bottleneck = "none"

    device = torch.device(args.device)
    model = HamMedSAM(
        sam_checkpoint=mcfg.get("sam_checkpoint"), model_type=mcfg.get("model_type", "vit_b"),
        backend=mcfg.get("backend", "medsam_vitb"),
        bottleneck=bottleneck, ablation=mcfg.get("ablation", "none"),
        prompt_free=mcfg.get("prompt_free", False),
        use_pssp_decoder=mcfg.get("use_pssp_decoder", False),
        multiclass_head=mcfg.get("multiclass_head", False),
        num_classes=num_classes, input_size=input_size,
        energy_prompt=mcfg.get("energy_prompt", "box"),
    ).to(device)

    loss_fn = CombinedLoss(multiclass=multiclass, num_classes=num_classes,
                           momentum_signal=args.momentum_signal,
                           **LOSS_FLAGS[args.loss]).to(device)

    train_loader = build_loader(data_root, "train", cfg, batch_size, True,
                                input_size, multiclass, num_classes)
    try:
        val_loader = build_loader(data_root, "val", cfg, batch_size, False,
                                  input_size, multiclass, num_classes)
    except Exception:
        val_loader = train_loader  # synthetic / no val split

    # VM-MedSAM freezes the mask decoder for the first N epochs, then unfreezes.
    unfreeze_ep = tcfg.get("decoder_unfreeze_epoch", 10)
    if unfreeze_ep > 0:
        model.set_mask_decoder_trainable(False)
    # Optimiser holds ALL model+loss params (incl. the to-be-unfrozen decoder and
    # the projection-mode momentum head). PyTorch skips params whose grad is None,
    # so frozen params stay untouched until unfrozen mid-run.
    train_params = list(model.parameters()) + list(loss_fn.parameters())
    opt = AdamW(train_params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=epochs) if tcfg.get("cosine", True) else None
    use_amp = tcfg.get("amp", True) and device.type == "cuda"
    amp_dtype = pick_amp_dtype(device.type, use_amp)
    # GradScaler is only needed for fp16 (bf16 has fp32 range and doesn't underflow).
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = make_grad_scaler(device.type, use_scaler)
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)
    if use_amp:
        print(f"AMP enabled with dtype={str(amp_dtype).split('.')[-1]} "
              f"(grad scaler={'on' if use_scaler else 'off'}, clip={max_grad_norm})")
    nonfinite = 0

    json.dump(vars(args) | {"bottleneck": bottleneck}, open(
        os.path.join(args.output_dir, "args.json"), "w"), indent=2)

    patience = (args.early_stop_patience if args.early_stop_patience is not None
                else tcfg.get("early_stop_patience", 0))
    best = -1.0
    since_improve = 0
    pf_after = args.prompt_free_after
    for ep in range(epochs):
        if ep == unfreeze_ep:
            model.set_mask_decoder_trainable(True)
        if pf_after is not None and ep == pf_after and not model.prompt_free:
            model.prompt_free = True       # switch to prompt-free
            best, since_improve = -1.0, 0  # metric regime changed -> reset
            print(f"epoch {ep+1}: switching to PROMPT-FREE training "
                  f"(energy-derived boxes); best/patience reset")
        model.train()
        for batch in train_loader:
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)
            box = batch["box"].to(device)
            opt.zero_grad()
            with amp_autocast(device.type, use_amp, amp_dtype):
                out = model(img, box=None if model.prompt_free else box)
                target = tgt.squeeze(1) if multiclass else tgt
                loss, parts = loss_fn(out["mask"], target, p=out["p"], energy=out["H_map"])
            if not torch.isfinite(loss):       # guard: skip a bad batch
                nonfinite += 1
                continue
            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(train_params, max_grad_norm)
            scaler.step(opt); scaler.update()
        if sched:
            sched.step()

        # validation (Dice)
        if (ep + 1) % tcfg.get("val_interval", 1) == 0:
            model.eval(); dices = []
            with torch.no_grad():
                for batch in val_loader:
                    img = batch["image"].to(device); box = batch["box"].to(device)
                    out = model(img, box=None if model.prompt_free else box)
                    if multiclass:
                        pred = out["mask"].argmax(1) > 0
                        gt = batch["mask"].to(device) > 0
                    else:
                        pred = torch.sigmoid(out["mask"]) > 0.5
                        gt = batch["mask"].to(device) > 0.5
                    for b in range(img.shape[0]):
                        dices.append(dice_score(pred[b], gt[b]))
            vdice = sum(dices) / len(dices)
            nf = f"  nonfinite_batches={nonfinite}" if nonfinite else ""
            print(f"epoch {ep+1}/{epochs}  loss={loss.item():.4f}  val_dice={vdice:.4f}{nf}")
            if vdice > best:
                best = vdice
                since_improve = 0
                torch.save({"model": model.state_dict(), "epoch": ep, "val_dice": vdice,
                            "config": cfg, "args": vars(args)},
                           os.path.join(args.output_dir, "best.ckpt"))
            else:
                since_improve += 1
                if patience and since_improve >= patience:
                    print(f"early stop: no val improvement for {patience} validations "
                          f"(best={best:.4f})")
                    break
    print(f"done. best val_dice={best:.4f}  -> {args.output_dir}/best.ckpt")


if __name__ == "__main__":
    main()
