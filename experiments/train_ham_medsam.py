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
from experiments._common import set_seed, build_loader, LOSS_FLAGS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--encoder", default="ham", choices=["ham", "baseline", "rvm_plus"])
    p.add_argument("--bottleneck", default=None, choices=[None, "deepest", "all", "none"])
    p.add_argument("--loss", default="dice+ce", choices=list(LOSS_FLAGS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
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
    ).to(device)

    loss_fn = CombinedLoss(multiclass=multiclass, num_classes=num_classes,
                           **LOSS_FLAGS[args.loss]).to(device)

    train_loader = build_loader(data_root, "train", cfg, batch_size, True,
                                input_size, multiclass, num_classes)
    try:
        val_loader = build_loader(data_root, "val", cfg, batch_size, False,
                                  input_size, multiclass, num_classes)
    except Exception:
        val_loader = train_loader  # synthetic / no val split

    # include loss params (the projection-mode momentum loss has a learned head)
    train_params = [q for q in list(model.parameters()) + list(loss_fn.parameters())
                    if q.requires_grad]
    opt = AdamW(train_params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=epochs) if tcfg.get("cosine", True) else None
    use_amp = tcfg.get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    unfreeze_ep = tcfg.get("decoder_unfreeze_epoch", 10)

    json.dump(vars(args) | {"bottleneck": bottleneck}, open(
        os.path.join(args.output_dir, "args.json"), "w"), indent=2)

    best = -1.0
    for ep in range(epochs):
        if ep == unfreeze_ep:
            model.set_mask_decoder_trainable(True)
        model.train()
        for batch in train_loader:
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)
            box = batch["box"].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(img, box=None if model.prompt_free else box)
                target = tgt.squeeze(1) if multiclass else tgt
                loss, parts = loss_fn(out["mask"], target, p=out["p"])
            scaler.scale(loss).backward()
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
            print(f"epoch {ep+1}/{epochs}  loss={loss.item():.4f}  val_dice={vdice:.4f}")
            if vdice > best:
                best = vdice
                torch.save({"model": model.state_dict(), "epoch": ep, "val_dice": vdice,
                            "config": cfg, "args": vars(args)},
                           os.path.join(args.output_dir, "best.ckpt"))
    print(f"done. best val_dice={best:.4f}  -> {args.output_dir}/best.ckpt")


if __name__ == "__main__":
    main()
