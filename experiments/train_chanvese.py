"""Train the deep Chan-Vese phase-field segmenter (prompt-free) + test eval.

No box, no SAM: this is a from-scratch physics-native segmenter. The control
(--no_physics) is the same network with the relaxation removed, so the only
difference is whether the variational dynamics are in the loop.

    python experiments/train_chanvese.py --data <ISIC2017> --output_dir runs/cv/phys_s42
    python experiments/train_chanvese.py --data <ISIC2017> --no_physics --output_dir runs/cv/ctrl_s42
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deep_chanvese import DeepChanVese          # noqa: E402
from src.losses import CombinedLoss                  # noqa: E402
from src.metrics import dice_score, iou_score, hd95  # noqa: E402
from experiments._common import set_seed, build_loader, LOSS_FLAGS  # noqa: E402


def evaluate(model, loader, device):
    model.eval()
    d, i, h = [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            gt = (batch["mask"][:, 0].to(device) > 0.5)
            pred = (torch.sigmoid(model(img)["mask"][:, 0]) > 0.5)
            for b in range(img.shape[0]):
                d.append(dice_score(pred[b], gt[b]))
                i.append(iou_score(pred[b], gt[b]))
                v = hd95(pred[b], gt[b])
                if v == v:
                    h.append(v)
    mean = lambda x: round(sum(x) / len(x), 4) if x else None
    return {"dice": mean(d), "iou": mean(i), "hd95": mean(h), "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--no_physics", action="store_true", help="capacity-matched control")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--input_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model = DeepChanVese(steps=args.steps, base=args.base,
                         use_physics=not args.no_physics,
                         input_size=args.input_size).to(device)
    print(f"physics={'OFF (control)' if args.no_physics else 'ON'}  "
          f"params={model.num_parameters():,}  steps={args.steps}")

    loss_fn = CombinedLoss(**LOSS_FLAGS["dice+ce"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    cfg = {"data": args.data, "input_size": args.input_size}
    train_loader = build_loader(args.data, "train", cfg, args.batch_size, True,
                                args.input_size, False, 1,
                                frac=args.train_frac, frac_seed=args.seed)
    try:
        val_loader = build_loader(args.data, "val", cfg, args.batch_size, False,
                                  args.input_size, False, 1)
    except Exception:
        val_loader = train_loader

    best, since = -1.0, 0
    for ep in range(args.epochs):
        model.train()
        last = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)
            opt.zero_grad()
            out = model(img)
            loss, _ = loss_fn(out["mask"], tgt)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = loss.item()
        sched.step()
        vd = evaluate(model, val_loader, device)["dice"]
        print(f"epoch {ep+1}/{args.epochs}  loss={last:.4f}  val_dice={vd:.4f}")
        if vd > best:
            best, since = vd, 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_dice": vd}, os.path.join(args.output_dir, "best.ckpt"))
        else:
            since += 1
            if args.early_stop_patience and since >= args.early_stop_patience:
                print(f"early stop (best={best:.4f})"); break

    # final test eval with best checkpoint
    ck = torch.load(os.path.join(args.output_dir, "best.ckpt"),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    try:
        test_loader = build_loader(args.data, "test", cfg, args.batch_size, False,
                                   args.input_size, False, 1)
        res = evaluate(model, test_loader, device)
        print(f"TEST: dice={res['dice']}  iou={res['iou']}  hd95={res['hd95']}  n={res['n']}")
        json.dump({"prompt_mode": "prompt_free_chanvese",
                   "physics": not args.no_physics,
                   "macro": res},
                  open(os.path.join(args.output_dir, "test_per_organ.json"), "w"), indent=2)
    except Exception as e:
        print(f"(no test split: {e})")


if __name__ == "__main__":
    main()
