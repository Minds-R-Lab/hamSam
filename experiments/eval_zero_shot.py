"""Zero-shot cross-modality evaluation (PLAN.md ext 6).

Loads a Ham-MedSAM checkpoint trained on abdomen CT and evaluates -- without
fine-tuning -- on ISIC 2018 (dermoscopy), TN3K (thyroid US), ACDC (cardiac MRI).
Reports Dice / IoU / HD95 / sensitivity / specificity per dataset, plus the gap
to HamSeg's fully supervised numbers (HamVision Tables 2-3).

prompt_mode: 'box' = GT-derived oracle box; 'auto' = energy-derived prompt-free.
"""
import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ham_medsam import HamMedSAM        # noqa: E402
from src.metrics import all_metrics          # noqa: E402
from experiments._common import build_loader  # noqa: E402

HAMSEG_SUPERVISED = {"isic2018": 90.32, "tn3k": 87.39, "acdc": 93.81}


def evaluate(model, loader, multiclass, prompt_mode, device):
    model.eval(); agg = {}
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            box = batch["box"].to(device) if prompt_mode == "box" else None
            if prompt_mode == "auto":
                model.prompt_free = True
            out = model(img, box=box)
            # Branch on the MODEL'\''s output, not the dataset: a binary
            # abdomen-CT model transferred to multi-class ACDC is scored as
            # foreground (whole-structure) Dice.
            if getattr(model, "multiclass", False) and out["mask"].shape[1] > 1:
                pred = out["mask"].argmax(1) > 0
            else:
                pred = torch.sigmoid(out["mask"][:, 0]) > 0.5
            gt = (batch["mask"].to(device) > (0 if multiclass else 0.5))
            if gt.ndim == 4:
                gt = gt[:, 0]
            for b in range(img.shape[0]):
                for k, v in all_metrics(pred[b], gt[b]).items():
                    agg.setdefault(k, []).append(v)
    return {k: float(torch.tensor([x for x in v if x == x]).mean()) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/zero_shot.yaml")
    ap.add_argument("--datasets", nargs="+", default=["isic2018", "tn3k", "acdc"])
    ap.add_argument("--prompt_mode", default="auto", choices=["box", "auto"])
    ap.add_argument("--data", default=None, help="'synthetic' for smoke test")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = ckpt.get("config", {}).get("model", {})
    input_size = cfg.get("input_size", 1024)

    rows = []
    for name in args.datasets:
        dcfg = cfg["datasets"][name]
        mc, nc = dcfg.get("multiclass", False), dcfg.get("num_classes", 1)
        model = HamMedSAM(sam_checkpoint=mcfg.get("sam_checkpoint"),
                          bottleneck=mcfg.get("bottleneck", "deepest"),
                          num_classes=nc, input_size=input_size).to(device)
        try:
            model.load_state_dict(ckpt["model"], strict=False)
        except Exception as e:
            print(f"[warn] partial load for {name}: {e}")
        root = "synthetic" if args.data == "synthetic" else dcfg["root"]
        loader = build_loader(root, "test", {}, 4, False, input_size, mc, nc)
        m = evaluate(model, loader, mc, args.prompt_mode, device)
        m["dice_pct"] = round(m.get("dice", 0) * 100, 2)
        m["hamseg_supervised_pct"] = HAMSEG_SUPERVISED.get(name)
        if m["hamseg_supervised_pct"]:
            m["gap_pp"] = round(m["dice_pct"] - m["hamseg_supervised_pct"], 2)
        rows.append({"dataset": name, **m})
        print(f"{name:10s} dice={m['dice_pct']:.2f}%  hd95={m.get('hd95'):.2f}  "
              f"gap_to_HamSeg={m.get('gap_pp')}pp")
    json.dump(rows, open(os.path.join(args.output_dir, "zero_shot.json"), "w"), indent=2)
    print(f"-> {args.output_dir}/zero_shot.json")


if __name__ == "__main__":
    main()
