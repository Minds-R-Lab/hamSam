"""Prompt-free vs box-prompted comparison (PLAN.md ext 2, Phase 3).

Runs the same checkpoint in (a) box mode (GT-derived oracle box) and (b) auto
mode (energy-derived box, no user input), and reports the Dice gap plus the
IoU of the auto box against the GT box.
"""
import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ham_medsam import HamMedSAM        # noqa: E402
from src.metrics import dice_score           # noqa: E402
from data.datasets import box_from_mask      # noqa: E402
from experiments._common import build_loader  # noqa: E402


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / ua) if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="data root or 'synthetic'")
    ap.add_argument("--input_size", type=int, default=1024)
    ap.add_argument("--quantile", type=float, default=0.80)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = ckpt.get("config", {}).get("model", {})
    model = HamMedSAM(sam_checkpoint=mcfg.get("sam_checkpoint"),
                      backend=mcfg.get("backend", "medsam_vitb"),
                      bottleneck=mcfg.get("bottleneck", "deepest"),
                      input_size=args.input_size).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.energy_to_box.quantile = args.quantile

    loader = build_loader(args.data, "test", {}, 4, False, args.input_size, False, 1)
    res = {"box": [], "auto": [], "box_iou": []}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device); gtbox = batch["box"]
            gt = (batch["mask"].to(device) > 0.5)
            model.prompt_free = False
            ob = model(img, box=gtbox.to(device))
            model.prompt_free = True
            oa = model(img, box=None)
            pb = torch.sigmoid(ob["mask"]) > 0.5
            pa = torch.sigmoid(oa["mask"]) > 0.5
            for b in range(img.shape[0]):
                res["box"].append(dice_score(pb[b], gt[b]))
                res["auto"].append(dice_score(pa[b], gt[b]))
                res["box_iou"].append(box_iou(gtbox[b].tolist(), oa["box"][b].tolist()))
    summary = {k: round(sum(v) / len(v), 4) for k, v in res.items()}
    summary["dice_drop_pp"] = round((summary["box"] - summary["auto"]) * 100, 2)
    print(f"box dice={summary['box']:.4f}  auto dice={summary['auto']:.4f}  "
          f"drop={summary['dice_drop_pp']}pp  auto/GT box IoU={summary['box_iou']:.3f}")
    json.dump(summary, open(os.path.join(args.output_dir, "prompt_free.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
