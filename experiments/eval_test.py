"""Per-organ evaluation on the held-out TEST split.

Groups metrics by the organ name embedded in each sample filename
(<pid>_z<zzz>_org<L>_<name>.npy, produced by data/prepare_data.py) and reports
per-organ Dice / IoU / HD95 plus the macro average across organs. Use this
instead of the single pooled val number for anything reportable.

    python experiments/eval_test.py --checkpoint outputs/ham/seed_42/best.ckpt \
        --data data/processed/flare22 --device cuda
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ham_medsam import HamMedSAM        # noqa: E402
from src.metrics import dice_score, iou_score, hd95  # noqa: E402
from experiments._common import build_loader  # noqa: E402

_ORG = re.compile(r"_org\d+_(.+)$")


def _organ(name):
    m = _ORG.search(name)
    return m.group(1) if m else "all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="dataset root (uses its test/ split)")
    ap.add_argument("--input_size", type=int, default=1024)
    ap.add_argument("--prompt_mode", default="box", choices=["box", "auto"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    mcfg = ckpt.get("config", {}).get("model", {})
    model = HamMedSAM(sam_checkpoint=mcfg.get("sam_checkpoint"),
                      backend=mcfg.get("backend", "medsam_vitb"),
                      bottleneck=mcfg.get("bottleneck", "deepest"),
                      input_size=args.input_size).to(dev)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    loader = build_loader(args.data, "test", {}, args.batch_size, False,
                          args.input_size, False, 1)
    per = defaultdict(lambda: defaultdict(list))   # organ -> metric -> [values]
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(dev)
            box = batch["box"].to(dev) if args.prompt_mode == "box" else None
            model.prompt_free = (args.prompt_mode == "auto")
            out = model(img, box=box)
            pred = (torch.sigmoid(out["mask"][:, 0]) > 0.5)
            gt = (batch["mask"][:, 0].to(dev) > 0.5)
            for b in range(img.shape[0]):
                org = _organ(batch["name"][b])
                per[org]["dice"].append(dice_score(pred[b], gt[b]))
                per[org]["iou"].append(iou_score(pred[b], gt[b]))
                h = hd95(pred[b], gt[b])
                if h == h:  # not NaN
                    per[org]["hd95"].append(h)

    rows, macro = [], defaultdict(list)
    for org in sorted(per):
        r = {"organ": org, "n": len(per[org]["dice"])}
        for m in ("dice", "iou", "hd95"):
            vals = per[org][m]
            r[m] = round(sum(vals) / len(vals), 4) if vals else None
            if r[m] is not None:
                macro[m].append(r[m])
        rows.append(r)
        print(f"{org:20s} n={r['n']:5d}  dice={r['dice']}  iou={r['iou']}  hd95={r['hd95']}")
    macro_row = {"organ": "MACRO_AVG",
                 **{m: round(sum(v) / len(v), 4) if v else None for m, v in macro.items()}}
    print(f"{'MACRO_AVG':20s}        dice={macro_row.get('dice')}  "
          f"iou={macro_row.get('iou')}  hd95={macro_row.get('hd95')}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump({"prompt_mode": args.prompt_mode, "per_organ": rows,
                   "macro": macro_row},
                  open(os.path.join(args.output_dir, "test_per_organ.json"), "w"), indent=2)
        print(f"-> {args.output_dir}/test_per_organ.json")


if __name__ == "__main__":
    main()
