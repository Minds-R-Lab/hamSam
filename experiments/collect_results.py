"""Aggregate per-organ eval JSONs into one comparison table.

Scans a directory tree for test_per_organ.json files (written by eval_test.py),
infers a run label from each file's parent path, and prints + writes a table of
macro Dice / IoU / HD95 per run (and optionally a per-organ Dice matrix).

    python experiments/collect_results.py --root outputs --out outputs/summary
"""
import argparse
import csv
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs", help="dir tree to scan")
    ap.add_argument("--out", default=None, help="output dir for summary.csv/.md")
    ap.add_argument("--per_organ", action="store_true", help="also dump per-organ Dice matrix")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", "test_per_organ.json"),
                             recursive=True))
    if not files:
        print(f"no test_per_organ.json found under {args.root}"); return

    runs = []
    organs = set()
    for fp in files:
        data = json.load(open(fp))
        label = os.path.relpath(os.path.dirname(fp), args.root).replace(os.sep, "/")
        m = data.get("macro", {})
        row = {"run": label, "prompt_mode": data.get("prompt_mode", "?"),
               "macro_dice": m.get("dice"), "macro_iou": m.get("iou"),
               "macro_hd95": m.get("hd95")}
        row["_per"] = {r["organ"]: r.get("dice") for r in data.get("per_organ", [])}
        organs.update(row["_per"])
        runs.append(row)

    runs.sort(key=lambda r: (r["macro_dice"] is None, -(r["macro_dice"] or 0)))
    print(f"{'run':40s} {'prompt':6s} {'Dice':>7s} {'IoU':>7s} {'HD95':>7s}")
    for r in runs:
        print(f"{r['run'][:40]:40s} {r['prompt_mode']:6s} "
              f"{r['macro_dice']!s:>7} {r['macro_iou']!s:>7} {r['macro_hd95']!s:>7}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "prompt_mode", "macro_dice", "macro_iou", "macro_hd95"])
            for r in runs:
                w.writerow([r["run"], r["prompt_mode"], r["macro_dice"],
                            r["macro_iou"], r["macro_hd95"]])
        with open(os.path.join(args.out, "summary.md"), "w") as fh:
            fh.write("| run | prompt | Dice | IoU | HD95 |\n|---|---|---|---|---|\n")
            for r in runs:
                fh.write(f"| {r['run']} | {r['prompt_mode']} | {r['macro_dice']} "
                         f"| {r['macro_iou']} | {r['macro_hd95']} |\n")
        if args.per_organ:
            cols = sorted(organs)
            with open(os.path.join(args.out, "per_organ_dice.csv"), "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["run"] + cols)
                for r in runs:
                    w.writerow([r["run"]] + [r["_per"].get(c) for c in cols])
        print(f"-> {args.out}/summary.csv, summary.md"
              + (", per_organ_dice.csv" if args.per_organ else ""))


if __name__ == "__main__":
    main()
