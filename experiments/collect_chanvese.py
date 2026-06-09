"""Aggregate deep Chan-Vese: physics-on vs capacity-matched control (physics-off).

Reads runs/<root>/{phys,ctrl}_s<seed>/test_per_organ.json and reports macro
Dice / IoU / HD95 mean +/- std over seeds, plus the physics - control delta.
Context: Ham's prompt-free ceiling on ISIC2017 was ~0.82 Dice.
"""
import argparse
import json
import os
import statistics as st

HAM_PROMPTFREE_REF = 0.819  # learned-prompt-head test Dice, for context


def load(p, k):
    return json.load(open(p))["macro"].get(k) if os.path.exists(p) else None


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": sum(vals) / len(vals),
            "std": st.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/cv")
    ap.add_argument("--seeds", default="42 43 44")
    args = ap.parse_args()
    seeds = args.seeds.split()

    print(f"\n{'metric':>8s}{'Physics-ON':>20s}{'Control(off)':>20s}{'phys-ctrl':>12s}")
    print("-" * 60)
    summary = {}
    for metric in ("dice", "iou", "hd95"):
        p = agg([load(os.path.join(args.root, f"phys_s{s}", "test_per_organ.json"), metric) for s in seeds])
        c = agg([load(os.path.join(args.root, f"ctrl_s{s}", "test_per_organ.json"), metric) for s in seeds])
        if not (p and c):
            print(f"{metric:>8s}   incomplete"); continue
        d = p["mean"] - c["mean"]
        pooled = (p["std"] + c["std"]) / 2
        if metric == "hd95":
            verdict = "better" if d < 0 else "worse"          # lower HD95 is better
            note = f"{verdict}" if abs(d) >= 2 * max(pooled, 0.3) else "noise"
        else:
            note = "real" if abs(d) >= 2 * max(pooled, 0.003) else "noise"
        print(f"{metric:>8s}{p['mean']:>11.4f}+/-{p['std']:.4f}"
              f"{c['mean']:>11.4f}+/-{c['std']:.4f}{d:>+9.4f} {note}")
        summary[metric] = {"physics": round(p["mean"], 4), "physics_std": round(p["std"], 4),
                           "control": round(c["mean"], 4), "control_std": round(c["std"], 4),
                           "delta": round(d, 4), "note": note}

    if "dice" in summary:
        pd = summary["dice"]["physics"]
        print(f"\nContext: Ham prompt-free (learned head) was ~{HAM_PROMPTFREE_REF} Dice.")
        print(f"Deep Chan-Vese (physics, prompt-free) = {pd:.4f}  "
              f"-> {'beats' if pd > HAM_PROMPTFREE_REF else 'below'} Ham prompt-free.")
        print("Decision rule: physics earns its keep only if it beats the control "
              "(same params, dynamics off) by > ~2x seed std on Dice or HD95.")
    out = os.path.join(args.root, "chanvese_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
