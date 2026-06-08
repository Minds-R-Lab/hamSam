"""Aggregate the data-efficiency sweep: Ham vs capacity-matched variant-A.

Reads runs/<root>/{ham,variantA}_f<frac>_s<seed>/test_per_organ.json and prints
macro-Dice mean +/- std over seeds at each label fraction, plus the Ham - control
gap. The hypothesis: a physics inductive bias pays off most when labels are
scarce, so the gap should grow as the fraction shrinks. Differences below ~2x
the pooled seed std are flagged as noise.
"""
import argparse
import json
import os
import statistics as st


def macro_dice(p):
    return json.load(open(p))["macro"]["dice"] if os.path.exists(p) else None


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": sum(vals) / len(vals),
            "std": st.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/dataeff")
    ap.add_argument("--fracs", default="0.05 0.1 0.25 0.5 1.0")
    ap.add_argument("--seeds", default="42 43 44")
    args = ap.parse_args()
    fracs = args.fracs.split()
    seeds = args.seeds.split()

    print(f"\n{'frac':>6s}{'Ham Dice':>18s}{'Variant-A Dice':>20s}{'Ham-A':>12s}{'verdict':>14s}")
    print("-" * 70)
    summary = {}
    for f in fracs:
        cell = {}
        for name in ("ham", "variantA"):
            cell[name] = agg([macro_dice(
                os.path.join(args.root, f"{name}_f{f}_s{s}", "test_per_organ.json"))
                for s in seeds])
        h, a = cell["ham"], cell["variantA"]
        if not (h and a):
            print(f"{f:>6s}   incomplete"); continue
        d = (h["mean"] - a["mean"]) * 100
        pooled = (h["std"] + a["std"]) / 2 * 100
        verdict = "real" if abs(d) >= 2 * max(pooled, 0.3) else "noise"
        print(f"{f:>6s}   {h['mean']:.4f}+/-{h['std']:.4f}   "
              f"{a['mean']:.4f}+/-{a['std']:.4f}   {d:+6.2f}pp{verdict:>10s}")
        summary[f] = {"ham": round(h["mean"], 4), "ham_std": round(h["std"], 4),
                      "variantA": round(a["mean"], 4), "variantA_std": round(a["std"], 4),
                      "delta_pp": round(d, 2), "pooled_std_pp": round(pooled, 2),
                      "verdict": verdict}

    # does the gap grow as labels shrink?
    if len(summary) >= 2:
        ks = sorted(summary, key=float)
        lo, hi = summary[ks[0]]["delta_pp"], summary[ks[-1]]["delta_pp"]
        print(f"\nHam-A gap: {hi:+.2f}pp at {ks[-1]} labels -> {lo:+.2f}pp at {ks[0]} labels "
              f"({'GROWS as labels shrink (prior helps when scarce)' if lo > hi else 'does not grow'}).")

    out = os.path.join(args.root, "dataeff_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
