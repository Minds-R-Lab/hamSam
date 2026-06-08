"""Aggregate the premise-test matrix: Ham vs capacity-matched variant-A.

Reads runs/<root>/{ham,variantA}_s<seed>/{indist,ood}/test_per_organ.json,
reports macro-Dice mean +/- std over seeds for in-distribution and OOD, the
OOD generalisation drop per encoder, and the Ham - variant-A delta with a
rough significance flag (|delta| vs the pooled seed noise).

This is the experiment that decides whether the Hamiltonian *mechanism* (not
its parameters -- variant-A is matched to ~99% of Ham's param count) buys
anything a ConvNeXt of equal size cannot.
"""
import argparse
import json
import os
import statistics as st


def macro_dice(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))["macro"]["dice"]


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(m, 4), "std": round(s, 4), "n": len(vals), "vals": vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/premise")
    ap.add_argument("--seeds", default="42 43 44")
    args = ap.parse_args()
    seeds = args.seeds.split()

    names = {"ham": "Hamiltonian", "variantA": "Variant-A (conv, matched)"}
    conds = {"indist": "in-dist (ISIC2017)", "ood": "OOD (ISIC2018)"}
    table = {}
    for key in names:
        for c in conds:
            table[(key, c)] = agg([macro_dice(
                os.path.join(args.root, f"{key}_s{s}", c, "test_per_organ.json"))
                for s in seeds])

    print(f"\n{'encoder':28s}{'in-dist Dice':>18s}{'OOD Dice':>18s}{'OOD drop':>12s}")
    print("-" * 76)
    for key, label in names.items():
        idd, ood = table[(key, "indist")], table[(key, "ood")]
        ids = f"{idd['mean']:.4f}+/-{idd['std']:.4f}" if idd else "  --  "
        oos = f"{ood['mean']:.4f}+/-{ood['std']:.4f}" if ood else "  --  "
        drop = (f"{(idd['mean']-ood['mean'])*100:+.2f}pp"
                if idd and ood else "  --  ")
        print(f"{label:28s}{ids:>18s}{oos:>18s}{drop:>12s}")

    print("\nHam - Variant-A (positive = physics helps):")
    summary = {"per_cell": {f"{k}|{c}": table[(k, c)] for k in names for c in conds},
               "deltas": {}}
    for c, clabel in conds.items():
        h, a = table[("ham", c)], table[("variantA", c)]
        if not (h and a):
            print(f"  {clabel:22s}: incomplete"); continue
        d = (h["mean"] - a["mean"]) * 100
        pooled = (h["std"] + a["std"]) / 2 * 100
        flag = "likely real" if abs(d) >= 2 * max(pooled, 0.3) else "within noise"
        print(f"  {clabel:22s}: {d:+.2f}pp   (pooled std ~{pooled:.2f}pp -> {flag})")
        summary["deltas"][c] = {"delta_pp": round(d, 2),
                                "pooled_std_pp": round(pooled, 2), "verdict": flag}

    if all(table[(k, c)] for k in names for c in conds):
        h_drop = (table[("ham", "indist")]["mean"] - table[("ham", "ood")]["mean"]) * 100
        a_drop = (table[("variantA", "indist")]["mean"] - table[("variantA", "ood")]["mean"]) * 100
        print(f"\nOOD robustness: Ham drops {h_drop:+.2f}pp vs control {a_drop:+.2f}pp "
              f"under shift (smaller = more robust).")
        summary["ood_robustness"] = {"ham_drop_pp": round(h_drop, 2),
                                     "control_drop_pp": round(a_drop, 2)}

    out = os.path.join(args.root, "premise_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
