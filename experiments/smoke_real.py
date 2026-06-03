"""Turnkey end-to-end smoke test of EVERY component on the real SAM decoder.

Builds the real segment-anything PromptEncoder + MaskDecoder from scratch (no
checkpoint download needed), then runs forward + backward for every model
configuration and every loss option, on synthetic data. Prints a component
checklist and exits non-zero on any failure. Use this to validate the whole
stack on the H100 before launching real training.

    python experiments/smoke_real.py                # CPU/GPU, input 256
    python experiments/smoke_real.py --input_size 1024 --device cuda
"""
import argparse
import os
import sys
import warnings

import torch

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ham_medsam import HamMedSAM            # noqa: E402
from src.losses import CombinedLoss              # noqa: E402
from src.metrics import all_metrics              # noqa: E402
from experiments._common import LOSS_FLAGS       # noqa: E402


def _check(name, fn, results):
    try:
        detail = fn()
        results.append((name, True, detail))
        print(f"  [PASS] {name}: {detail}")
    except Exception as e:  # noqa: BLE001
        results.append((name, False, repr(e)))
        print(f"  [FAIL] {name}: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", default=None, help="optional MedSAM/SAM ckpt (needs input 1024)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    S = args.input_size
    print(f"== Ham-MedSAM real-SAM smoke test == device={dev} input_size={S} "
          f"cuda={torch.cuda.is_available()}")

    img = torch.randn(2, 3, S, S, device=dev)
    box = torch.tensor([[S * .1, S * .1, S * .8, S * .8],
                        [S * .2, S * .15, S * .9, S * .85]], device=dev)
    gt = torch.zeros(2, 1, S, S, device=dev)
    gt[:, :, S // 5:4 * S // 5, S // 5:3 * S // 5] = 1
    results = []

    # backend sanity
    m0 = HamMedSAM(input_size=S, sam_checkpoint=args.checkpoint).to(dev)
    print(f"  SAM backend kind = '{m0.sam_kind}' (expect 'sam' when segment-anything is installed)")

    model_configs = {
        "box-prompted (deepest)": dict(),
        "prompt-free (energy box)": dict(prompt_free=True),
        "PSSP decoder adapter": dict(use_pssp_decoder=True),
        "multi-class head (4)": dict(multiclass_head=True, num_classes=4),
        "bottleneck=all": dict(bottleneck="all"),
        "bottleneck=none (baseline)": dict(bottleneck="none"),
    }
    for name, kw in model_configs.items():
        def fn(kw=kw):
            m = HamMedSAM(input_size=S, sam_checkpoint=args.checkpoint, **kw).to(dev)
            pf = kw.get("prompt_free") or kw.get("multiclass_head")
            out = m(img, box=None if pf else box)
            out["mask"].float().mean().backward()
            assert out["mask"].shape[-2:] == (S, S)
            g = [p for p in m.parameters() if p.requires_grad and p.grad is not None]
            assert g, "no gradients"
            return f"mask={tuple(out['mask'].shape)} kind={m.sam_kind} grads={len(g)}"
        _check(name, fn, results)

    for loss_name, flags in LOSS_FLAGS.items():
        def fn(flags=flags):
            m = HamMedSAM(input_size=S, sam_checkpoint=args.checkpoint).to(dev)
            lf = CombinedLoss(**flags, momentum_channels=256).to(dev)
            out = m(img, box=box)
            loss, parts = lf(out["mask"], gt, p=out["p"])
            loss.backward()
            return "+".join(k for k in parts if k != "total") + f" total={loss.item():.3f}"
        _check(f"loss: {loss_name}", fn, results)

    # prompt-free metrics path
    def fn_metrics():
        m = HamMedSAM(input_size=S, prompt_free=True, sam_checkpoint=args.checkpoint).to(dev)
        out = m(img)
        pred = torch.sigmoid(out["mask"][:, 0]) > 0.5
        mets = all_metrics(pred[0], gt[0, 0])
        return f"box={[round(v,1) for v in out['box'][0].tolist()]} dice={mets['dice']:.3f}"
    _check("prompt-free metrics", fn_metrics, results)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n== {len(results) - n_fail}/{len(results)} checks passed ==")
    if n_fail:
        sys.exit(1)
    print("ALL COMPONENTS OK on the real SAM decoder path.")


if __name__ == "__main__":
    main()
