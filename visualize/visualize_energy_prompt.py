"""Figure: how the energy map H drives the auto bounding box (PLAN.md ext 2).

Per row: input image | energy map H_map | threshold mask | bbox overlay.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ham_medsam import HamMedSAM        # noqa: E402
from data.sam_preprocess import preprocess_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--quantile", type=float, default=0.80)
    ap.add_argument("--input_size", type=int, default=1024)
    ap.add_argument("--output", default="energy_prompt_figure.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mcfg = ckpt.get("config", {}).get("model", {})
    model = HamMedSAM(bottleneck=mcfg.get("bottleneck", "deepest"),
                      input_size=args.input_size)
    model.load_state_dict(ckpt["model"], strict=False); model.eval()

    n = len(args.images)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    axes = np.atleast_2d(axes)
    for r, path in enumerate(args.images):
        img, _ = preprocess_image(np.array(Image.open(path)), args.input_size)
        with torch.no_grad():
            enc = model.image_encoder(img.unsqueeze(0))
            H = enc["H_map"]
            box = model.energy_to_box(H)[0].tolist()
        Hn = H[0, 0].numpy()
        thr = np.quantile(Hn, args.quantile)
        axes[r, 0].imshow(img.permute(1, 2, 0).numpy()); axes[r, 0].set_title("input")
        axes[r, 1].imshow(Hn, cmap="magma"); axes[r, 1].set_title("energy H_map")
        axes[r, 2].imshow(Hn >= thr, cmap="gray"); axes[r, 2].set_title(f"top-{int((1-args.quantile)*100)}% mask")
        axes[r, 3].imshow(img.permute(1, 2, 0).numpy())
        x0, y0, x1, y1 = box
        axes[r, 3].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="lime", lw=2))
        axes[r, 3].set_title("auto bbox")
        for c in range(4):
            axes[r, c].axis("off")
    plt.tight_layout(); plt.savefig(args.output, dpi=120)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
