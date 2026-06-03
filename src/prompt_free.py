"""Energy-driven prompt-free pipeline for Ham-MedSAM (PLAN.md s3, extension 2).

Turns the Hamiltonian energy map H_map into a bounding box automatically, so
inference is a single forward pass with no user-drawn box.

Algorithm:
    1. Per-image energy map H_map (B,1,h,w).
    2. Threshold at the top-(1-q) quantile (per-image).
    3. Largest connected component of the thresholded map.
    4. Axis-aligned bbox, rescaled to SAM input coords (default 1024).
Empty-energy fallback: the full-image box.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label


class EnergyToBox(nn.Module):
    def __init__(self, quantile: float = 0.80, sam_input_size: int = 1024,
                 smooth: bool = True):
        super().__init__()
        self.quantile = quantile
        self.sam_input_size = sam_input_size
        self.smooth = smooth

    @torch.no_grad()
    def forward(self, H_map: torch.Tensor) -> torch.Tensor:
        assert H_map is not None, "EnergyToBox needs H_map; encoder ran with bottleneck='none'?"
        B, _, h, w = H_map.shape
        S = self.sam_input_size
        boxes = []
        hm = H_map.detach().float().cpu()
        if self.smooth:
            hm = torch.nn.functional.avg_pool2d(hm, 3, stride=1, padding=1)
        for b in range(B):
            m = hm[b, 0].numpy()
            thr = np.quantile(m, self.quantile)
            binary = m >= thr
            lbl, n = label(binary)
            if n == 0:
                boxes.append([0, 0, S, S]); continue
            sizes = [(lbl == i).sum() for i in range(1, n + 1)]
            keep = 1 + int(np.argmax(sizes))
            ys, xs = np.where(lbl == keep)
            sx, sy = S / w, S / h
            x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
            y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
            boxes.append([x0, y0, x1, y1])
        return torch.tensor(boxes, dtype=torch.float32, device=H_map.device)
