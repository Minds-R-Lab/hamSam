"""Segmentation metrics: Dice, IoU, HD95, sensitivity, specificity.

All operate on binary numpy/torch masks (foreground=1). HD95 uses symmetric
boundary distance transforms; returns NaN when either mask is empty.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


def _to_np(x):
    import torch
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def dice_score(pred, gt, eps=1e-6):
    p, g = _to_np(pred) > 0.5, _to_np(gt) > 0.5
    inter = (p & g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))


def iou_score(pred, gt, eps=1e-6):
    p, g = _to_np(pred) > 0.5, _to_np(gt) > 0.5
    inter = (p & g).sum()
    union = (p | g).sum()
    return float((inter + eps) / (union + eps))


def sensitivity(pred, gt, eps=1e-6):
    p, g = _to_np(pred) > 0.5, _to_np(gt) > 0.5
    tp = (p & g).sum()
    return float((tp + eps) / (g.sum() + eps))


def specificity(pred, gt, eps=1e-6):
    p, g = _to_np(pred) > 0.5, _to_np(gt) > 0.5
    tn = (~p & ~g).sum()
    return float((tn + eps) / ((~g).sum() + eps))


def _surface_distances(a, b):
    """Distances from surface of a to surface of b (and store DT of b)."""
    a_surf = a ^ _erode(a)
    dt_b = distance_transform_edt(~b)
    return dt_b[a_surf]


def _erode(m):
    from scipy.ndimage import binary_erosion
    return binary_erosion(m, border_value=0)


def hd95(pred, gt):
    p, g = _to_np(pred) > 0.5, _to_np(gt) > 0.5
    if not p.any() or not g.any():
        return float("nan")
    d_pg = _surface_distances(p, g)
    d_gp = _surface_distances(g, p)
    if d_pg.size == 0 or d_gp.size == 0:
        return float("nan")
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def all_metrics(pred, gt):
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "hd95": hd95(pred, gt),
        "sensitivity": sensitivity(pred, gt),
        "specificity": specificity(pred, gt),
    }
