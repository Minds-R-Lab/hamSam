"""SAM/MedSAM image preprocessing: resize longest side to 1024, pad to square,
replicate grayscale to 3 channels, optional ImageNet normalisation.

For medical encoders trained from scratch, dataset statistics often beat
ImageNet stats (data/README.md); `normalize='none'` keeps [0,1] scaling.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(img, size=1024, normalize='none'):
    """img: (H,W) or (H,W,3) or (3,H,W) numpy/tensor in [0,255] or [0,1].
    Returns (3,size,size) float tensor and the (scale, padx, pady) used so masks
    / boxes can be mapped back.
    """
    t = torch.as_tensor(np.asarray(img)).float()
    if t.ndim == 2:
        t = t.unsqueeze(0).repeat(3, 1, 1)
    elif t.ndim == 3 and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
    if t.max() > 1.5:
        t = t / 255.0
    _, h, w = t.shape
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    t = F.interpolate(t.unsqueeze(0), size=(nh, nw), mode='bilinear',
                      align_corners=False)[0]
    pad = torch.zeros(3, size, size)
    pady, padx = (size - nh) // 2, (size - nw) // 2
    pad[:, pady:pady + nh, padx:padx + nw] = t
    if normalize == 'imagenet':
        pad = (pad - IMAGENET_MEAN) / IMAGENET_STD
    return pad, (scale, padx, pady)


def preprocess_mask(mask, size=1024, scale_pad=None):
    """Resize/pad a label mask the same way as its image (nearest)."""
    t = torch.as_tensor(np.asarray(mask)).float()
    if t.ndim == 3:
        t = t[..., 0]
    h, w = t.shape
    if scale_pad is None:
        scale = size / max(h, w); padx = pady = None
    else:
        scale, padx, pady = scale_pad
    nh, nw = int(round(h * scale)), int(round(w * scale))
    t = F.interpolate(t[None, None], size=(nh, nw), mode='nearest')[0, 0]
    out = torch.zeros(size, size)
    if padx is None:
        pady, padx = (size - nh) // 2, (size - nw) // 2
    out[pady:pady + nh, padx:padx + nw] = t
    return out
