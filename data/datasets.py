"""Datasets and box-prompt generation for Ham-MedSAM.

MedSegDataset    -- generic 2D dataset: <root>/<split>/images/*.{png,npy} with
                    matching masks in <root>/<split>/masks/. Applies SAM
                    preprocessing and derives a box prompt from the GT mask
                    (with random perturbation during training, per VM-MedSAM).
SyntheticSegDataset -- random shapes; lets the pipeline run with no data
                    download (CI smoke tests).
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .sam_preprocess import preprocess_image, preprocess_mask


def box_from_mask(mask, perturb=0, size=1024):
    """Axis-aligned box of the foreground, optionally perturbed by +/-perturb px."""
    ys, xs = torch.where(mask > 0.5)
    if len(xs) == 0:
        return torch.tensor([0, 0, size, size], dtype=torch.float32)
    x0, x1 = xs.min().item(), xs.max().item()
    y0, y1 = ys.min().item(), ys.max().item()
    if perturb > 0:
        d = lambda: np.random.randint(-perturb, perturb + 1)
        x0, y0, x1, y1 = x0 + d(), y0 + d(), x1 + d(), y1 + d()
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(size - 1, x1), min(size - 1, y1)
    return torch.tensor([x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)], dtype=torch.float32)


class MedSegDataset(Dataset):
    def __init__(self, root, split='train', size=1024, normalize='none',
                 box_perturb=20, multiclass=False, num_classes=1):
        self.size, self.normalize, self.multiclass = size, normalize, multiclass
        self.num_classes = num_classes
        self.box_perturb = box_perturb if split == 'train' else 0
        idir = os.path.join(root, split, 'images')
        self.images = sorted(sum([glob.glob(os.path.join(idir, e))
                                  for e in ('*.png', '*.npy', '*.jpg')], []))
        self.mask_dir = os.path.join(root, split, 'masks')
        if not self.images:
            raise FileNotFoundError(f"no images under {idir}")

    def __len__(self):
        return len(self.images)

    def _load(self, path):
        if path.endswith('.npy'):
            return np.load(path)
        from PIL import Image
        return np.array(Image.open(path))

    def __getitem__(self, i):
        ipath = self.images[i]
        base = os.path.splitext(os.path.basename(ipath))[0]
        mpath = None
        for e in ('.png', '.npy', '.jpg'):
            cand = os.path.join(self.mask_dir, base + e)
            if os.path.exists(cand):
                mpath = cand; break
        img, sp = preprocess_image(self._load(ipath), self.size, self.normalize)
        mask = preprocess_mask(self._load(mpath), self.size, sp) if mpath else \
            torch.zeros(self.size, self.size)
        fg = (mask > 0.5).float()
        box = box_from_mask(fg, self.box_perturb, self.size)
        if self.multiclass:
            target = mask.long()
        else:
            target = fg.unsqueeze(0)
        return {'image': img, 'mask': target, 'box': box}


class SyntheticSegDataset(Dataset):
    """Random rectangles/disks -- for smoke tests only."""

    def __init__(self, n=16, size=256, multiclass=False, num_classes=1, seed=0):
        self.n, self.size = n, size
        self.multiclass, self.num_classes = multiclass, num_classes
        self.rng = np.random.RandomState(seed)
        self.items = [self._make() for _ in range(n)]

    def _make(self):
        s = self.size
        img = self.rng.rand(3, s, s).astype(np.float32) * 0.3
        mask = np.zeros((s, s), np.int64)
        cls = self.rng.randint(1, max(2, self.num_classes)) if self.multiclass else 1
        cx, cy = self.rng.randint(s // 4, 3 * s // 4, 2)
        r = self.rng.randint(s // 8, s // 4)
        yy, xx = np.ogrid[:s, :s]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        mask[disk] = cls
        img[:, disk] += 0.6
        return img, mask

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img, mask = self.items[i]
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        fg = (mask > 0).float()
        box = box_from_mask(fg, 0, self.size)
        target = mask if self.multiclass else fg.unsqueeze(0)
        return {'image': img, 'mask': target, 'box': box}
