"""Shared helpers for the training/eval entry points."""
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import (MedSegDataset, SyntheticSegDataset,  # noqa: E402
                            MultiRootSegDataset)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(data_root, split, cfg, batch_size, shuffle, input_size,
                 multiclass, num_classes):
    if data_root == "synthetic":
        ds = SyntheticSegDataset(n=batch_size * 2, size=input_size,
                                 multiclass=multiclass, num_classes=num_classes,
                                 seed=0 if split == "train" else 1)
    elif isinstance(data_root, (list, dict)):     # VM-MedSAM joint training
        ds = MultiRootSegDataset(data_root, split=split, size=input_size,
                                 normalize=cfg.get("normalize", "none"),
                                 box_perturb=cfg.get("box_perturb", 20))
    else:
        ds = MedSegDataset(data_root, split=split, size=input_size,
                           normalize=cfg.get("normalize", "none"),
                           box_perturb=cfg.get("box_perturb", 20),
                           multiclass=multiclass, num_classes=num_classes)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


LOSS_FLAGS = {
    "dice+ce": dict(use_hausdorff=False, use_momentum=False),
    "dice+ce+hausdorff": dict(use_hausdorff=True, use_momentum=False),
    "dice+ce+momentum": dict(use_hausdorff=False, use_momentum=True),
    "dice+ce+hausdorff+momentum": dict(use_hausdorff=True, use_momentum=True),
}


def amp_autocast(device_type, enabled):
    """torch.amp.autocast across versions (avoids the deprecated cuda.amp API)."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)  # pragma: no cover


def make_grad_scaler(device_type, enabled):
    """torch.amp.GradScaler across versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type, enabled=enabled)
        except TypeError:  # pragma: no cover
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover
