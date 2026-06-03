import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import MultiRootSegDataset, box_from_mask
import torch


def _make_ds(root, n=3):
    for sub in ('images', 'masks'):
        os.makedirs(os.path.join(root, 'train', sub), exist_ok=True)
    for i in range(n):
        np.save(os.path.join(root, 'train', 'images', f'{i}.npy'),
                (np.random.rand(32, 32) * 255).astype(np.uint8))
        m = np.zeros((32, 32), np.uint8); m[8:20, 8:20] = 1
        np.save(os.path.join(root, 'train', 'masks', f'{i}.npy'), m)


def test_multiroot_concatenates_and_yields_boxes():
    with tempfile.TemporaryDirectory() as d:
        _make_ds(os.path.join(d, 'a'), 3); _make_ds(os.path.join(d, 'b'), 2)
        ds = MultiRootSegDataset({'a': os.path.join(d, 'a'), 'b': os.path.join(d, 'b')},
                                 split='train', size=64)
        assert len(ds) == 5
        item = ds[0]
        assert item['image'].shape == (3, 64, 64)
        assert item['mask'].shape == (1, 64, 64)
        assert item['box'].shape == (4,) and item['box'][2] > item['box'][0]


def test_box_from_empty_mask_is_full_image():
    b = box_from_mask(torch.zeros(64, 64), 0, 64).tolist()
    assert b == [0, 0, 64, 64]


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"): v(); print("PASS", k)
