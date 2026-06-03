import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prompt_free import EnergyToBox


def test_box_localises_energy_blob():
    H = torch.zeros(1, 1, 32, 32); H[:, :, 8:16, 10:14] = 10.0
    box = EnergyToBox(quantile=0.85, sam_input_size=1024)(H)[0].tolist()
    x0, y0, x1, y1 = box
    # blob centre ~ (col 12, row 12) of 32 -> ~ (384, 384) in 1024 coords
    assert 0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 1024
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert 250 < cx < 520 and 200 < cy < 560, box


def test_empty_energy_fallback_full_box():
    H = torch.zeros(1, 1, 16, 16)
    box = EnergyToBox()(H)[0].tolist()
    assert box == [0, 0, 1024, 1024]


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"): v(); print("PASS", k)
