"""Real segment-anything decoder path. Skipped if the package is unavailable."""
import os, sys, warnings
import pytest
import torch
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sam_utils import ensure_sam_importable
ensure_sam_importable()
pytest.importorskip("segment_anything.modeling")
from src.ham_medsam import HamMedSAM  # noqa: E402


def test_real_sam_forward_backward_box():
    m = HamMedSAM(input_size=128)            # no checkpoint -> real SAM from scratch
    assert m.sam_kind == "sam"
    img = torch.randn(2, 3, 128, 128)
    box = torch.tensor([[10., 10, 100, 100], [20, 30, 110, 100]])
    out = m(img, box=box)
    assert out["mask"].shape == (2, 1, 128, 128)
    out["mask"].mean().backward()
    assert any(p.grad is not None for p in m.image_encoder.parameters())


def test_real_sam_prompt_free():
    m = HamMedSAM(input_size=128, prompt_free=True)
    out = m(torch.randn(1, 3, 128, 128))
    assert out["mask"].shape == (1, 1, 128, 128) and out["box"] is not None


def test_prompt_encoder_frozen_by_default():
    m = HamMedSAM(input_size=128)
    assert all(not p.requires_grad for p in m.prompt_encoder.parameters())


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"): v(); print("PASS", k)
