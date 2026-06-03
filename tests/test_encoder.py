import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ham_encoder import HamEncoder


def test_output_resolution_matches_sam():
    enc = HamEncoder(bottleneck='deepest', input_size=256)
    out = enc(torch.randn(1, 3, 256, 256))
    assert out['feat'].shape == (1, 256, 16, 16)   # 256/16; at 1024 -> 64
    assert out['p'].shape == (1, 256, 16, 16)
    assert out['H_map'].shape == (1, 1, 16, 16)
    assert (out['H_map'] >= 0).all()


def test_placements():
    for mode, has_p in [('deepest', True), ('all', True), ('none', False)]:
        out = HamEncoder(bottleneck=mode, input_size=128)(torch.randn(1, 3, 128, 128))
        assert (out['p'] is not None) == has_p
        assert out['feat'].shape == (1, 256, 8, 8)


def test_encoder_backward():
    enc = HamEncoder(bottleneck='deepest', input_size=128)
    x = torch.randn(1, 3, 128, 128, requires_grad=True)
    out = enc(x)
    (out['feat'].mean() + out['p'].mean()).backward()
    assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"): v(); print("PASS", k)
