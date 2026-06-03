"""Unit tests for the Hamiltonian primitives. Guards the scan-direction fix."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hamiltonian import (  # noqa: E402
    HamiltonianScanLine, HamiltonianSS2D, HamiltonianBottleneck,
)


def test_reshape_roundtrip_is_identity_all_directions():
    ss = HamiltonianSS2D(d_model=8)
    x = torch.randn(2, 8, 16, 12)
    for d in range(4):
        lines, h, w = ss._to_lines(x, d)
        back = ss._to_2d(lines, x.shape[0], h, w, d)
        assert (back - x).abs().max().item() < 1e-5, f"dir {d} round-trip broken"


def test_four_directions_are_genuinely_reversed():
    ss = HamiltonianSS2D(d_model=4)
    B, C, H, W = 1, 4, 3, 5
    x = torch.zeros(B, C, H, W)
    for h in range(H):
        for w in range(W):
            x[0, :, h, w] = 100 * w + h
    seqs = {d: ss._to_lines(x, d)[0][0, :, 0].tolist() for d in range(4)}
    assert seqs[0] == [0, 100, 200, 300, 400]
    assert seqs[1] == [400, 300, 200, 100, 0], f"d=1 not reversed: {seqs[1]}"
    assert seqs[2] == [0, 1, 2]
    assert seqs[3] == [2, 1, 0], f"d=3 not reversed: {seqs[3]}"


def test_scanline_finite_and_bounded_across_resolutions():
    for L, clamp in [(28, 50.0), (64, 50.0), (128, 90.0), (512, 200.0)]:
        sl = HamiltonianScanLine(d_model=8, value_clamp=clamp)
        q, p, e = sl(torch.randn(4, L, 8))
        assert torch.isfinite(e).all(), f"non-finite energy at L={L}"
        assert q.abs().max() <= clamp + 1e-3
        assert (e >= 0).all()


def test_bottleneck_output_shapes_and_ablations():
    x = torch.randn(2, 16, 8, 8)
    out, mom, en = HamiltonianBottleneck(16, ablation='none')(x)
    assert out.shape == (2, 16, 8, 8) and mom.shape == (2, 16, 8, 8)
    assert en.shape == (2, 1, 8, 8) and (en >= 0).all()
    out, mom, en = HamiltonianBottleneck(16, ablation='A')(x)
    assert out.shape == (2, 16, 8, 8) and mom is None and en is None
    out, mom, en = HamiltonianBottleneck(16, ablation='B')(x)
    assert out.shape == (2, 16, 8, 8) and mom.shape == (2, 16, 8, 8)


def test_bottleneck_is_differentiable():
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    out, mom, en = HamiltonianBottleneck(16)(x)
    (out.sum() + mom.sum() + en.sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} hamiltonian tests passed.")
