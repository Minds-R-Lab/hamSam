"""Hamiltonian image encoder for Ham-MedSAM.

Drop-in replacement for the RVM+ encoder of VM-MedSAM (Li et al., Med. Phys.
2026). Built on the HamiltonianBottleneck primitive (src/hamiltonian.py),
which produces three structured outputs:

    f      -- fused features (ConvNeXt + oscillator gated fusion)
    p      -- momentum (band-pass spatial differentiator) -> boundary loss
    H_map  -- single-channel energy saliency               -> prompt-free box

Input:   (B, 3, 1024, 1024)
Output:  dict(feat=(B, out_dim, 64, 64), p=(B, out_dim, 64, 64) or None,
              H_map=(B, 1, 64, 64) or None)

The 1024 -> 64 downsample ratio matches VM-MedSAM so the SAM prompt
encoder / mask decoder are reused without re-training.

`bottleneck` placement (configurable; the user asked to test several):
    'deepest' -- ConvNeXt stages + a single Hamiltonian bottleneck at 64x64
                 (matches HamVision's single 28x28 bottleneck; cheapest).
    'all'     -- a Hamiltonian bottleneck at every stage (512/256/128/64).
                 Earlier stages discard p/H. value_clamp is widened per stage.
    'none'    -- pure-ConvNeXt encoder (VM-MedSAM-style baseline / ablation A);
                 p and H_map are None.
"""
import math

import torch
import torch.nn as nn

from .hamiltonian import ConvNeXtBlock, HamiltonianBottleneck


def _value_clamp_for_length(L: int) -> float:
    """Resolution-aware output clamp for the oscillator scan (see CHANGELOG)."""
    return max(50.0, 5.0 * math.sqrt(L))


class _Downsample(nn.Module):
    """ConvNeXt-style patch-merging downsample: LayerNorm then 2x2 stride-2 conv."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.norm = nn.GroupNorm(1, c_in, eps=1e-6)  # GN(1)==LayerNorm over channels
        self.reduce = nn.Conv2d(c_in, c_out, kernel_size=2, stride=2)

    def forward(self, x):
        return self.reduce(self.norm(x))


class _Stage(nn.Module):
    """depth x ConvNeXt at c_in, optional Hamiltonian bottleneck, then downsample.

    Returns (downsampled_feature, momentum_or_None, energy_or_None) where the
    momentum/energy come from this stage's bottleneck (only kept when exposed).
    """

    def __init__(self, c_in, c_out, depth, line_len, use_bottleneck,
                 ablation='none', drop_rate=0.1, damping_clamp=5.0):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvNeXtBlock(c_in) for _ in range(depth)])
        self.use_bottleneck = use_bottleneck
        if use_bottleneck:
            self.bottleneck = HamiltonianBottleneck(
                c_in, damping_clamp=damping_clamp, drop_rate=drop_rate,
                ablation=ablation, value_clamp=_value_clamp_for_length(line_len))
        self.down = _Downsample(c_in, c_out)

    def forward(self, x):
        x = self.blocks(x)
        p = h = None
        if self.use_bottleneck:
            x, p, h = self.bottleneck(x)
        return self.down(x), p, h


class HamEncoder(nn.Module):
    """SAM-compatible image encoder built on Hamiltonian bottlenecks.

    Channels follow VM-MedSAM: 32 -> 64 -> 128 -> 256, 7x7 stride-2 stem, a
    stride-2 downsample between stages, deepest stage at 256ch on a 64x64 grid.
    """

    def __init__(self, in_channels=3, embed_dim=32, depths=(2, 2, 2),
                 out_dim=256, bottleneck='deepest', ablation='none',
                 damping_clamp=5.0, drop_rate=0.1, input_size=1024):
        super().__init__()
        assert bottleneck in ('deepest', 'all', 'none')
        self.bottleneck_mode = bottleneck
        c0 = embed_dim
        chans = [c0, c0 * 2, c0 * 4, out_dim]            # 32, 64, 128, 256
        # spatial sizes after stem: 512, then 256, 128, 64
        stem_size = input_size // 2
        sizes = [stem_size, stem_size // 2, stem_size // 4, stem_size // 8]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(1, c0, eps=1e-6),
            nn.GELU(),
        )

        # Three downsampling stages (512->256->128->64).
        stage_uses_bn = bottleneck == 'all'
        self.stages = nn.ModuleList([
            _Stage(chans[i], chans[i + 1], depths[i], sizes[i],
                   use_bottleneck=stage_uses_bn, ablation=ablation,
                   drop_rate=drop_rate, damping_clamp=damping_clamp)
            for i in range(3)
        ])

        # Deepest bottleneck at 64x64. Hamiltonian unless 'none' (baseline).
        if bottleneck == 'none':
            self.head = nn.Sequential(ConvNeXtBlock(out_dim), ConvNeXtBlock(out_dim))
            self.deep_bottleneck = None
        else:
            self.deep_bottleneck = HamiltonianBottleneck(
                out_dim, damping_clamp=damping_clamp, drop_rate=drop_rate,
                ablation=ablation, value_clamp=_value_clamp_for_length(sizes[3]))

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x, _, _ = stage(x)                  # discard early-stage p/H
        if self.deep_bottleneck is None:
            feat = self.head(x)
            return {'feat': feat, 'p': None, 'H_map': None}
        feat, p, H_map = self.deep_bottleneck(x)
        return {'feat': feat, 'p': p, 'H_map': H_map}

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
