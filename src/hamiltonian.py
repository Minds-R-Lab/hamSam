"""Hamiltonian primitives for Ham-MedSAM.

Originally copied from the HamVision/HamSeg release (Li et al., *HamVision:
Hamiltonian Dynamics as Inductive Bias for Medical Image Analysis*). Classes:

    ConvNeXtBlock          -- standard ConvNeXt residual block
    HamiltonianScanLine    -- 1-D damped harmonic oscillator parallel scan
    HamiltonianSS2D        -- 4-direction 2-D scan that returns (q, p, energy)
    HamiltonianBottleneck  -- ConvNeXt | oscillator fusion with sigmoid gate,
                              energy-channel SE attention; emits (f, p, H_map).

------------------------------------------------------------------------------
MODIFICATIONS FROM THE UPSTREAM HamVision SOURCE
------------------------------------------------------------------------------
1. BUG FIX (correctness): HamiltonianSS2D._to_lines reversed the *channel*
   axis instead of the *spatial* axis for the right-to-left (d=1) and
   bottom-to-top (d=3) sweeps, so two of the four advertised scan directions
   were not actually reversed. The paper ("four-direction parallel scan
   (left, right, up, down)") specifies four genuine spatial directions. Fixed
   by flipping dim=2 (the spatial axis of the (B*L, C, L) tensor). A
   round-trip + reversal unit test guards this (tests/test_hamiltonian.py).

2. RESOLUTION-AWARE CLAMPS: the upstream value clamp (+/-50) and decay clamp
   ([-5, 0]) were calibrated for a 28x28 bottleneck (L=28). Ham-MedSAM may
   place the bottleneck at coarser stages where L reaches 512; the scan's
   random-walk magnitude grows ~sqrt(L) (empirically ~44 at L=512, i.e. the
   +/-50 clamp begins binding). value_clamp is now a constructor argument so
   the encoder can widen it per stage; the default preserves upstream for
   L<=~64.

3. torch.cuda.amp.autocast(enabled=False) -> version-robust _autocast_disabled().

Everything else (gate bias +2.0 warm start, SE energy attention Eq.(1),
dropout 0.1, ablation variants A/B) is unchanged and matches HamVision 3.1/3.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _autocast_disabled():
    """Context manager that disables autocast, across torch versions/devices."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.amp.autocast(device_type, enabled=False)
    return torch.cuda.amp.autocast(enabled=False)  # pragma: no cover


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class HamiltonianScanLine(nn.Module):
    """Damped harmonic oscillator as an SSM via parallel scan over one axis.

    Scans rows/columns (length L) independently; reformulated as a cumulative
    sum in a rescaled frame so it runs without a Python loop. The per-step
    transition exp(-nu*dt) has magnitude < 1 (nu, dt > 0 via softplus), giving
    a BIBO-stable scan without normalisation.
    """

    def __init__(self, d_model, damping_clamp=5.0, value_clamp=50.0,
                 decay_clamp=5.0):
        super().__init__()
        self.damping_clamp = damping_clamp
        self.value_clamp = value_clamp
        self.decay_clamp = decay_clamp
        self.log_k = nn.Parameter(torch.linspace(-1, 3, d_model))
        self.nu_scale = nn.Parameter(torch.ones(d_model))
        self.nu_bias = nn.Parameter(torch.ones(d_model) * 1.0)
        self.dt_scale = nn.Parameter(torch.ones(d_model) * 0.3)
        self.dt_bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        B, L, D = x.shape
        x_f = x.float()
        omega = torch.exp(self.log_k.float() / 2.0)

        nu = torch.clamp(F.softplus(x_f * self.nu_scale + self.nu_bias) + 1e-6,
                         max=self.damping_clamp)
        dt = F.softplus(x_f * self.dt_scale + self.dt_bias) + 1e-6

        log_decay = -nu * dt
        angle = omega.unsqueeze(0).unsqueeze(0) * dt

        L_re = torch.cumsum(log_decay, dim=1).clamp(-self.decay_clamp, 0)
        L_im = torch.cumsum(angle, dim=1)

        scale = torch.exp(-L_re)
        rot_re = x_f * scale * torch.cos(-L_im)
        rot_im = x_f * scale * torch.sin(-L_im)

        acc_re = torch.cumsum(rot_re, dim=1)
        acc_im = torch.cumsum(rot_im, dim=1)

        unscale = torch.exp(L_re)
        cos_L = torch.cos(L_im)
        sin_L = torch.sin(L_im)

        q = unscale * (cos_L * acc_re - sin_L * acc_im)
        p = unscale * (sin_L * acc_re + cos_L * acc_im)

        q = q.clamp(-self.value_clamp, self.value_clamp)
        p = p.clamp(-self.value_clamp, self.value_clamp)
        energy = 0.5 * (q * q + p * p)
        return q.to(x.dtype), p.to(x.dtype), energy.to(x.dtype)


class HamiltonianSS2D(nn.Module):
    """4-direction Hamiltonian scan on 2D feature maps.

    d=0 rows L->R, d=1 rows R->L, d=2 cols T->B, d=3 cols B->T. Reverse
    directions flip the spatial axis (dim 2 of the (B*L, C, L) tensor) so the
    scan genuinely runs in reverse. (Upstream flipped the channel axis here.)
    """

    def __init__(self, d_model, damping_clamp=5.0, value_clamp=50.0,
                 decay_clamp=5.0):
        super().__init__()
        self.scans = nn.ModuleList([
            HamiltonianScanLine(d_model, damping_clamp, value_clamp, decay_clamp)
            for _ in range(4)
        ])
        self.pos_merge = nn.Linear(d_model * 4, d_model)
        self.mom_merge = nn.Linear(d_model * 4, d_model)

    def _to_lines(self, x, d):
        B, C, H, W = x.shape
        if d == 0:
            return x.permute(0, 2, 1, 3).reshape(B * H, C, W).permute(0, 2, 1), H, W
        elif d == 1:
            return x.permute(0, 2, 1, 3).reshape(B * H, C, W).flip(2).permute(0, 2, 1), H, W
        elif d == 2:
            return x.permute(0, 3, 1, 2).reshape(B * W, C, H).permute(0, 2, 1), H, W
        else:
            return x.permute(0, 3, 1, 2).reshape(B * W, C, H).flip(2).permute(0, 2, 1), H, W

    def _to_2d(self, s, B, H, W, d):
        C = s.shape[2]
        if d == 0:
            return s.permute(0, 2, 1).reshape(B, H, C, W).permute(0, 2, 1, 3)
        elif d == 1:
            return s.flip(1).permute(0, 2, 1).reshape(B, H, C, W).permute(0, 2, 1, 3)
        elif d == 2:
            return s.permute(0, 2, 1).reshape(B, W, C, H).permute(0, 2, 3, 1)
        else:
            return s.flip(1).permute(0, 2, 1).reshape(B, W, C, H).permute(0, 2, 3, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        pos_l, mom_l, eng_l = [], [], []
        for d in range(4):
            lines, h, w = self._to_lines(x, d)
            q, p, e = self.scans[d](lines)
            pos_l.append(self._to_2d(q, B, h, w, d))
            mom_l.append(self._to_2d(p, B, h, w, d))
            eng_l.append(self._to_2d(e, B, h, w, d))
        pos = self.pos_merge(torch.cat(pos_l, 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        mom = self.mom_merge(torch.cat(mom_l, 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        energy = torch.stack(eng_l, 0).mean(0)
        return pos, mom, energy


class HamiltonianBottleneck(nn.Module):
    """Hamiltonian Bottleneck with optional ablation variants.

    ablation='none' : ConvNeXt + SS2D with sigmoid-gated fusion -> (f, p, Hmap).
    ablation='A'    : ConvNeXt-only; returns (f, None, None).
    ablation='B'    : oscillator-only; returns (f, p, Hmap).
    """

    def __init__(self, dim, damping_clamp=5.0, drop_rate=0.1, ablation='none',
                 value_clamp=50.0, decay_clamp=5.0):
        super().__init__()
        self.ablation = ablation
        self.dim = dim

        if ablation == 'A':
            self.conv_only = ConvNeXtBlock(dim)
            self.drop = nn.Dropout2d(drop_rate)
            return

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.ss2d = HamiltonianSS2D(dim, damping_clamp, value_clamp, decay_clamp)
        self.pos_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.drop = nn.Dropout2d(drop_rate)
        self.energy_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

        if ablation == 'B':
            return

        self.conv_block = ConvNeXtBlock(dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[0].bias, 2.0)

    def forward(self, x):
        if self.ablation == 'A':
            out = self.drop(self.conv_only(x))
            return out, None, None

        if self.ablation != 'B':
            conv_out = self.conv_block(x)
        x_n = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        with _autocast_disabled():
            pos, mom, energy_raw = self.ss2d(x_n.float())
            ham_out = self.pos_proj(pos)
            if self.ablation == 'B':
                out = ham_out
            else:
                g = self.gate(torch.cat([conv_out.float(), ham_out], 1))
                out = conv_out.float() * g + ham_out * (1 - g)

        out = self.drop(out.to(x.dtype))
        mom = mom.to(x.dtype)
        energy_f = energy_raw.to(x.dtype)
        ch_weights = self.energy_attn(energy_f).unsqueeze(-1).unsqueeze(-1)
        energy_map = (energy_f * ch_weights).mean(dim=1, keepdim=True)
        return out, mom, energy_map
