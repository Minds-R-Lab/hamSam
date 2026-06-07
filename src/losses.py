"""Loss functions for Ham-MedSAM.

    SoftDiceLoss          -- standard soft Dice.
    HausdorffMaskLoss     -- differentiable distance-transform Hausdorff on the
                             predicted mask (Karimi & Salcudean, IEEE TMI 2020).
                             Reproduces VM-MedSAM's boundary objective so we can
                             ablate it against MomentumBoundaryLoss under
                             identical conditions.
    MomentumBoundaryLoss  -- NOVEL: supervises the encoder momentum |p| against
                             a boundary-proximity map derived from the GT.
    CombinedLoss          -- dice + ce/bce (+ optional hausdorff) (+ optional
                             momentum), with configurable weights.

------------------------------------------------------------------------------
NOTE ON MomentumBoundaryLoss TARGET (resolves an inconsistency in PLAN.md s3,
extension 3):  the plan's prose says the loss should "enforce high momentum
exactly where the boundary is", but the formula it writes, ||softplus(|p|) -
DT(boundary)||_1, uses the raw distance transform, which is ZERO on the
boundary and grows away from it -- i.e. it would push |p| LOW at the boundary,
the opposite of the stated intent. We implement the stated *intent*: the
default target is a boundary-PROXIMITY map (1 at the boundary, decaying away).
The literal distance-transform target is still available via target='distance'
for completeness. This choice is documented and should be reported as such.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


# --------------------------------------------------------------------------- #
# Dice
# --------------------------------------------------------------------------- #
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred, target):
        if self.from_logits:
            pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        target = target.flatten(1).float()
        inter = (pred * target).sum(1)
        denom = pred.sum(1) + target.sum(1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return (1 - dice).mean()


# --------------------------------------------------------------------------- #
# Hausdorff on the predicted mask (Karimi & Salcudean 2020) -- VM-MedSAM baseline
# --------------------------------------------------------------------------- #
def _edt_batch(binary_np):
    """Per-sample Euclidean distance transform of a (B,1,H,W) numpy binary map."""
    out = np.zeros_like(binary_np, dtype=np.float32)
    for b in range(binary_np.shape[0]):
        m = binary_np[b, 0]
        if m.any():
            out[b, 0] = distance_transform_edt(m)
    return out


class HausdorffMaskLoss(nn.Module):
    """Differentiable DT-based Hausdorff surrogate on the predicted mask.

    L = mean( (p - g)^2 * (dt_p^alpha + dt_g^alpha) )

    where dt_p, dt_g are the (foreground) Euclidean distance transforms of the
    thresholded prediction and the GT. The DT weights are computed on detached
    tensors (non-differentiable), so gradients flow through the (p - g)^2 term
    only -- the standard Karimi formulation.
    """

    def __init__(self, alpha: float = 2.0, from_logits: bool = True,
                 edt_size: int = 256):
        super().__init__()
        self.alpha = alpha
        self.from_logits = from_logits
        # SciPy EDT cost ~ H*W; computing it at 256 instead of 1024 is ~16x
        # cheaper. The distance-weighting is a smooth regulariser, so the
        # coarser grid is fine. edt_size=0 keeps full resolution.
        self.edt_size = edt_size

    def forward(self, pred, target):
        p = torch.sigmoid(pred) if self.from_logits else pred
        g = target.float()
        if self.edt_size and p.shape[-1] > self.edt_size:
            import torch.nn.functional as _F
            p = _F.interpolate(p, size=(self.edt_size, self.edt_size),
                               mode="bilinear", align_corners=False)
            g = _F.interpolate(g, size=(self.edt_size, self.edt_size),
                               mode="nearest")
        with torch.no_grad():
            p_bin = (p > 0.5).float().cpu().numpy()
            g_bin = (g > 0.5).float().cpu().numpy()
            dt_p = _edt_batch(p_bin) + _edt_batch(1 - p_bin)
            dt_g = _edt_batch(g_bin) + _edt_batch(1 - g_bin)
            weight = torch.from_numpy(dt_p ** self.alpha + dt_g ** self.alpha).to(p.device)
            if weight.max() > 0:
                weight = weight / weight.max()
        return ((p - g) ** 2 * weight).mean()


# --------------------------------------------------------------------------- #
# Momentum-supervised boundary loss (NOVEL)
# --------------------------------------------------------------------------- #
def _boundary_proximity(gt_mask, out_hw, clip_dt, target):
    """Build the supervision map at feature resolution from a full-res GT mask.

    Returns (B,1,h,w) in [0,1]. target:
        'proximity' (default) -- 1 on the boundary, decaying linearly to 0 at
                                  >= clip_dt px (realises PLAN.md's stated intent).
        'distance'            -- normalised distance-to-boundary (the literal,
                                  inverted formula; 0 on the boundary).
        'band'                -- Gaussian band exp(-dt^2 / (2 sigma^2)).
    """
    B = gt_mask.shape[0]
    g = F.interpolate(gt_mask.float(), size=out_hw, mode='nearest').cpu().numpy()
    prox = np.zeros((B, 1, out_hw[0], out_hw[1]), dtype=np.float32)
    for b in range(B):
        m = g[b, 0] > 0.5
        if not m.any() or m.all():
            continue
        # boundary pixels: foreground pixels adjacent to background
        inner = m & ~np.all(
            np.stack([np.roll(m, s, ax) for ax in (0, 1) for s in (-1, 1)]), axis=0)
        boundary = inner if inner.any() else m
        dt = distance_transform_edt(~boundary).astype(np.float32)
        dt_n = np.clip(dt, 0, clip_dt) / clip_dt           # 0 on boundary -> 1 far
        if target == 'distance':
            prox[b, 0] = dt_n
        elif target == 'band':
            sigma = max(clip_dt / 3.0, 1.0)
            prox[b, 0] = np.exp(-(dt ** 2) / (2 * sigma ** 2))
        else:  # 'proximity'
            prox[b, 0] = 1.0 - dt_n
    return torch.from_numpy(prox).to(gt_mask.device)


class MomentumBoundaryLoss(nn.Module):
    """Encoder-level boundary supervision on the Hamiltonian momentum p.

    RECOMMENDED mode='projection' (default): a small learned 1x1 conv reads a
    boundary logit out of p, supervised (BCE + soft-Dice) against a soft
    boundary band derived from the GT. This serves the loss's purpose (sharper
    boundaries / lower 95%-Hausdorff) while still pushing gradients into p, and
    it does NOT overwrite p's natural interior-dominant structure (HamVision
    reports |p|: interior > boundary > exterior). The projection params live in
    this module, so add `loss_fn.parameters()` to the optimizer (the training
    script does this).

    mode='template': the original hypothesis -- match softplus(|p|_avg)
    (per-sample min-max normalised) directly to a target map. Kept for ablation.
    See the module docstring for why the literal 'distance' target inverts the
    plan's stated intent; 'band'/'proximity' realise the intent.

    target in {'band','proximity','distance'} selects the supervision map.
    """

    def __init__(self, lambda_p: float = 0.1, clip_dt: float = 20.0,
                 mode: str = 'projection', target: str = 'band',
                 momentum_channels: int = 256, signal: str = 'momentum'):
        super().__init__()
        assert mode in ('projection', 'template')
        assert target in ('proximity', 'distance', 'band')
        assert signal in ('momentum', 'grad_energy', 'combo')
        self.lambda_p = lambda_p
        self.clip_dt = clip_dt
        self.mode = mode
        self.target = target
        self.signal = signal
        # input channels to the projection head depend on which signal we read
        in_ch = {'momentum': momentum_channels, 'grad_energy': 1,
                 'combo': momentum_channels + 1}[signal]
        if mode == 'projection':
            self.proj = nn.Conv2d(in_ch, 1, 1)
            self.bce = nn.BCEWithLogitsLoss()
        # fixed Sobel kernels for |grad H| (boundary = where energy transitions)
        kx = torch.tensor([[1., 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
        self.register_buffer('_sx', kx)
        self.register_buffer('_sy', kx.transpose(-1, -2).contiguous())

    def _grad_energy(self, energy):
        gx = F.conv2d(energy, self._sx.to(energy.dtype), padding=1)
        gy = F.conv2d(energy, self._sy.to(energy.dtype), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)        # (B,1,h,w)

    def _input_signal(self, p, energy):
        if self.signal == 'momentum':
            return p
        if energy is None:
            raise ValueError(f"signal='{self.signal}' needs the energy map (pass energy=H_map)")
        ge = self._grad_energy(energy)
        if self.signal == 'grad_energy':
            return ge
        return torch.cat([p, ge], dim=1)                   # combo

    def forward(self, p, gt_mask, energy=None):
        if p is None:                       # baseline encoder (bottleneck='none')
            return torch.zeros((), device=gt_mask.device)
        sig = self._input_signal(p, energy)
        out_hw = sig.shape[-2:]
        with torch.no_grad():
            tgt = _boundary_proximity(gt_mask, out_hw, self.clip_dt, self.target)

        if self.mode == 'projection':
            logit = self.proj(sig)          # (B,1,h,w)
            bce = self.bce(logit, tgt)
            prob = torch.sigmoid(logit).flatten(1)
            t = tgt.flatten(1)
            dice = 1 - (2 * (prob * t).sum(1) + 1) / (prob.sum(1) + t.sum(1) + 1)
            return self.lambda_p * (bce + dice.mean())

        # template mode
        p_avg = F.softplus(sig.abs().mean(dim=1, keepdim=True))
        flat = p_avg.flatten(1)
        mn = flat.min(1, keepdim=True).values.view(-1, 1, 1, 1)
        mx = flat.max(1, keepdim=True).values.view(-1, 1, 1, 1)
        p_norm = (p_avg - mn) / (mx - mn + 1e-6)
        return self.lambda_p * (p_norm - tgt).abs().mean()


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    """dice + ce/bce (+ hausdorff) (+ momentum).

    multiclass=False -> BCEWithLogits + binary Dice.
    multiclass=True  -> CrossEntropy + per-foreground-class Dice (one-hot).
    `use_*` flags select the spec strings used in the training script
    (dice+ce, dice+ce+hausdorff, dice+ce+momentum, ...).
    """

    def __init__(self, multiclass=False, num_classes=1,
                 w_dice=1.0, w_ce=1.0, w_hausdorff=1.0,
                 use_hausdorff=False, use_momentum=False,
                 lambda_p=0.1, clip_dt=20.0, momentum_mode='projection',
                 momentum_target='band', momentum_channels=256,
                 momentum_signal='momentum'):
        super().__init__()
        self.multiclass = multiclass
        self.num_classes = num_classes
        self.w_dice, self.w_ce, self.w_hd = w_dice, w_ce, w_hausdorff
        self.use_hausdorff = use_hausdorff
        self.use_momentum = use_momentum
        self.dice = SoftDiceLoss(from_logits=True)
        self.ce = nn.CrossEntropyLoss() if multiclass else nn.BCEWithLogitsLoss()
        if use_hausdorff:
            self.hausdorff = HausdorffMaskLoss()
        if use_momentum:
            self.momentum = MomentumBoundaryLoss(lambda_p, clip_dt, momentum_mode,
                                                 momentum_target, momentum_channels,
                                                 signal=momentum_signal)

    def forward(self, logits, target, p=None, energy=None):
        """logits: (B,1,H,W) binary or (B,C,H,W) multiclass.
        target:  (B,1,H,W) {0,1} binary or (B,H,W) int64 multiclass.
        p:       encoder momentum for the momentum loss (or None).
        """
        out = {}
        if self.multiclass:
            ce = self.ce(logits, target.long())
            probs = logits.softmax(1)
            oh = F.one_hot(target.long(), self.num_classes).permute(0, 3, 1, 2).float()
            dice = self.dice.__class__(from_logits=False)(probs[:, 1:], oh[:, 1:]) \
                if self.num_classes > 1 else self.dice(logits, target)
        else:
            ce = self.ce(logits, target.float())
            dice = self.dice(logits, target)
        out['ce'] = ce
        out['dice'] = dice
        total = self.w_ce * ce + self.w_dice * dice

        if self.use_hausdorff and not self.multiclass:
            hd = self.hausdorff(logits, target)
            out['hausdorff'] = hd
            total = total + self.w_hd * hd
        if self.use_momentum:
            mb = self.momentum(p, target if not self.multiclass
                               else (target > 0).float().unsqueeze(1), energy=energy)
            out['momentum'] = mb
            total = total + mb
        out['total'] = total
        return total, out
