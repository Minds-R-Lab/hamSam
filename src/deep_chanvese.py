"""Deep Chan-Vese / phase-field segmenter -- a physics-native, prompt-free model.

Motivation (from the Ham post-mortem): a physics prior bolted onto a frozen
foundation-model decoder gets washed out, and a *static* energy map cannot do
prompt-free segmentation because it has no mechanism for global region
competition. Here the physics IS the segmenter: the output is the equilibrium
of a learned free-energy gradient flow acting directly on the label field.

Free energy (Ginzburg-Landau / Chan-Vese form):

    E[u] = INT  (eps/2)|grad u|^2  +  (1/eps) W(u)  +  lambda * d_theta(x) * u   dx

  * W(u) = (u^2 - 1)^2 / 4  -- double well: forces u -> {-1 (bg), +1 (fg)}
  * |grad u|^2              -- forces smooth, closed, short boundaries (shape prior)
  * d_theta(x)             -- LEARNED signed data term from a small U-Net
                              (d<0 pulls foreground, d>0 background)

Forward pass = gradient flow  du/dt = -dE/du  unrolled for `steps`:

    du/dt = eps * g(x) * Laplacian(u)  -  (1/eps) (u^3 - u)  -  lambda * d(x)

  with an optional learned edge map g(x) in (0,1) (anisotropic diffusion: stop
  smoothing across image edges). The equilibrium field u* (thresholded at 0) is
  the mask. No box / seed / click -> prompt-free by construction.

`use_physics=False` skips the relaxation and reads the mask straight off the
data term -- a *capacity-matched control* (same backbone, physics removed) that
isolates whether the variational dynamics earn their keep.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cba(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False),
                         nn.GroupNorm(8, co), nn.GELU())


class SmallUNet(nn.Module):
    """Compact 3-level U-Net; returns a full-resolution feature map (base ch)."""
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.e1 = nn.Sequential(_cba(in_ch, base), _cba(base, base))
        self.e2 = nn.Sequential(_cba(base, base * 2), _cba(base * 2, base * 2))
        self.e3 = nn.Sequential(_cba(base * 2, base * 4), _cba(base * 4, base * 4))
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = nn.Sequential(_cba(base * 4, base * 2), _cba(base * 2, base * 2))
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = nn.Sequential(_cba(base * 2, base), _cba(base, base))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up3(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up2(d2), e1], 1))
        return d1


class DeepChanVese(nn.Module):
    def __init__(self, in_ch=3, base=32, steps=15, use_physics=True,
                 edge_aware=True, input_size=256):
        super().__init__()
        self.backbone = SmallUNet(in_ch, base)
        self.data_head = nn.Conv2d(base, 1, 1)
        self.edge_head = nn.Conv2d(base, 1, 1) if edge_aware else None
        self.use_physics = use_physics
        self.steps = steps
        self.input_size = input_size
        # physics scalars, bounded to a CFL-stable range via sigmoid
        self.p_dt = nn.Parameter(torch.tensor(0.0))    # dt  in (0.02, 0.15)
        self.p_eps = nn.Parameter(torch.tensor(0.0))   # eps in (0.10, 1.00)
        self.p_lam = nn.Parameter(torch.tensor(0.0))   # lam in (0.00, 2.00)
        self.readout = nn.Parameter(torch.tensor(4.0))
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("lap_k", lap)

    def _lap(self, u):
        u = F.pad(u, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(u, self.lap_k)

    def forward(self, image, box=None):          # box ignored -> prompt-free
        f = self.backbone(image)
        d = self.data_head(f)                    # (B,1,H,W) signed data term
        out = {"d": d}
        if not self.use_physics:                 # capacity-matched control
            out["mask"] = self.readout.abs() * torch.tanh(d)
            out["u"] = None
            return out
        g = torch.sigmoid(self.edge_head(f)) if self.edge_head is not None else 1.0
        dt = 0.02 + 0.13 * torch.sigmoid(self.p_dt)
        eps = 0.10 + 0.90 * torch.sigmoid(self.p_eps)
        lam = 2.00 * torch.sigmoid(self.p_lam)
        u = torch.tanh(d)                        # initialise field from data term
        for _ in range(self.steps):
            lap = self._lap(u.float())
            Wp = u * u * u - u                   # W'(u) = u^3 - u
            du = eps * (g * lap) - (1.0 / eps) * Wp - lam * d
            u = (u + dt * du).clamp(-1.5, 1.5)
        out["u"] = u
        out["mask"] = self.readout.abs() * u     # logits; (u>0) -> foreground
        return out

    def energy_uncertainty(self, u):
        """Label-free uncertainty: distance of the equilibrium field from the
        double-well minima (+/-1). High near ambiguous/boundary pixels."""
        return (1.0 - u.abs()).clamp(min=0.0)

    def num_parameters(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)
