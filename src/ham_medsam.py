"""Full Ham-MedSAM model: Hamiltonian image encoder + (frozen) SAM decoder.

    image (B,3,1024,1024)
        -> HamEncoder -> feat (B,256,64,64), p, H_map
    box (user-supplied OR auto from H_map when prompt_free=True)
        -> SAM prompt encoder
    feat + prompt -> SAM mask decoder -> mask (B,1,1024,1024)

Optional, paper extensions:
    use_pssp_decoder  (ext 4): Phase-Space Spectral Pooling features from
        z = feat + i*p augment the decoder input.
    multiclass_head   (ext 5): one forward pass -> N-class logits from
        fused features + energy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ham_encoder import HamEncoder
from .prompt_free import EnergyToBox
from .sam_utils import build_sam_components


class PSSPModule(nn.Module):
    """Phase-Space Spectral Pooling on z = feat + i*p (PLAN.md ext 4, K=12).

    Per-row FFT of the complex signal, keep K low-frequency bins, concatenate
    real/imag/magnitude, project back to embed_dim and add to feat. (We augment
    the decoder *input* feature rather than splice into SAM's cross-attention
    keys -- a simpler, decoder-agnostic realisation of the same signal.)
    """

    def __init__(self, dim, K=12):
        super().__init__()
        self.K = K
        self.proj = nn.Conv2d(dim + 3 * K, dim, 1)

    def forward(self, feat, p):
        if p is None:
            return feat
        B, C, H, W = feat.shape
        z = torch.complex(feat.float(), p.float())          # (B,C,H,W)
        Z = torch.fft.fft(z, dim=-1)[..., :self.K]           # (B,C,H,k), k=min(K,W)
        def _spread(t):                                      # (B,H,k) -> (B,K,H,W)
            k = t.shape[-1]
            if k < self.K:                                   # zero-pad bins to K
                t = F.pad(t, (0, self.K - k))
            return t.permute(0, 2, 1).unsqueeze(-1).expand(B, self.K, H, W)
        re = _spread(Z.real.mean(1)); im = _spread(Z.imag.mean(1)); mag = _spread(Z.abs().mean(1))
        spec = torch.cat([re, im, mag], 1).to(feat.dtype)    # (B,3K,H,W)
        return self.proj(torch.cat([feat, spec], 1))


class MultiClassEnergyHead(nn.Module):
    """feat (+ energy) -> N-class logits in one forward pass (PLAN.md ext 5)."""

    def __init__(self, dim, num_classes, out_size=1024):
        super().__init__()
        self.out_size = out_size
        self.head = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, num_classes, 1),
        )

    def forward(self, feat, H_map):
        h = H_map if H_map is not None else feat[:, :1] * 0
        x = self.head(torch.cat([feat, h], 1))
        return F.interpolate(x, size=(self.out_size, self.out_size),
                             mode='bilinear', align_corners=False)


class HamMedSAM(nn.Module):
    def __init__(self, sam_checkpoint=None, model_type="vit_b", backend="medsam_vitb",
                 bottleneck='deepest', ablation='none',
                 freeze_prompt_encoder=True, freeze_mask_decoder=False,
                 prompt_free=False, use_pssp_decoder=False,
                 multiclass_head=False, num_classes=1, input_size=1024):
        super().__init__()
        self.input_size = input_size
        self.prompt_free = prompt_free
        self.multiclass = multiclass_head

        self.image_encoder = HamEncoder(bottleneck=bottleneck, ablation=ablation,
                                        input_size=input_size)
        self.prompt_encoder, self.mask_decoder, self.sam_kind = \
            build_sam_components(sam_checkpoint, model_type, out_size=input_size,
                                 backend=backend)

        if freeze_prompt_encoder and self.prompt_encoder is not None:
            for q in self.prompt_encoder.parameters():
                q.requires_grad = False
        if freeze_mask_decoder and self.mask_decoder is not None:
            for q in self.mask_decoder.parameters():
                q.requires_grad = False

        self.energy_to_box = EnergyToBox(sam_input_size=input_size)
        self.use_pssp = use_pssp_decoder
        if use_pssp_decoder:
            self.pssp = PSSPModule(256)
        if multiclass_head:
            self.mc_head = MultiClassEnergyHead(256, num_classes, out_size=input_size)

    def set_mask_decoder_trainable(self, flag: bool):
        if self.mask_decoder is not None:
            for q in self.mask_decoder.parameters():
                q.requires_grad = flag

    def _decode(self, feat, box):
        if self.sam_kind == "sam":
            image_pe = self.prompt_encoder.get_dense_pe()
            sparse, dense = self.prompt_encoder(points=None, boxes=box, masks=None)
            low_res, _ = self.mask_decoder(
                image_embeddings=feat, image_pe=image_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False)
            return F.interpolate(low_res, size=(self.input_size, self.input_size),
                                 mode='bilinear', align_corners=False)
        return self.mask_decoder(feat, box)        # fallback

    def forward(self, image, box=None):
        enc = self.image_encoder(image)
        feat, p, H_map = enc['feat'], enc['p'], enc['H_map']

        if self.use_pssp:
            feat = self.pssp(feat, p)

        out = {'p': p, 'H_map': H_map}

        if self.multiclass:
            out['mask'] = self.mc_head(feat, H_map)   # (B, num_classes, H, W)
            out['box'] = None
            return out

        if self.prompt_free:
            assert box is None, "prompt_free=True derives the box from H_map."
            box = self.energy_to_box(H_map)
        elif box is None:
            raise ValueError("box is required unless prompt_free=True.")

        out['mask'] = self._decode(feat, box)         # (B,1,H,W) logits
        out['box'] = box
        return out

    def num_parameters(self, trainable_only=False):
        ps = self.parameters()
        return sum(q.numel() for q in ps if (q.requires_grad or not trainable_only))
