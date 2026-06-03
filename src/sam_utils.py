"""SAM / MedSAM prompt-encoder + mask-decoder loading, with a CPU fallback.

Real path (recommended for the H100 runs): install the official
`segment-anything` package and pass a MedSAM (ViT-B) checkpoint. We then reuse
SAM's PromptEncoder and MaskDecoder unchanged -- exactly as VM-MedSAM does --
and only the image encoder is swapped for HamEncoder.

Fallback path (no package / no checkpoint, e.g. CI smoke tests): a tiny
box-conditioned conv decoder so the whole model is still runnable end-to-end on
CPU. It is NOT a faithful SAM decoder and must not be used for reported numbers;
build_sam_components() emits a clear warning when it falls back.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _box_to_dense_mask(boxes, size, device):
    """Rasterise (B,4) xyxy boxes (in `size` px) to a (B,1,size,size) mask."""
    B = boxes.shape[0]
    m = torch.zeros(B, 1, size, size, device=device)
    for b in range(B):
        x0, y0, x1, y1 = boxes[b].clamp(0, size - 1).round().int().tolist()
        m[b, 0, y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = 1.0
    return m


class FallbackMaskDecoder(nn.Module):
    """Tiny stand-in: image embedding (B,256,64,64) + box -> mask (B,1,1024,1024)."""

    def __init__(self, embed_dim=256, out_size=1024):
        super().__init__()
        self.out_size = out_size
        self.box_embed = nn.Conv2d(1, 32, 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim + 32, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, image_embeddings, boxes):
        B, _, h, w = image_embeddings.shape
        box_mask = _box_to_dense_mask(boxes, self.out_size, image_embeddings.device)
        box_feat = F.interpolate(box_mask, size=(h, w), mode='bilinear', align_corners=False)
        box_feat = self.box_embed(box_feat)
        x = self.fuse(torch.cat([image_embeddings, box_feat], 1))
        return F.interpolate(x, size=(self.out_size, self.out_size),
                             mode='bilinear', align_corners=False)


# Why MedSAM ViT-B is the default backend (June 2026 review):
#   * VM-MedSAM (the model we extend) uses SAM ViT-B; its image embedding is
#     256x64x64, exactly what HamEncoder emits, so only the encoder is swapped
#     and the comparison is apples-to-apples.
#   * SAM 2 / MedSAM2 (Hiera + memory, 3D/video) need MULTI-SCALE FPN features
#     in the mask decoder, not a single embedding -- HamEncoder would need an
#     FPN-style multi-output head first. Different research thread.
#   * SAM 3 / 3.1 (Nov 2025 / Mar 2026) is open-vocabulary *concept* (text)
#     segmentation, a different paradigm from box-prompted single-structure
#     MedSAM. Out of scope for this paper.
# The sam2/medsam2/sam3 backends are documented extension points below.
SUPPORTED_BACKENDS = ("medsam_vitb", "sam_vitb", "sam2", "medsam2", "sam3")


def build_sam_components(sam_checkpoint=None, model_type="vit_b", out_size=1024,
                         backend="medsam_vitb"):
    """Return (prompt_encoder, mask_decoder, kind). kind in {'sam','fallback'}.

    backend:
        'medsam_vitb' / 'sam_vitb' -- official segment-anything ViT-B (default).
        'sam2' / 'medsam2' / 'sam3' -- not wired yet; raise with the reason.
    """
    if backend in ("sam2", "medsam2"):
        raise NotImplementedError(
            f"backend='{backend}' is a documented extension point, not wired. "
            "SAM2/MedSAM2 mask decoders consume multi-scale Hiera FPN features; "
            "HamEncoder must expose an FPN-style multi-output head before this "
            "backend can be used. Keep 'medsam_vitb' for the VM-MedSAM comparison.")
    if backend == "sam3":
        raise NotImplementedError(
            "backend='sam3' (open-vocabulary concept/text segmentation) is a "
            "different paradigm from box-prompted MedSAM; not applicable to this "
            "model. A text-promptable Ham-MedSAM would be separate future work.")
    try:
        if sam_checkpoint is None:
            raise RuntimeError("no checkpoint provided")
        from segment_anything import sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        return sam.prompt_encoder, sam.mask_decoder, "sam"
    except NotImplementedError:
        raise
    except Exception as e:  # pragma: no cover - exercised only without the package
        warnings.warn(
            f"segment_anything unavailable or no checkpoint ({e}); using the "
            "FallbackMaskDecoder. Do NOT use for reported metrics -- install "
            "segment-anything and pass a MedSAM checkpoint for real runs.")
        return None, FallbackMaskDecoder(out_size=out_size), "fallback"
