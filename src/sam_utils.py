"""SAM / MedSAM prompt-encoder + mask-decoder loading, with three tiers.

Tiers (chosen automatically by build_sam_components):
  1. 'sam' + checkpoint -- official segment-anything ViT-B with MedSAM/SAM
     pretrained prompt encoder + mask decoder (input must be 1024). Use for
     real H100 runs.
  2. 'sam' from scratch -- real segment-anything PromptEncoder + MaskDecoder
     built at image_embedding_size = input_size//16, RANDOM init, no download.
     Exercises the true decoder code path on CPU at any resolution; turnkey for
     smoke testing every component end-to-end.
  3. 'fallback' -- a tiny box-conditioned conv decoder, only if segment-anything
     is unavailable. NOT for reported metrics.

Note: VM-MedSAM (and MedSAM) use SAM ViT-B; the image embedding is 256x64x64,
which HamEncoder reproduces, so only the image encoder is swapped.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_BACKENDS = ("medsam_vitb", "sam_vitb", "sam2", "medsam2", "sam3")
SAM_EMBED_DIM = 256
SAM_PATCH = 16  # 1024 input -> 64x64 embedding


def ensure_sam_importable():
    """Make `segment_anything.modeling` importable even if torchvision is broken.

    segment_anything's package __init__ pulls in SamPredictor / the automatic
    mask generator, which import torchvision -- neither is used by Ham-MedSAM.
    On environments with a missing/mismatched torchvision we register minimal
    stubs for exactly the symbols those modules import, so the *modeling*
    classes (which need no torchvision) load. On a healthy install this is a
    no-op.
    """
    try:
        import torchvision  # noqa: F401
        return
    except Exception:
        import sys
        import types

        def pkg(name):
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
            return m

        def _unavailable(*a, **k):
            raise RuntimeError("stubbed torchvision symbol is not available")

        pkg("torchvision")
        pkg("torchvision.transforms")
        tf = pkg("torchvision.transforms.functional")
        tf.resize = _unavailable
        tf.to_pil_image = _unavailable
        pkg("torchvision.ops")
        ob = pkg("torchvision.ops.boxes")
        ob.batched_nms = _unavailable
        ob.box_area = _unavailable
        sys.modules["torchvision.transforms"].functional = tf
        sys.modules["torchvision.ops"].boxes = ob


def _build_sam_from_scratch(out_size):
    """Real segment-anything PromptEncoder + MaskDecoder, random init, no ckpt."""
    ensure_sam_importable()
    from segment_anything.modeling import (
        PromptEncoder, MaskDecoder, TwoWayTransformer)
    es = out_size // SAM_PATCH
    prompt_encoder = PromptEncoder(
        embed_dim=SAM_EMBED_DIM, image_embedding_size=(es, es),
        input_image_size=(out_size, out_size), mask_in_chans=16)
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=SAM_EMBED_DIM,
                                      mlp_dim=2048, num_heads=8),
        transformer_dim=SAM_EMBED_DIM, iou_head_depth=3, iou_head_hidden_dim=256)
    return prompt_encoder, mask_decoder


def _box_to_dense_mask(boxes, size, device):
    B = boxes.shape[0]
    m = torch.zeros(B, 1, size, size, device=device)
    for b in range(B):
        x0, y0, x1, y1 = boxes[b].clamp(0, size - 1).round().int().tolist()
        m[b, 0, y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = 1.0
    return m


class FallbackMaskDecoder(nn.Module):
    """Tiny stand-in used only when segment-anything is unavailable."""

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


def build_sam_components(sam_checkpoint=None, model_type="vit_b", out_size=1024,
                         backend="medsam_vitb"):
    """Return (prompt_encoder, mask_decoder, kind). kind in {'sam','fallback'}."""
    if backend in ("sam2", "medsam2"):
        raise NotImplementedError(
            f"backend='{backend}' is a documented extension point, not wired. "
            "SAM2/MedSAM2 mask decoders consume multi-scale Hiera FPN features; "
            "HamEncoder must expose an FPN-style multi-output head first.")
    if backend == "sam3":
        raise NotImplementedError(
            "backend='sam3' (open-vocabulary concept/text segmentation) is a "
            "different paradigm from box-prompted MedSAM; not applicable here.")

    # Tier 1: pretrained checkpoint via the registry (1024 input only).
    if sam_checkpoint is not None:
        if out_size != 1024:
            raise ValueError(
                "A SAM/MedSAM checkpoint expects 1024 input (64x64 embedding); "
                f"got out_size={out_size}. Use input_size=1024 with a checkpoint, "
                "or omit the checkpoint to build from scratch at this resolution.")
        ensure_sam_importable()
        from segment_anything import sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        return sam.prompt_encoder, sam.mask_decoder, "sam"

    # Tier 2: real SAM modules from scratch (random init, no download).
    try:
        pe, md = _build_sam_from_scratch(out_size)
        return pe, md, "sam"
    except Exception as e:  # pragma: no cover
        warnings.warn(
            f"segment-anything unavailable ({e}); using FallbackMaskDecoder. "
            "Install segment-anything for the real decoder path.")
        return None, FallbackMaskDecoder(out_size=out_size), "fallback"
