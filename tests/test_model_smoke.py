import os, sys, warnings
import torch
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ham_medsam import HamMedSAM


def _run(**kw):
    m = HamMedSAM(input_size=128, **kw)
    img = torch.randn(1, 3, 128, 128)
    box = torch.tensor([[20, 20, 100, 100]], dtype=torch.float32)
    prompt_free = kw.get("prompt_free") or kw.get("multiclass_head")
    out = m(img, box=None if prompt_free else box)
    return m, out


def test_all_configs_forward_backward():
    configs = [dict(), dict(prompt_free=True), dict(use_pssp_decoder=True),
               dict(multiclass_head=True, num_classes=4),
               dict(bottleneck='all'), dict(bottleneck='none')]
    for kw in configs:
        m, out = _run(**kw)
        assert out['mask'].shape[-2:] == (128, 128)
        out['mask'].mean().backward()   # gradients flow through trainable params
        grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"): v(); print("PASS", k)
