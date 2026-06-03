import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.losses import SoftDiceLoss, HausdorffMaskLoss, MomentumBoundaryLoss, CombinedLoss


def _square():
    gt = torch.zeros(2, 1, 64, 64); gt[:, :, 20:44, 24:40] = 1
    return gt


def test_dice_perfect_is_low():
    gt = _square()
    perfect = (gt * 20 - 10)   # very confident logits
    assert SoftDiceLoss()(perfect, gt).item() < 0.05


def test_hausdorff_nonneg_and_grad():
    gt = _square(); logits = (gt * 4 - 2).requires_grad_(True)
    hd = HausdorffMaskLoss()(logits, gt)
    assert hd.item() >= 0
    hd.backward(); assert torch.isfinite(logits.grad).all()


def test_momentum_loss_targets():
    gt = _square(); p = torch.randn(2, 32, 16, 16, requires_grad=True)
    for tgt in ('proximity', 'distance', 'band'):
        loss = MomentumBoundaryLoss(target=tgt)(p, gt)
        assert torch.isfinite(loss) and loss.item() >= 0
    MomentumBoundaryLoss()(None, gt)   # tolerates p=None (baseline encoder)


def test_combined_backward_all_terms():
    gt = _square(); logits = (gt * 4 - 2).requires_grad_(True)
    p = torch.randn(2, 32, 16, 16, requires_grad=True)
    loss, parts = CombinedLoss(use_hausdorff=True, use_momentum=True)(logits, gt, p)
    loss.backward()
    assert {'ce', 'dice', 'hausdorff', 'momentum', 'total'} <= set(parts)
    assert torch.isfinite(logits.grad).all() and torch.isfinite(p.grad).all()


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"): v(); print("PASS", k)
