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


def test_momentum_loss_modes_and_targets():
    gt = _square(); p = torch.randn(2, 32, 16, 16, requires_grad=True)
    for mode in ('projection', 'template'):
        for tgt in ('proximity', 'distance', 'band'):
            loss = MomentumBoundaryLoss(mode=mode, target=tgt,
                                        momentum_channels=32)(p, gt)
            assert torch.isfinite(loss) and loss.item() >= 0
    # projection mode has a learned head; template mode does not
    assert sum(x.numel() for x in
               MomentumBoundaryLoss(mode='projection', momentum_channels=32).parameters()) > 0
    assert sum(x.numel() for x in
               MomentumBoundaryLoss(mode='template').parameters()) == 0
    # tolerates p=None (baseline encoder, bottleneck='none')
    assert float(MomentumBoundaryLoss(momentum_channels=32)(None, gt)) == 0.0


def test_combined_backward_all_terms():
    gt = _square(); logits = (gt * 4 - 2).requires_grad_(True)
    p = torch.randn(2, 32, 16, 16, requires_grad=True)
    loss, parts = CombinedLoss(use_hausdorff=True, use_momentum=True,
                               momentum_channels=32)(logits, gt, p)
    loss.backward()
    assert {'ce', 'dice', 'hausdorff', 'momentum', 'total'} <= set(parts)
    assert torch.isfinite(logits.grad).all() and torch.isfinite(p.grad).all()




def test_momentum_signal_variants():
    gt = _square()
    p = torch.randn(2, 32, 16, 16, requires_grad=True)
    energy = torch.rand(2, 1, 16, 16)
    from src.losses import MomentumBoundaryLoss
    # momentum (no energy needed)
    assert MomentumBoundaryLoss(signal="momentum", momentum_channels=32)(p, gt).item() >= 0
    # grad_energy (1-channel input from |grad H|)
    L = MomentumBoundaryLoss(signal="grad_energy", momentum_channels=32)
    v = L(p, gt, energy=energy); v.backward(retain_graph=True); assert torch.isfinite(v)
    # combo (p + |grad H|)
    Lc = MomentumBoundaryLoss(signal="combo", momentum_channels=32)
    vc = Lc(p, gt, energy=energy); assert torch.isfinite(vc)
    # grad_energy without energy must error
    try:
        MomentumBoundaryLoss(signal="grad_energy", momentum_channels=32)(p, gt)
        assert False, "expected error when energy missing"
    except ValueError:
        pass


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"): v(); print("PASS", k)
