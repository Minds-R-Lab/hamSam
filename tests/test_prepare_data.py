"""Converter logic: patient-level split, per-organ binary explosion, windowing.
Skipped if nibabel is unavailable."""
import os, sys, tempfile
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nib = pytest.importorskip("nibabel")
from data.prepare_data import convert_nifti_dataset, _patient_id


def test_patient_id_extraction():
    assert _patient_id("FLARE22_Tr_0012_0000.nii.gz", True) == "12"   # strip _0000 channel
    assert _patient_id("FLARE22_Tr_0012.nii.gz", False) == "12"       # label
    assert _patient_id("img0007.nii.gz", True) == "7"
    assert _patient_id("label0007.nii.gz", False) == "7"


def _fake_volume(path_img, path_lbl):
    yy, xx = np.ogrid[:48, :48]
    vol = (np.random.RandomState(0).rand(48, 48, 12) * 400 - 200).astype(np.float32)
    lab = np.zeros((48, 48, 12), np.int16)
    for L, (cy, cx, r) in {1: (12, 12, 6), 3: (30, 30, 5)}.items():
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        for z in range(3, 9):
            lab[:, :, z][disk] = L
    nib.save(nib.Nifti1Image(vol, np.eye(4)), path_img)
    nib.save(nib.Nifti1Image(lab.astype(np.float32), np.eye(4)), path_lbl)


def test_convert_patient_split_and_binary_masks():
    with tempfile.TemporaryDirectory() as d:
        idir, ldir = os.path.join(d, "img"), os.path.join(d, "lab")
        os.makedirs(idir); os.makedirs(ldir)
        for pid in range(1, 6):  # 5 patients
            _fake_volume(os.path.join(idir, f"FLARE22_Tr_{pid:04d}_0000.nii.gz"),
                         os.path.join(ldir, f"FLARE22_Tr_{pid:04d}.nii.gz"))
        out = os.path.join(d, "proc")
        from data.prepare_data import FLARE22_LABELS
        convert_nifti_dataset(idir, ldir, out, FLARE22_LABELS,
                              normalize="ct_window", hu=(-160, 240),
                              split=(0.6, 0.2, 0.2), min_organ_px=5, seed=0)
        import glob
        pid_of = lambda p: os.path.basename(p).split("_")[0]
        splits = {sp: {pid_of(x) for x in glob.glob(f"{out}/{sp}/images/*.npy")}
                  for sp in ("train", "val", "test")}
        # no patient appears in more than one split (no leakage)
        all_pids = [p for s in splits.values() for p in s]
        assert len(all_pids) == len(set(all_pids)), f"patient leakage: {splits}"
        # masks binary, images in [0,1]
        m = np.load(glob.glob(f"{out}/*/masks/*.npy")[0])
        im = np.load(glob.glob(f"{out}/*/images/*.npy")[0])
        assert set(np.unique(m)).issubset({0, 1})
        assert 0.0 <= im.min() and im.max() <= 1.0


def test_convert_image_dataset_2d():
    pytest.importorskip("PIL")
    from PIL import Image
    from data.prepare_data import convert_image_dataset
    with tempfile.TemporaryDirectory() as d:
        idir = os.path.join(d, "images"); mdir = os.path.join(d, "masks")
        os.makedirs(idir); os.makedirs(mdir)
        for i in range(6):
            Image.fromarray((np.random.rand(32, 40, 3) * 255).astype(np.uint8)).save(f"{idir}/{i}.png")
            m = np.zeros((32, 40), np.uint8); m[8:20, 10:28] = 255
            Image.fromarray(m).save(f"{mdir}/{i}.png")
        out = os.path.join(d, "out")
        convert_image_dataset(idir, d, out, "polyp", mask_subdirs=["masks"],
                              pair_by="stem", thresh=0, seed=0)
        import glob
        fi = glob.glob(f"{out}/*/images/*.npy"); mi = glob.glob(f"{out}/*/masks/*.npy")
        assert len(fi) == 6 and len(mi) == 6
        assert "_org1_polyp" in os.path.basename(fi[0])
        assert set(np.unique(np.load(mi[0]))).issubset({0, 1})


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"): v(); print("PASS", k)
