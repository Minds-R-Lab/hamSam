"""Dataset converters: raw NIfTI volumes -> Ham-MedSAM training layout.

Currently implements the two abdomen-CT datasets used by VM-MedSAM (BTCV,
FLARE22). Each volume is windowed, sliced axially, and EXPLODED into per-organ
binary (image, mask) pairs -- the SAM/MedSAM box-promptable paradigm, where one
sample = one organ instance on one slice. Splits are PATIENT-LEVEL (no slice
leakage), matching VM-MedSAM.

Output (read by data.datasets.MedSegDataset):
    <out>/<dataset>/{train,val,test}/images/<pid>_z<zzz>_org<L>_<name>.npy
    <out>/<dataset>/{train,val,test}/masks/ <pid>_z<zzz>_org<L>_<name>.npy

ACCESS (see data/README.md):
  * FLARE22 labeled set is openly downloadable from Zenodo (record 7860267).
  * BTCV requires a Synapse account + data-use agreement (syn3193805).
  Use scripts/download_datasets.sh.

NOTE: label maps below follow the widely-used conventions; verify against the
documentation shipped with YOUR download (label order has varied across
re-releases). Pass --label_map_json to override.
"""
import argparse
import glob
import json
import os
import random

import numpy as np

# Standard label conventions (1-indexed; 0 = background).
FLARE22_LABELS = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas",
                  5: "aorta", 6: "ivc", 7: "right_adrenal", 8: "left_adrenal",
                  9: "gallbladder", 10: "esophagus", 11: "stomach",
                  12: "duodenum", 13: "left_kidney"}
BTCV_LABELS = {1: "spleen", 2: "right_kidney", 3: "left_kidney",
               4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach",
               8: "aorta", 9: "ivc", 10: "portal_splenic_vein",
               11: "pancreas", 12: "right_adrenal", 13: "left_adrenal"}

DATASETS = {
    "btcv":    dict(labels=BTCV_LABELS,    hu=(-160, 240)),
    "flare22": dict(labels=FLARE22_LABELS, hu=(-160, 240)),
}


import re


def _stem(name):
    b = os.path.basename(name)
    for ext in (".nii.gz", ".nii"):
        if b.endswith(ext):
            return b[:-len(ext)]
    return os.path.splitext(b)[0]


def _patient_id(name, is_image):
    """Patient id = last digit-run of the stem. For images, first strip a
    trailing _DDDD modality-channel suffix (nnU-Net/FLARE convention, e.g.
    FLARE22_Tr_0001_0000 -> 0001), falling back to the full stem if that
    leaves no digits. Avoids matching on dataset-name digits like 'FLARE22'."""
    stem = _stem(name)
    cand = re.sub(r"_\d{4}$", "", stem) if is_image else stem
    runs = re.findall(r"\d+", cand) or re.findall(r"\d+", stem)
    return (runs[-1].lstrip("0") or "0") if runs else stem


def _pair_files(images_dir, labels_dir):
    """Pair image/label NIfTI files by patient id (channel-suffix aware)."""
    exts = ("*.nii", "*.nii.gz")
    imgs = sorted(sum([glob.glob(os.path.join(images_dir, e)) for e in exts], []))
    lbls = sorted(sum([glob.glob(os.path.join(labels_dir, e)) for e in exts], []))
    img_by_id = {}
    for im in imgs:
        img_by_id.setdefault(_patient_id(im, True), im)
    pairs = []
    for lb in lbls:
        pid = _patient_id(lb, False)
        if pid in img_by_id:
            pairs.append((pid, img_by_id[pid], lb))
    return pairs


_IMG_DIR_NAMES = ("images", "imagestr", "imagests", "img", "imgs", "image")
_LBL_DIR_NAMES = ("labels", "labelstr", "labelsts", "label", "lbl", "masks", "mask", "gt")


def _has_nifti(d):
    return bool(glob.glob(os.path.join(d, "*.nii")) or glob.glob(os.path.join(d, "*.nii.gz")))


def autodetect_dirs(root):
    """Find the images/labels subdirs under `root` (handles e.g. the FLARE22
    zip's FLARE22Train/{images,labels} layout). Matches by directory name and
    the presence of NIfTI files."""
    img_dir = lbl_dir = None
    for dirpath, _, _ in os.walk(root):
        base = os.path.basename(dirpath).lower()
        if not _has_nifti(dirpath):
            continue
        if img_dir is None and base in _IMG_DIR_NAMES:
            img_dir = dirpath
        elif lbl_dir is None and base in _LBL_DIR_NAMES:
            lbl_dir = dirpath
    if img_dir is None or lbl_dir is None:
        raise FileNotFoundError(
            f"could not autodetect images/labels dirs under {root}. "
            f"Pass --images_dir and --labels_dir explicitly. "
            f"(found images={img_dir}, labels={lbl_dir})")
    return img_dir, lbl_dir


def _window_ct(vol, lo, hi):
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo)).astype(np.float32)


def convert_nifti_dataset(images_dir, labels_dir, out_dir, label_map, hu_window,
                          split=(0.7, 0.1, 0.2), min_organ_px=20, seed=42):
    try:
        import nibabel as nib
    except ImportError as e:
        raise SystemExit("nibabel is required for NIfTI conversion: "
                         "`pip install nibabel` (or pip install -r requirements.txt)") from e

    pairs = _pair_files(images_dir, labels_dir)
    if not pairs:
        raise FileNotFoundError(
            f"no NIfTI image/label pairs found in {images_dir} / {labels_dir}")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_tr, n_va = int(n * split[0]), int(n * split[1])
    split_of = {}
    for i, (pid, _, _) in enumerate(pairs):
        split_of[pid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    for sp in ("train", "val", "test"):
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(out_dir, sp, sub), exist_ok=True)

    lo, hi = hu_window
    counts = {"train": 0, "val": 0, "test": 0}
    for pid, img_path, lbl_path in pairs:
        sp = split_of[pid]
        img = nib.as_closest_canonical(nib.load(img_path)).get_fdata()
        lbl = nib.as_closest_canonical(nib.load(lbl_path)).get_fdata().astype(np.int16)
        if img.shape != lbl.shape:
            print(f"[skip] shape mismatch {pid}: {img.shape} vs {lbl.shape}")
            continue
        img = _window_ct(img, lo, hi)
        Z = img.shape[2]                              # axial = last axis (canonical)
        for z in range(Z):
            sl_lbl = lbl[:, :, z]
            present = [L for L in label_map if (sl_lbl == L).sum() >= min_organ_px]
            if not present:
                continue
            sl_img = np.rot90(img[:, :, z])           # upright for viewing
            sl_lbl_r = np.rot90(sl_lbl)
            for L in present:
                name = label_map[L]
                base = f"{pid}_z{z:03d}_org{L}_{name}"
                np.save(os.path.join(out_dir, sp, "images", base + ".npy"),
                        sl_img.astype(np.float32))
                np.save(os.path.join(out_dir, sp, "masks", base + ".npy"),
                        (sl_lbl_r == L).astype(np.uint8))
                counts[sp] += 1
    print(f"done: {counts} samples across train/val/test (patients: "
          f"{n_tr}/{n_va}/{n - n_tr - n_va})")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--root", default=None,
                    help="raw dataset root; auto-detects images/labels subdirs "
                         "(e.g. data/raw/flare22 -> FLARE22Train/{images,labels})")
    ap.add_argument("--images_dir", default=None, help="dir of *.nii/.nii.gz CT volumes")
    ap.add_argument("--labels_dir", default=None, help="dir of *.nii/.nii.gz label maps")
    ap.add_argument("--out", required=True, help="output root, e.g. data/processed/flare22")
    ap.add_argument("--min_organ_px", type=int, default=20)
    ap.add_argument("--label_map_json", default=None,
                    help="optional JSON {int_label: name} overriding the default")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    label_map = cfg["labels"]
    if args.label_map_json:
        label_map = {int(k): v for k, v in json.load(open(args.label_map_json)).items()}
    images_dir, labels_dir = args.images_dir, args.labels_dir
    if images_dir is None or labels_dir is None:
        if args.root is None:
            ap.error("provide either --root, or both --images_dir and --labels_dir")
        images_dir, labels_dir = autodetect_dirs(args.root)
        print(f"autodetected images={images_dir}  labels={labels_dir}")
    convert_nifti_dataset(images_dir, labels_dir, args.out, label_map,
                          cfg["hu"], min_organ_px=args.min_organ_px, seed=args.seed)


if __name__ == "__main__":
    main()
