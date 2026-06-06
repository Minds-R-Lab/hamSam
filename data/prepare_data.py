"""Dataset converters: raw NIfTI volumes -> Ham-MedSAM training layout.

Implements the FOUR volumetric VM-MedSAM datasets:
  * btcv, flare22  -- abdomen CT, 13 organs -> per-organ binary 2D slices.
  * msd_lung       -- chest CT, lung cancer (lung window) -> binary.
  * brats          -- brain MRI (FLAIR), whole tumor -> binary.

Each volume is normalised (CT HU window or MRI percentile), sliced axially, and
written as binary (image, mask) pairs -- the SAM/MedSAM box-promptable paradigm
(one sample = one structure on one slice). Splits are PATIENT-LEVEL (no slice
leakage), matching VM-MedSAM.

Output (read by data.datasets.MedSegDataset):
    <out>/<dataset>/{train,val,test}/images/<pid>_z<zzz>_org<L>_<name>.npy
    <out>/<dataset>/{train,val,test}/masks/ <pid>_z<zzz>_org<L>_<name>.npy

ACCESS (see data/README.md): FLARE22/MSD/BraTS have open mirrors; BTCV needs a
Synapse account + DUA. Use scripts/download_datasets.sh.

NOTES / verify against YOUR download:
  * Label maps follow common conventions; override with --label_map_json.
  * BraTS: MSD Task01 stores a 4-D image [FLAIR,T1w,T1gd,T2w]; we take FLAIR
    (modality_index=0). If your BraTS ships separate *_flair.nii.gz, point
    --images_dir at those (3-D) and it just works. Whole tumor = any nonzero
    seg label.
  * MSD-Lung: Task06 label 1 = tumour; lung window (-1000, 400) HU.
"""
import argparse
import glob
import json
import os
import random
import re

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

# normalize: 'ct_window' uses hu=(lo,hi); 'mri_percentile' clips to [p0.5,p99.5]
# of nonzero voxels then min-max. merge_foreground collapses all nonzero labels
# into one class named fg_name (for whole-tumor). modality_index picks a channel
# from a 4-D image.
DATASETS = {
    "btcv":    dict(labels=BTCV_LABELS,    normalize="ct_window", hu=(-160, 240)),
    "flare22": dict(labels=FLARE22_LABELS, normalize="ct_window", hu=(-160, 240)),
    "msd_lung": dict(labels={1: "lung_cancer"}, normalize="ct_window",
                     hu=(-1000, 400), min_organ_px=5),
    "brats":   dict(labels=None, normalize="mri_percentile", merge_foreground=True,
                    fg_name="whole_tumor", modality_index=0, min_organ_px=20),
}


def _stem(name):
    b = os.path.basename(name)
    for ext in (".nii.gz", ".nii"):
        if b.endswith(ext):
            return b[:-len(ext)]
    return os.path.splitext(b)[0]


def _patient_id(name, is_image):
    """Patient id = last digit-run of the stem. For images, first strip a
    trailing _DDDD modality-channel suffix (nnU-Net/FLARE convention)."""
    stem = _stem(name)
    cand = re.sub(r"_\d{4}$", "", stem) if is_image else stem
    runs = re.findall(r"\d+", cand) or re.findall(r"\d+", stem)
    return (runs[-1].lstrip("0") or "0") if runs else stem


def _pair_files(images_dir, labels_dir):
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


def _normalize_volume(vol, mode, hu):
    if mode == "ct_window":
        lo, hi = hu
        vol = np.clip(vol, lo, hi)
        return ((vol - lo) / (hi - lo)).astype(np.float32)
    if mode == "mri_percentile":
        nz = vol[vol > 0]
        if nz.size:
            lo, hi = np.percentile(nz, [0.5, 99.5])
        else:
            lo, hi = float(vol.min()), float(vol.max())
        if hi <= lo:
            hi = lo + 1.0
        vol = np.clip(vol, lo, hi)
        return ((vol - lo) / (hi - lo)).astype(np.float32)
    raise ValueError(f"unknown normalize mode {mode}")


def convert_nifti_dataset(images_dir, labels_dir, out_dir, label_map,
                          normalize="ct_window", hu=(-160, 240),
                          merge_foreground=False, fg_name="lesion",
                          modality_index=None, split=(0.7, 0.1, 0.2),
                          min_organ_px=20, seed=42):
    try:
        import nibabel as nib
    except ImportError as e:
        raise SystemExit("nibabel is required: `pip install nibabel`") from e

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

    counts = {"train": 0, "val": 0, "test": 0}
    for pid, img_path, lbl_path in pairs:
        sp = split_of[pid]
        img = nib.as_closest_canonical(nib.load(img_path)).get_fdata()
        if img.ndim == 4:                              # 4-D (e.g. MSD BraTS): pick modality
            mi = modality_index if modality_index is not None else 0
            img = img[..., mi]
        lbl = nib.as_closest_canonical(nib.load(lbl_path)).get_fdata().astype(np.int16)
        if img.shape != lbl.shape:
            print(f"[skip] shape mismatch {pid}: {img.shape} vs {lbl.shape}")
            continue
        img = _normalize_volume(img, normalize, hu)
        Z = img.shape[2]
        for z in range(Z):
            sl_lbl = lbl[:, :, z]
            if merge_foreground:
                present = [1] if (sl_lbl > 0).sum() >= min_organ_px else []
            else:
                present = [L for L in label_map if (sl_lbl == L).sum() >= min_organ_px]
            if not present:
                continue
            sl_img = np.rot90(img[:, :, z])
            sl_lbl_r = np.rot90(sl_lbl)
            for L in present:
                if merge_foreground:
                    name, mask = fg_name, (sl_lbl_r > 0)
                else:
                    name, mask = label_map[L], (sl_lbl_r == L)
                base = f"{pid}_z{z:03d}_org{L}_{name}"
                np.save(os.path.join(out_dir, sp, "images", base + ".npy"),
                        sl_img.astype(np.float32))
                np.save(os.path.join(out_dir, sp, "masks", base + ".npy"),
                        mask.astype(np.uint8))
                counts[sp] += 1
    print(f"done: {counts} samples across train/val/test (patients: "
          f"{n_tr}/{n_va}/{n - n_tr - n_va})")
    return counts




# --------------------------------------------------------------------------- #
# 2-D image datasets (single-target -> ideal prompt-free testbeds)
# --------------------------------------------------------------------------- #
# pair_by: 'stem' (image stem == mask stem, after stripping mask_suffix) or
# 'id' (match on the trailing digit-run, for DRIVE's 21_training / 21_manual1).
# mask_subdirs: list -> OR-merge (Montgomery left+right lung). binarize: pixels
# > thresh are foreground. VERIFY these against your actual download layout.
DATASETS_2D = {
    "cvc_clinicdb": dict(name="polyp",        mask_subdirs=["masks"],
                         pair_by="stem", thresh=0),
    "busi":         dict(name="breast_tumor", mask_subdirs=["masks"],
                         pair_by="stem", mask_suffix="_mask", thresh=127),
    "drive":        dict(name="vessel",       mask_subdirs=["masks"],
                         pair_by="id", thresh=0),
    "montgomery":   dict(name="lung",         mask_subdirs=["leftMask", "rightMask"],
                         pair_by="stem", thresh=0),
}

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.gif")


def _list_images(d):
    return sorted(sum([glob.glob(os.path.join(d, e)) for e in _IMG_EXTS], []))


def _id_of(path):
    runs = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    return (runs[-1].lstrip("0") or "0") if runs else os.path.basename(path)


def _find_masks(img_path, mask_dirs, pair_by, mask_suffix):
    """Return matching mask paths (one per mask_dir) for an image."""
    istem = os.path.splitext(os.path.basename(img_path))[0]
    iid = _id_of(img_path)
    found = []
    for md in mask_dirs:
        hit = None
        for m in _list_images(md):
            mstem = os.path.splitext(os.path.basename(m))[0]
            if mask_suffix:
                mstem = mstem.replace(mask_suffix, "")
            if (pair_by == "stem" and mstem == istem) or \
               (pair_by == "id" and _id_of(m) == iid):
                hit = m
                break
        if hit:
            found.append(hit)
    return found


def convert_image_dataset(images_dir, mask_root, out_dir, name,
                          mask_subdirs=("masks",), pair_by="stem",
                          mask_suffix=None, thresh=0, split=(0.7, 0.1, 0.2),
                          seed=42):
    """2-D image/mask pairs -> training layout. Image-level split (no patient
    grouping). Multiple mask_subdirs are OR-merged (e.g. Montgomery L/R lung).
    Saved as <stem>_org1_<name>.npy so eval groups them as one target."""
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("Pillow is required: `pip install Pillow`") from e

    imgs = _list_images(images_dir)
    if not imgs:
        raise FileNotFoundError(f"no images under {images_dir}")
    mask_dirs = [os.path.join(mask_root, sd) for sd in mask_subdirs]
    rng = random.Random(seed)
    rng.shuffle(imgs)
    n = len(imgs)
    n_tr, n_va = int(n * split[0]), int(n * split[1])
    for sp in ("train", "val", "test"):
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(out_dir, sp, sub), exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    for i, ip in enumerate(imgs):
        sp = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        masks = _find_masks(ip, mask_dirs, pair_by, mask_suffix)
        if not masks:
            print(f"[skip] no mask for {os.path.basename(ip)}")
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        merged = None
        for mp in masks:
            m = np.array(Image.open(mp).convert("L"))
            merged = m if merged is None else np.maximum(merged, m)
        if merged.shape != img.shape[:2]:
            from PIL import Image as _I
            merged = np.array(_I.fromarray(merged).resize(
                (img.shape[1], img.shape[0]), _I.NEAREST))
        fg = (merged > thresh).astype(np.uint8)
        if fg.sum() == 0:
            continue
        stem = os.path.splitext(os.path.basename(ip))[0].replace(" ", "_")
        base = f"{stem}_org1_{name}"
        np.save(os.path.join(out_dir, sp, "images", base + ".npy"), img.astype(np.uint8))
        np.save(os.path.join(out_dir, sp, "masks", base + ".npy"), fg)
        counts[sp] += 1
    print(f"done: {counts} samples across train/val/test ({n} images)")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=sorted(DATASETS) + sorted(DATASETS_2D))
    ap.add_argument("--root", default=None,
                    help="raw dataset root; auto-detects images/labels subdirs")
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--labels_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_organ_px", type=int, default=None)
    ap.add_argument("--label_map_json", default=None,
                    help="optional JSON {int_label: name} overriding the default")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset in DATASETS_2D:
        c = DATASETS_2D[args.dataset]
        mask_root = args.labels_dir or args.root or args.images_dir
        img_dir = args.images_dir or (os.path.join(args.root, "images") if args.root else None)
        if img_dir is None:
            ap.error("2D dataset: provide --images_dir (and --labels_dir as mask root) or --root")
        convert_image_dataset(img_dir, mask_root, args.out, c["name"],
                              mask_subdirs=c["mask_subdirs"], pair_by=c["pair_by"],
                              mask_suffix=c.get("mask_suffix"), thresh=c["thresh"],
                              seed=args.seed)
        return
    cfg = DATASETS[args.dataset]
    label_map = cfg.get("labels")
    if args.label_map_json:
        label_map = {int(k): v for k, v in json.load(open(args.label_map_json)).items()}
    images_dir, labels_dir = args.images_dir, args.labels_dir
    if images_dir is None or labels_dir is None:
        if args.root is None:
            ap.error("provide either --root, or both --images_dir and --labels_dir")
        images_dir, labels_dir = autodetect_dirs(args.root)
        print(f"autodetected images={images_dir}  labels={labels_dir}")

    convert_nifti_dataset(
        images_dir, labels_dir, args.out, label_map,
        normalize=cfg.get("normalize", "ct_window"), hu=cfg.get("hu", (-160, 240)),
        merge_foreground=cfg.get("merge_foreground", False),
        fg_name=cfg.get("fg_name", "lesion"),
        modality_index=cfg.get("modality_index"),
        min_organ_px=args.min_organ_px if args.min_organ_px is not None
        else cfg.get("min_organ_px", 20),
        seed=args.seed)


if __name__ == "__main__":
    main()
