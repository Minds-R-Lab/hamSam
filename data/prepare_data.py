"""Dataset download/conversion entry point.

Targets the EXACT eight datasets used by VM-MedSAM (Li et al. 2026, Table 1),
so Ham-MedSAM is trained and evaluated on the same data. Network downloads
can't run in CI; this documents the layout and is where per-dataset converters
go as you obtain access.

Target layout per dataset (binary, box-promptable -- the SAM/MedSAM paradigm:
one (image, box, binary-mask) sample per structure instance; multi-organ CT
volumes are exploded into per-organ binary masks during conversion):

    <out>/<dataset>/{train,val,test}/images/<id>.npy   # HxW or HxWx3
    <out>/<dataset>/{train,val,test}/masks/<id>.npy    # HxW {0,1}

VM-MedSAM uses PATIENT-LEVEL splits for 3D volumes to avoid slice leakage --
replicate that when slicing CT/MRI.
"""
import argparse

# VM-MedSAM Table 1 (the eight datasets, with reported sample sizes).
VM_MEDSAM_DATASETS = {
    "btcv":          ("Abdominal CT",   "13 organs",     "5,000 slices",
                      "Synapse multi-atlas labeling beyond the cranial vault"),
    "flare22":       ("Abdominal CT",   "13 organs",     "5,000 slices",
                      "https://flare22.grand-challenge.org"),
    "msd_lung":      ("Chest CT",       "lung cancer",   "630 slices",
                      "Medical Segmentation Decathlon -- http://medicaldecathlon.com"),
    "brats":         ("MRI (FLAIR)",    "brain tumor",   "4,840 slices",
                      "http://braintumorsegmentation.org"),
    "cvc_clinicdb":  ("Colonoscopy",    "polyp",         "612 images",
                      "https://polyp.grand-challenge.org/CVCClinicDB/"),
    "busi":          ("Ultrasound",     "breast tumor",  "1,312 images",
                      "Breast Ultrasound Images dataset (BUSI)"),
    "drive":         ("Fundus photo",   "retinal vessel","20 images",
                      "https://drive.grand-challenge.org/"),
    "montgomery":    ("Chest X-ray",    "lung",          "138 images",
                      "Montgomery County X-ray set"),
}

# Optional out-of-distribution probes for the zero-shot extension. NOT part of
# VM-MedSAM; used only to test transfer to genuinely unseen domains.
ZERO_SHOT_PROBES = {
    "isic2018": ("Dermoscopy", "skin lesion", "binary", "unseen modality (no dermoscopy in VM-MedSAM)"),
    "tn3k":     ("Thyroid US", "nodule",      "binary", "unseen organ (US present via BUSI/breast)"),
    "acdc":     ("Cardiac MRI","4 structures","multiclass", "unseen organ (MRI present via BraTS/brain)"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=sorted(VM_MEDSAM_DATASETS) + sorted(ZERO_SHOT_PROBES))
    ap.add_argument("--src", required=True, help="path to raw downloaded data")
    ap.add_argument("--out", required=True, help="output root for the split layout")
    args = ap.parse_args()
    info = VM_MEDSAM_DATASETS.get(args.dataset) or ZERO_SHOT_PROBES[args.dataset]
    raise NotImplementedError(
        f"Converter for '{args.dataset}' ({info}) not implemented in this "
        f"environment (requires the raw data). Implement the volume->slice + "
        f"patient-level split + per-organ binary explosion here, writing "
        f"{args.out}/{args.dataset}/<split>/{{images,masks}}/*.npy.")


if __name__ == "__main__":
    main()
