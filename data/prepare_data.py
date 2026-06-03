"""Dataset download/conversion entry point.

Network downloads can't run in CI; this documents the expected layout and
converts source volumes/images into the <root>/<split>/{images,masks}/ layout
that data/datasets.MedSegDataset reads. Fill in per-dataset converters as you
obtain access (FLARE22/BTCV require challenge registration).

Target layout per dataset:
    <out>/<dataset>/{train,val,test}/images/<id>.npy   # HxW or HxWx3
    <out>/<dataset>/{train,val,test}/masks/<id>.npy    # HxW int labels

VM-MedSAM uses patient-level splits to avoid slice leakage -- replicate that
when slicing 3D CT/MRI volumes.
"""
import argparse

DATASETS = {
    "flare22": "Abdomen CT, 13 organs -- https://flare22.grand-challenge.org",
    "btcv": "Abdomen CT, 13 organs -- Synapse multi-atlas labeling",
    "msd_lung": "Chest CT, lung cancer -- http://medicaldecathlon.com",
    "brats21": "Brain MRI (FLAIR) -- http://braintumorsegmentation.org",
    "isic2018": "Dermoscopy, binary lesion (zero-shot)",
    "tn3k": "Thyroid ultrasound, binary (zero-shot)",
    "acdc": "Cardiac MRI, 4-class (zero-shot)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--src", required=True, help="path to raw downloaded data")
    ap.add_argument("--out", required=True, help="output root for the split layout")
    args = ap.parse_args()
    raise NotImplementedError(
        f"Converter for {args.dataset} not implemented in this environment "
        f"(requires the raw data: {DATASETS[args.dataset]}). Implement the "
        f"volume->slice + patient-level split here, writing {args.out}/{args.dataset}"
        f"/<split>/{{images,masks}}/*.npy.")


if __name__ == "__main__":
    main()
