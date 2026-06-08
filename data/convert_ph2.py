"""Convert the PH2 dermoscopy dataset into a TEST-only {images,masks} split.

PH2 is used here purely as a genuine out-of-distribution probe for a model
trained on ISIC2017 (different hospital/camera, no image overlap). Layout:

    <root>/IMD003/IMD003_Dermoscopic_Image/IMD003.bmp
    <root>/IMD003/IMD003_lesion/IMD003_lesion.bmp

Produces  <out>/test/images/IMD003.png  and  <out>/test/masks/IMD003.png
(matching stems, so MedSegDataset pairs them directly). Point the OOD eval at
<out>:  DATA18=<out>  or  --data <out>.

Usage:
    python data/convert_ph2.py --root "/path/PH2 Dataset images" --out data/processed/ph2
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the 'PH2 Dataset images' directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        raise SystemExit("Pillow + numpy required: pip install Pillow numpy") from e

    img_out = os.path.join(args.out, "test", "images")
    msk_out = os.path.join(args.out, "test", "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    cases = sorted(d for d in glob.glob(os.path.join(args.root, "IMD*"))
                   if os.path.isdir(d))
    if not cases:
        raise SystemExit(f"no IMD* case folders under {args.root}")

    n = 0
    for c in cases:
        cid = os.path.basename(c)
        imgs = glob.glob(os.path.join(c, f"{cid}_Dermoscopic_Image", "*.bmp"))
        msks = glob.glob(os.path.join(c, f"{cid}_lesion", "*.bmp"))
        if not imgs or not msks:
            print(f"  skip {cid}: missing image or lesion mask")
            continue
        img = np.array(Image.open(imgs[0]).convert("RGB"))
        m = np.array(Image.open(msks[0]).convert("L"))
        fg = (m > 0).astype("uint8") * 255          # binarise lesion
        Image.fromarray(img).save(os.path.join(img_out, f"{cid}.png"))
        Image.fromarray(fg).save(os.path.join(msk_out, f"{cid}.png"))
        n += 1
    print(f"PH2 -> {args.out}/test : {n} image/mask pairs")


if __name__ == "__main__":
    main()
