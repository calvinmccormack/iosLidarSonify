#!/usr/bin/env python3
# Create train/val/test splits from your auto-labeled outputs.
# Usage:
#   python split_dataset.py \
#     --images_dir segmented_photos/grayscale \
#     --masks_dir  segmented_photos/masks_shape \
#     --out_dir    dataset --val 0.15 --test 0.15
import argparse, random, shutil
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)  # folder with *_gray.png
    ap.add_argument("--masks_dir",  required=True)  # folder with *_shape.(png|jpg|jpeg)
    ap.add_argument("--out_dir",    required=True)  # output dataset root
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = sorted(Path(args.images_dir).glob("*.png"))
    masks  = sorted(list(Path(args.masks_dir).glob("*.png")) +
                    list(Path(args.masks_dir).glob("*.jpg")) +
                    list(Path(args.masks_dir).glob("*.jpeg")))

    # Map mask stems (strip "_shape") -> mask path
    mask_by_stem = {p.stem.replace("_shape",""): p for p in masks}

    pairs = []
    for img in images:
        stem = img.stem.replace("_gray","")
        m = mask_by_stem.get(stem)
        if m is not None:
            pairs.append((img, m))

    if not pairs:
        raise SystemExit("No (image,mask) pairs found. Check filenames/paths.")

    random.seed(args.seed)
    random.shuffle(pairs)
    n = len(pairs); n_val = int(args.val*n); n_test = int(args.test*n)
    splits = {
        "train": pairs[:n-n_val-n_test],
        "val":   pairs[n-n_val-n_test:n-n_test],
        "test":  pairs[n-n_test:]
    }

    out = Path(args.out_dir)
    for s in splits:
        (out/f"images/{s}").mkdir(parents=True, exist_ok=True)
        (out/f"labels/{s}").mkdir(parents=True, exist_ok=True)

    for s, items in splits.items():
        for img, mask in items:
            dst_img  = out/f"images/{s}/{img.stem}.png"
            dst_mask = out/f"labels/{s}/{img.stem.replace('_gray','')}.png"
            shutil.copy2(img, dst_img)
            m = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
            if m is None:
                raise SystemExit(f"Failed to read mask: {mask}")
            cv2.imwrite(str(dst_mask), m)  # force PNG so label IDs are preserved

    print("Done:", {k: len(v) for k,v in splits.items()})
    print("Dataset root:", out.resolve())

if __name__ == "__main__":
    main()