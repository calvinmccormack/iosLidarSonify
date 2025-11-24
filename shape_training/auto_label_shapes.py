#!/usr/bin/env python3
# Auto-label colored shapes into shape masks.
# Usage:
#   python auto_label_shapes.py --in_dir /path/to/photos --out_dir /path/to/out

import cv2, numpy as np, os, argparse, json

# HSV thresholds (OpenCV: H in [0..180], S,V in [0..255])
# Works well for saturated PLA colors under indoor light. Tweak S/V if needed.
R1_L, R1_H = (0,   80, 70),  (10, 255, 255)     # red low band
R2_L, R2_H = (170, 80, 70),  (180,255, 255)     # red high band (wrap)
# Green varies; cover deep + yellowish greens:
G1_L, G1_H = (40,  70, 60),  (85, 255, 255)
G2_L, G2_H = (35,  70, 60),  (55, 255, 255)
B_L,  B_H  = (95,  80, 60),  (135,255, 255)

LABELS = {"red":1, "green":2, "blue":3}

def color_mask(hsv, color):
    if color == "red":
        m1 = cv2.inRange(hsv, np.array(R1_L), np.array(R1_H))
        m2 = cv2.inRange(hsv, np.array(R2_L), np.array(R2_H))
        m = cv2.bitwise_or(m1, m2)
    elif color == "green":
        g1 = cv2.inRange(hsv, np.array(G1_L), np.array(G1_H))
        g2 = cv2.inRange(hsv, np.array(G2_L), np.array(G2_H))
        m = cv2.bitwise_or(g1, g2)
    else:  # blue
        m = cv2.inRange(hsv, np.array(B_L), np.array(B_H))

    # Clean edges; remove specks
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), iterations=1)

    # Keep only components with reasonable area
    num, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    h, w = m.shape[:2]
    min_area = int((h*w) * 0.001)   # 0.1% of image
    max_area = int((h*w) * 0.7)     # ignore full-frame accidents
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if min_area <= a <= max_area:
            out[cc == i] = 255
    return out

def label_image(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_r = color_mask(hsv, "red")
    mask_g = color_mask(hsv, "green")
    mask_b = color_mask(hsv, "blue")

    lab = np.zeros(mask_r.shape, dtype=np.uint8)
    lab[mask_r > 0] = LABELS["red"]
    lab[mask_g > 0] = LABELS["green"]
    lab[mask_b > 0] = LABELS["blue"]

    return lab

def draw_overlay(img, lab):
    overlay = img.copy()
    for val, color in [(1,(0,0,255)), (2,(0,255,0)), (3,(255,0,0))]:
        cnts, _ = cv2.findContours((lab==val).astype(np.uint8)*255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color, 3)
    return overlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "masks_shape"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "overlays"),    exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "grayscale"),   exist_ok=True)

    report = []
    exts = (".jpg",".jpeg",".png",".JPG",".JPEG",".PNG",".heic",".HEIC")
    for name in sorted(os.listdir(args.in_dir)):
        if not name.endswith(exts): 
            continue
        path = os.path.join(args.in_dir, name)
        img  = cv2.imread(path)
        if img is None:
            report.append({"path":path, "status":"read_fail"}); continue

        lab = label_image(img)
        base = os.path.splitext(name)[0]

        mask_path = os.path.join(args.out_dir, "masks_shape", base + "_shape.png")
        ov_path   = os.path.join(args.out_dir, "overlays",    base + "_overlay.jpg")
        gray_path = os.path.join(args.out_dir, "grayscale",   base + "_gray.png")

        cv2.imwrite(mask_path, lab)
        cv2.imwrite(ov_path,   draw_overlay(img, lab))
        cv2.imwrite(gray_path, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        h,w = lab.shape[:2]
        area = float((lab>0).sum())/(h*w)
        report.append({"path":path, "status":"ok", "area_frac":round(area,4),
                       "mask":mask_path, "overlay":ov_path, "gray":gray_path})

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()