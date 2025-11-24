#!/usr/bin/env python3
# Train a small U-Net (MobileNetV3 encoder) on grayscale->multiclass masks.
# Install in your conda env:
#   conda install -c conda-forge numpy opencv albumentations timm -y
#   conda install -c pytorch pytorch torchvision -y
#   pip install segmentation-models-pytorch
#
# Run:
#   python train_unet_smp.py --data dataset --epochs 40 --img_w 640 --img_h 360
import argparse, os, cv2, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class SegDS(Dataset):
    def __init__(self, root, split, img_w, img_h, aug=False):
        self.xr = Path(root)/f"images/{split}"
        self.yr = Path(root)/f"labels/{split}"
        self.names = sorted([p.name for p in self.xr.glob("*.png")])
        t = [A.Resize(img_h, img_w, interpolation=cv2.INTER_LINEAR)]
        if aug:
            t += [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=7,
                                   border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            ]
        t += [ToTensorV2()]
        self.tf = A.Compose(t)

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        x = cv2.imread(str(self.xr/name), cv2.IMREAD_GRAYSCALE)            # (H,W)
        label_name = name.replace('_gray','')  # keeps the .png extension
        y_path = self.yr / label_name
        y = cv2.imread(str(y_path), cv2.IMREAD_UNCHANGED)
        if x is None:
            raise FileNotFoundError(f"Missing image: {self.xr / name}")
        if y is None:
            raise FileNotFoundError(f"Missing label: {y_path}")
        if y.ndim == 3: y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
        x = x[...,None]                                                    # (H,W,1)
        s = self.tf(image=x, mask=y)
        x_t = (s["image"].float()/255.0).repeat(3,1,1)                     # (3,H,W)
        y_t = s["mask"].long()                                             # (H,W)
        return x_t, y_t

def miou(cm):
    ious=[]
    for c in range(cm.shape[0]):
        inter = cm[c,c]
        denom = cm[c,:].sum() + cm[:,c].sum() - inter + 1e-9
        ious.append(float(inter/denom))
    return float(np.mean(ious)), [float(i) for i in ious]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--img_h", type=int, default=360)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/shape_unet")
    args = ap.parse_args()

    train_dl = DataLoader(SegDS(args.data,"train",args.img_w,args.img_h,aug=True),
                          batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(SegDS(args.data,"val",args.img_w,args.img_h,aug=False),
                          batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    model = smp.Unet(encoder_name="mobilenet_v2",
                     encoder_weights="imagenet", in_channels=3, classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)

    ce   = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2,1.0,1.0,1.0], device=device))
    dice = smp.losses.DiceLoss(mode="multiclass")
    def loss_fn(p,t): return 0.7*ce(p,t) + 0.3*dice(p,t)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    best = -1.0

    for ep in range(1, args.epochs+1):
        model.train(); tr=0.0
        for xb,yb in train_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); pred = model(xb); L = loss_fn(pred,yb)
            L.backward(); opt.step(); tr += float(L.item())

        model.eval(); cm = np.zeros((4,4), dtype=np.int64)
        with torch.no_grad():
            for xb,yb in val_dl:
                xb = xb.to(device)
                p = model(xb).argmax(1).cpu().numpy()
                y = yb.numpy()
                for ct in range(4):
                    for cp in range(4):
                        cm[ct,cp] += np.sum((y==ct)&(p==cp))
        m, per = miou(cm)
        print(f"Epoch {ep:03d} | train_loss={tr/len(train_dl):.4f} | val_mIoU={m:.3f} | IoU={per}")
        if m > best:
            best = m
            outdir = Path(args.out)
            outdir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "state_dict": model.state_dict(),
                "img_size": (args.img_h, args.img_w),
                "epoch": ep,
                "val_mIoU": m,
            }
            best_path = outdir / "best.pt"
            torch.save(ckpt, str(best_path))
            ver_path = outdir / f"best_ep{ep:03d}.pt"
            torch.save(ckpt, str(ver_path))
            print("  saved", best_path, "and", ver_path)

        # also save a rolling checkpoint each epoch
        last_ckpt = {
            "state_dict": model.state_dict(),
            "img_size": (args.img_h, args.img_w),
            "epoch": ep,
            "val_mIoU": m,
        }
        torch.save(last_ckpt, str(Path(args.out)/"last.pt"))

    print("Done. Best mIoU:", best)

if __name__ == "__main__":
    main()