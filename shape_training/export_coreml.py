# export_coreml.py
# Patch for NumPy 2.0 + coremltools: provide np.issctype before importing coremltools.
import numpy as np
if not hasattr(np, "issctype"):
    def _issctype(rep):
        try:
            return issubclass(rep, np.generic)
        except TypeError:
            return False
    np.issctype = _issctype  # coremltools expects this symbol

# NumPy 2.0 dropped np.issubclass_. CoreML expects it; alias to builtin.
if not hasattr(np, "issubclass_"):
    np.issubclass_ = issubclass

import torch
import segmentation_models_pytorch as smp
import coremltools as ct
from pathlib import Path

# === Paths ===
CKPT = Path("runs/shape_unet/pinned_best.pt")
if not CKPT.exists():
    CKPT = Path("runs/shape_unet/best.pt")  # fallback

# === Load checkpoint ===
ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=True)
H, W = ckpt.get("img_size", (360, 640))  # (height, width) saved during training

# === Build the same network you trained ===
model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=3,
    classes=4,
)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# === TorchScript trace ===
example = torch.randn(1, 3, H, W)
traced = torch.jit.trace(model, example)
traced.eval()

# === Convert to Core ML ===
mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[ct.ImageType(name="input",
                         shape=example.shape,
                         color_layout=ct.colorlayout.RGB)],
    compute_precision=ct.precision.FLOAT16,  # use FP32 if you prefer
)

# Optional metadata for downstream use
mlmodel.user_defined_metadata.update({
    "classes": "0:background,1:sphere,2:triangle,3:cube",
    "img_size": f"{H}x{W}",
    "encoder": "mobilenet_v2",
})

out_path = Path("ShapeSeg_unet_mbv2_best.mlpackage")  # use .mlpackage for ML Program
mlmodel.save(str(out_path))
print(f"Saved {out_path.resolve()}")