import torch, coremltools as ct
import segmentation_models_pytorch as smp
from pathlib import Path

H, W = 360, 640

# --- Resolve checkpoint robustly (handles moved folders) ---
SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATES = [
    SCRIPT_DIR / "runs/shape_unet/best.pt",                                         # new layout: iosLidarSonify/shape_training/runs/...
    Path.cwd() / "runs/shape_unet/best.pt",                                         # current working dir
    (SCRIPT_DIR.parents[2] / "shape_training" / "runs" / "shape_unet" / "best.pt")  # old layout: depthSonify/shape_training/runs/...
]
CKPT_PATH = None
for p in CANDIDATES:
    if p.exists():
        CKPT_PATH = p
        break
if CKPT_PATH is None:
    msg = (
        "Could not find checkpoint 'best.pt'.\n"
        f"Tried:\n - {CANDIDATES[0]}\n - {CANDIDATES[1]}\n - {CANDIDATES[2]}\n"
        "Fix: move best.pt to shape_training/runs/shape_unet/ or edit this script to point to it."
    )
    raise FileNotFoundError(msg)
print(f"[export] Using checkpoint: {CKPT_PATH}")

# Build the same model that trained
model = smp.Unet(encoder_name="mobilenet_v2",
                 encoder_weights=None,
                 classes=4, in_channels=3)
ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
model.load_state_dict(ckpt["state_dict"])  # expects the same head (#classes=4)
model.eval()

# Dummy 3-channel input (you trained with grayscale replicated to 3 channels)
example = torch.randn(1, 3, H, W)
traced = torch.jit.trace(model, example)

# Export as ML Program with correct image preprocessing
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="image",
        shape=(1, 3, H, W),            # NCHW
        scale=1/255.0,                 # normalize to 0..1
        bias=[0.0, 0.0, 0.0],          # no mean subtraction
        color_layout="RGB"             # Vision will handle BGRA->RGB for ARKit buffers
    )],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS16
)

out_path = SCRIPT_DIR / "ShapeSeg_unet_mbv2_best.mlpackage"
mlmodel.save(str(out_path))
print(f"Saved: {out_path}")