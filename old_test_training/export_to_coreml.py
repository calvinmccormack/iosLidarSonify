import torch
import coremltools as ct

from model import MiniUNet

# 1. Load trained PyTorch model (6 classes, same as training)
num_classes = 6
model = MiniUNet(num_classes=num_classes)
checkpoint = torch.load("checkpoint.pth.tar", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. Dummy input matching your training size
dummy = torch.zeros(1, 3, 240, 320)  # [N,C,H,W]; adjust H,W if your dataset differs

# 3. Trace the model
traced = torch.jit.trace(model, dummy)

# 4. Convert to CoreML
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.ImageType(
            name="input",
            shape=dummy.shape,      # (1,3,240,320)
            scale=1.0 / 255.0,      # basic 0–255 → 0–1 scaling
            # if you normalize with mean/std in dataset.py,
            # you can also bake that in here later
        )
    ],
)

mlmodel.save("MiniUNetHW6.mlpackage")
print("Saved MiniUNetHW6.mlpackage")