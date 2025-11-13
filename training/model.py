import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    def __init__(self, num_classes: int = 6):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()

        # convolve
        self.conv1a = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2a = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # bottleneck
        self.conv5a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5b = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # decode
        self.dec4a = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)
        self.dec4b = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.dec3a = nn.Conv2d(128+64, 64, kernel_size=3, padding=1)
        self.dec3b = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.dec2a = nn.Conv2d(64+32, 32, kernel_size=3, padding=1)
        self.dec2b = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.dec1a = nn.Conv2d(32+16, 16, kernel_size=3, padding=1)
        self.dec1b = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.conv1x1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1a(x));
        x1 = F.relu(self.conv1b(x1));
        p1 = self.pool1(x1)
        x2 = F.relu(self.conv2a(p1));
        x2 = F.relu(self.conv2b(x2));
        p2 = self.pool2(x2)
        x3 = F.relu(self.conv3a(p2));
        x3 = F.relu(self.conv3b(x3));
        p3 = self.pool3(x3)
        x4 = F.relu(self.conv4a(p3));
        x4 = F.relu(self.conv4b(x4));
        p4 = self.pool4(x4)

        # Bottleneck
        b = F.relu(self.conv5a(p4))
        b = F.relu(self.conv5b(b))

        # Decoder
        u4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = torch.cat([u4, x4], dim=1)
        d4 = F.relu(self.dec4a(d4));
        d4 = F.relu(self.dec4b(d4))

        u3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([u3, x3], dim=1)
        d3 = F.relu(self.dec3a(d3));
        d3 = F.relu(self.dec3b(d3))

        u2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([u2, x2], dim=1)
        d2 = F.relu(self.dec2a(d2));
        d2 = F.relu(self.dec2b(d2))

        u1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([u1, x1], dim=1)
        d1 = F.relu(self.dec1a(d1));
        d1 = F.relu(self.dec1b(d1))

        return self.conv1x1(d1)


if __name__ == '__main__':
    model = MiniUNet(num_classes=6)
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
