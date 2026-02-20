#Reference of pre-trained model being used (MiDaS) - https://github.com/isl-org/MiDaS

# Installing pytorch, timm for MiDas model, opencv, matplotlib for graphs
#!pip install -q torch torchvision timm opencv-python matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random                                     # Introduce a random seed for consistency
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Loading a MiDas_small model, transforms for pseudo labels in training the model with augumentations
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
tfm   = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Creating a dataset with different augumentations of the given image.
# Length of dataset = 300 and size of each image in dataset is 256*256.
class PseudoDepthDataset(Dataset):
    def __init__(self, pil_img, length=300, target_size=(256, 256)):
        self.base = pil_img
        self.length = length
        self.target_sz = target_size

        self.aug = T.Compose([                         # Added parameter p for randomness
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.RandomRotation(10, fill=(0, 0, 0)),
            T.ColorJitter(0.3, 0.3, 0.2, 0.2),
            T.Resize(self.target_sz, interpolation=Image.BILINEAR)
        ])
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        aug_pil = self.aug(self.base)
        img_t = self.to_tensor(aug_pil)

        np_rgb = np.array(aug_pil)
        np_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

        batch = tfm(np_bgr).to(device)
        with torch.no_grad():
            depth = midas(batch)

        depth = depth.squeeze(1).squeeze(0).cpu()
        return img_t, depth.unsqueeze(0)           # Input image (Tensor), Target Depth map from MiDas (using this as label)

# Loading input image and creating a dataset with 300 augmentations.
base_img = Image.open("/content/jcsmr.jpg").convert("RGB")
dataset = PseudoDepthDataset(base_img, length=300)

# Dividing the dataset into training, validation sets in ration of 80:20.
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

# Creating a student U-Net architecture
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

# Introducing encoders (to compress input), decoders (upsample back to depth map size)
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 32)     # 2 convolution layers
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))  # 2 convolution layers
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU())  # 1 deconvolution layer
        self.out  = nn.Conv2d(32, 1, 3, padding=1)  # 1 convolution layer

    def forward(self, x):
        x1 = self.enc1(x)       # First encoder block
        x2 = self.enc2(x1)      # Second encoder block, with downsampling
        y  = self.dec1(x2)      # Decoder
        y  = y + x1             # Skip connection: add encoder output to decoder
        return self.out(y)      # Final output convolution layer, 1-channel depth output

# Computing MSE, edge_smooth loss functions for training
# Calculates MSE loss at full, half, quarter size resolutions.
def multiscale_loss(pred, target, base_loss):
    loss = base_loss(pred, target)
    for scale in [2, 4]:
        p = F.interpolate(pred, scale_factor=1/scale, mode='bilinear', align_corners=False)
        t = F.interpolate(target, scale_factor=1/scale, mode='bilinear', align_corners=False)
        loss += base_loss(p, t)
    return loss

# edge-aware smoothness loss to enforce spatial consistency, preserves edges.
def edge_smooth(pred, img):
    dy   = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx   = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    wgty = torch.exp(-torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), 1, keepdim=True))
    wgtx = torch.exp(-torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), 1, keepdim=True))
    return (dx * wgtx).mean() + (dy * wgty).mean()

# Training the model with training, validation sets as per the epochs
student = StudentNet().to(device)
opt = optim.Adam(student.parameters(), lr=1e-4)    # Learning rate, using adam optimiser to update weights
mse = nn.MSELoss()
epochs = 7

for ep in range(epochs):
    student.train()
    train_loss = 0.0
    for imgs, depths in train_dl:
        imgs, depths = imgs.to(device), depths.to(device)
        preds = student(imgs)
        loss = multiscale_loss(preds, depths, mse) + edge_smooth(preds, imgs) * 0.1     # Using both MSE loss and edge_smooth loss

        opt.zero_grad()    # Resets the gradients of all model parameters to zero.
        loss.backward()    # Computes the gradients of the loss function with respect to each model parameter
        opt.step()         # Updates weights
        train_loss += loss.item()

    student.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, depths in val_dl:
            imgs, depths = imgs.to(device), depths.to(device)
            preds = student(imgs)
            loss = multiscale_loss(preds, depths, mse) + edge_smooth(preds, imgs) * 0.1
            val_loss += loss.item()

    print(f"Epoch {ep+1}/{epochs} | Train Loss: {train_loss/len(train_dl):.4f} | Val Loss: {val_loss/len(val_dl):.4f}")

# TTA (Test-Time Augmentation)
# Flips the image horizontally, predicts again and averages both predictions.
# To reduce prediction noise and improve generalization.
def predict_with_tta(model, img):
    p1 = model(img)
    img_flipped = torch.flip(img, dims=[-1])
    p2 = model(img_flipped)
    p2 = torch.flip(p2, dims=[-1])
    return (p1 + p2) / 2

student.eval()
with torch.no_grad():
    resized_base = base_img.resize((256, 256))
    inp = T.ToTensor()(resized_base).unsqueeze(0).to(device)

    # Student prediction
    outd = predict_with_tta(student, inp)[0, 0].cpu().numpy()

    # Teacher prediction
    np_bgr = cv2.cvtColor(np.array(resized_base), cv2.COLOR_RGB2BGR)
    teacher_depth = midas(tfm(np_bgr).to(device)).squeeze().cpu().numpy()

# Comparing both teacher and student model outputs
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(teacher_depth, cmap="plasma")
plt.title("Teacher (MiDaS) Depth")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(outd, cmap="plasma")
plt.title("Student Predicted Depth")
plt.axis("off")

plt.tight_layout()
plt.show()

# Computing metrics RMSE (Root Mean Square Error), MAE (Mean Absolute Error), delta
# RMSE = sqrt(mean((pred - target)^2))
# MAE = mean(abs(pred - target))
# δ accuracy: % of pixels where prediction is within a factor t of the teacher depth
# δ = max(pred / target, target / pred)
# delta_acc = mean(delta < threshold) for t in [1.25, 1.25^2, 1.25^3]
def compute_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    pred = np.clip(pred, 1e-6, None)
    target = np.clip(target, 1e-6, None)

    abs_diff = np.abs(pred - target)
    sq_diff = (pred - target) ** 2

    rmse = np.sqrt(np.mean(sq_diff))
    mae = np.mean(abs_diff)

    ratio = np.maximum(pred / target, target / pred)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "δ < 1.25": delta1,
        "δ < 1.25²": delta2,
        "δ < 1.25³": delta3
    }

metrics = compute_metrics(outd, teacher_depth)
print("\n--- Evaluation Metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")