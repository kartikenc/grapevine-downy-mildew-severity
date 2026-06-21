#!/usr/bin/env python3
"""
Script 04: Leaf Segmentation from Natural Backgrounds using U-Net
=================================================================
PhD Thesis: Plant Disease Severity Estimation
Paper 3: Automated Leaf Segmentation and Disease Severity Estimation

Pipeline:
  1. Convert RectLabel XML polygon annotations -> binary masks
  2. Split into train/val sets (80/20)
  3. Train U-Net (ResNet-34 encoder) for leaf vs background segmentation
  4. Evaluate with mIoU, Dice, pixel accuracy
  5. Generate leaf masks for ALL images (including unlabeled)

Author: Kartik E. Cholachgudda (R18PEC20)
"""

import os
import sys
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
ANNOTATED_DIR = Path(r"d:\Projects\AgRECA\PhD\PhD2\04_Dataset\Annotated\2")
DATASET_ROOT = Path(r"d:\Projects\AgRECA\PhD\PhD2\04_Dataset\Downy_Mildew\Original")
RESULTS_DIR = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper3_segmentation")
IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10
RANDOM_SEED = 42
VAL_SPLIT = 0.2

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# Step 1: Convert XML Polygon Annotations to Binary Masks
# ============================================================
def parse_rectlabel_xml(xml_path):
    """Parse RectLabel XML annotation and extract polygon points."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    filename = root.find('filename').text
    w = int(root.find('size/width').text)
    h = int(root.find('size/height').text)

    polygons = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        polygon_elem = obj.find('polygon')
        if polygon_elem is not None:
            # Extract x,y pairs from polygon children
            coords = {}
            for child in polygon_elem:
                coords[child.tag] = float(child.text)

            # Group into (x, y) pairs
            points = []
            i = 1
            while f'x{i}' in coords and f'y{i}' in coords:
                points.append([coords[f'x{i}'], coords[f'y{i}']])
                i += 1

            if points:
                polygons.append({
                    'label': label,
                    'points': np.array(points, dtype=np.float32)
                })

    return filename, w, h, polygons


def create_binary_mask(width, height, polygons):
    """Create a binary mask from polygon annotations."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = poly['points'].astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask


def convert_all_annotations():
    """Convert all XML annotations to binary masks and save them."""
    mask_dir = RESULTS_DIR / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ANNOTATED_DIR.glob("*.xml"))
    print(f"\nConverting {len(xml_files)} XML annotations to binary masks...")

    converted = []
    for xml_path in tqdm(xml_files, desc="Converting"):
        filename, w, h, polygons = parse_rectlabel_xml(xml_path)
        if polygons:
            mask = create_binary_mask(w, h, polygons)
            img_path = ANNOTATED_DIR / filename
            if img_path.exists():
                mask_path = mask_dir / f"{Path(filename).stem}_mask.png"
                cv2.imwrite(str(mask_path), mask * 255)
                converted.append({
                    'image': str(img_path),
                    'mask': str(mask_path),
                    'width': w,
                    'height': h,
                    'num_polygons': len(polygons)
                })

    print(f"  Converted {len(converted)} annotations to masks")
    return converted


# ============================================================
# Step 2: Dataset and DataLoader
# ============================================================
class LeafSegDataset(Dataset):
    """Dataset for leaf segmentation."""

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else torch.tensor(mask).unsqueeze(0)
        return image, mask


def get_transforms(train=True):
    """Get augmentation transforms."""
    if train:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


# ============================================================
# Step 3: U-Net Model (ResNet-34 Encoder)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetResNet34(nn.Module):
    """U-Net with ResNet-34 encoder (pretrained)."""

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)

        # Encoder
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64, /2
        self.pool1 = resnet.maxpool  # /4
        self.enc2 = resnet.layer1  # 64, /4
        self.enc3 = resnet.layer2  # 128, /8
        self.enc4 = resnet.layer3  # 256, /16
        self.enc5 = resnet.layer4  # 512, /32

        # Decoder
        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec5 = ConvBlock(512, 256)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = ConvBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(32, 32)

        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # [B, 64, H/2, W/2]
        e2 = self.enc2(self.pool1(e1))  # [B, 64, H/4, W/4]
        e3 = self.enc3(e2)      # [B, 128, H/8, W/8]
        e4 = self.enc4(e3)      # [B, 256, H/16, W/16]
        e5 = self.enc5(e4)      # [B, 512, H/32, W/32]

        # Decoder with skip connections
        d5 = self.up5(e5)
        d5 = self.dec5(torch.cat([d5, e4], dim=1))
        d4 = self.up4(d5)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# Step 4: Loss Function (Dice + BCE)
# ============================================================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sig = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()

        return bce_loss + dice_loss


# ============================================================
# Step 5: Training Loop
# ============================================================
def compute_metrics(pred, target):
    """Compute segmentation metrics."""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    smooth = 1e-6

    intersection = (pred_bin * target).sum(dim=(2, 3))
    union = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)

    dice = (2 * intersection + smooth) / (pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)

    correct = (pred_bin == target).sum()
    total = target.numel()
    pixel_acc = correct / total

    return iou.mean().item(), dice.mean().item(), pixel_acc.item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        iou, dice, _ = compute_metrics(outputs, masks)
        total_loss += loss.item()
        total_iou += iou
        total_dice += dice

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_pacc = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            iou, dice, pacc = compute_metrics(outputs, masks)
            total_loss += loss.item()
            total_iou += iou
            total_dice += dice
            total_pacc += pacc

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n, total_pacc / n


# ============================================================
# Step 6: Visualization
# ============================================================
def visualize_predictions(model, dataset, device, save_path, num_samples=6):
    """Generate visualization of segmentation predictions."""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 2].set_title('Prediction', fontsize=14, fontweight='bold')

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()

        # Denormalize image
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean).clip(0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mask.squeeze().numpy(), cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(pred > 0.5, cmap='gray')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved predictions: {save_path}")


def plot_training_history(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_iou'], label='Train', linewidth=2)
    axes[1].plot(history['val_iou'], label='Val', linewidth=2)
    axes[1].set_title('IoU', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['train_dice'], label='Train', linewidth=2)
    axes[2].plot(history['val_dice'], label='Val', linewidth=2)
    axes[2].set_title('Dice Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves: {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("PAPER 3: LEAF SEGMENTATION PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert annotations
    annotations = convert_all_annotations()

    # Step 2: Split into train/val
    random.shuffle(annotations)
    split_idx = int(len(annotations) * (1 - VAL_SPLIT))
    train_ann = annotations[:split_idx]
    val_ann = annotations[split_idx:]

    print(f"\n  Train: {len(train_ann)} images")
    print(f"  Val:   {len(val_ann)} images")

    train_images = [a['image'] for a in train_ann]
    train_masks = [a['mask'] for a in train_ann]
    val_images = [a['image'] for a in val_ann]
    val_masks = [a['mask'] for a in val_ann]

    train_dataset = LeafSegDataset(train_images, train_masks, get_transforms(train=True))
    val_dataset = LeafSegDataset(val_images, val_masks, get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Step 3: Build model
    print("\n" + "=" * 60)
    print("Training U-Net (ResNet-34 encoder)")
    print("=" * 60)

    model = UNetResNet34(num_classes=1, pretrained=True).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Step 4: Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [],
               'train_dice': [], 'val_dice': []}
    best_val_dice = 0
    patience_counter = 0
    start_time = datetime.now()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_iou, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_iou, val_dice, val_pacc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)

        if epoch == 1 or epoch % 5 == 0 or epoch == NUM_EPOCHS:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} Dice: {train_dice:.4f} | "
                  f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} Dice: {val_dice:.4f} PxAcc: {val_pacc:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), RESULTS_DIR / "best_unet_resnet34.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_time = (datetime.now() - start_time).total_seconds()
    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Best val Dice: {best_val_dice:.4f}")

    # Load best model
    model.load_state_dict(torch.load(RESULTS_DIR / "best_unet_resnet34.pt", weights_only=True))

    # Step 5: Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    val_loss, val_iou, val_dice, val_pacc = validate(model, val_loader, criterion, DEVICE)
    print(f"  Val mIoU:       {val_iou:.4f}")
    print(f"  Val Dice:       {val_dice:.4f}")
    print(f"  Val Pixel Acc:  {val_pacc:.4f}")

    # Step 6: Visualizations
    print("\nGenerating visualizations...")
    visualize_predictions(model, val_dataset, DEVICE, RESULTS_DIR / "segmentation_predictions.png")
    plot_training_history(history, RESULTS_DIR / "segmentation_training_curves.png")

    # Save metrics
    metrics = {
        'model': 'U-Net (ResNet-34)',
        'image_size': IMAGE_SIZE,
        'num_train': len(train_ann),
        'num_val': len(val_ann),
        'epochs_trained': len(history['train_loss']),
        'best_val_dice': float(best_val_dice),
        'val_miou': float(val_iou),
        'val_pixel_accuracy': float(val_pacc),
        'train_time_seconds': train_time,
        'total_parameters': total_params,
        'timestamp': datetime.now().isoformat()
    }
    with open(RESULTS_DIR / "segmentation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}")
    print("=" * 60)
    print("SEGMENTATION PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
