#!/usr/bin/env python3
"""
Script 05: HSV-Based Disease Severity Estimation on Segmented Leaves
=====================================================================
PhD Thesis: Plant Disease Severity Estimation
Paper 3: Automated Leaf Segmentation and Disease Severity Estimation

Two-Stage Pipeline:
  Stage 1: U-Net leaf segmentation (from Script 04) -> binary leaf mask
  Stage 2: HSV color thresholding on segmented leaf ->
           healthy (green) vs diseased (brown/yellow/necrotic) pixels ->
           severity ratio -> severity class (S0-S4)

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
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (confusion_matrix, classification_report,
                             cohen_kappa_score, ConfusionMatrixDisplay)
from scipy import stats
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
ORIGINAL_DIR = Path(r"d:\Projects\AgRECA\PhD\PhD2\04_Dataset\Downy_Mildew\Original")
SEG_MODEL_PATH = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper3_segmentation\best_unet_resnet34.pt")
RESULTS_DIR = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper3_severity")
IMAGE_SIZE = 512
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping: folder name -> severity label index
CLASS_MAP = {'O_0': 0, 'O_1': 1, 'O_2': 2, 'O_3': 3, 'O_4': 4}
CLASS_NAMES = ['S0', 'S1', 'S2', 'S3', 'S4']

# HSV thresholds (OpenCV: H 0-180, S 0-255, V 0-255)
# Healthy (green) pixels
GREEN_H_LOW, GREEN_H_HIGH = 25, 85
GREEN_S_MIN = 30
GREEN_V_MIN = 30

# Diseased (yellow/brown/necrotic) pixels
BROWN_H_LOW, BROWN_H_HIGH = 0, 25  # yellow-brown range
BROWN_S_MIN = 20
BROWN_V_MIN = 30

# Necrotic (very desaturated dark tissue)
NECROTIC_S_MAX = 40
NECROTIC_V_LOW, NECROTIC_V_HIGH = 25, 120

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")

# ============================================================
# U-Net Model (same architecture as Script 04)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNetResNet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet34(weights=None)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool1 = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d5 = self.dec5(torch.cat([self.up5(e5), e4], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(self.up1(d2))
        return self.final(d1)


# ============================================================
# Core Functions
# ============================================================
def load_segmentation_model():
    """Load the trained U-Net segmentation model."""
    model = UNetResNet34(num_classes=1, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(SEG_MODEL_PATH, weights_only=True, map_location=DEVICE))
    model.eval()
    print(f"  Loaded segmentation model from {SEG_MODEL_PATH.name}")
    return model


def segment_leaf(model, image_bgr):
    """Run U-Net to get leaf binary mask."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    augmented = transform(image=img_rgb)
    tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        mask = (torch.sigmoid(pred) > 0.5).float().cpu().squeeze().numpy()

    # Resize mask back to original image size
    h, w = image_bgr.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_resized.astype(np.uint8)


def estimate_severity_hsv(image_bgr, leaf_mask):
    """
    Estimate disease severity using HSV color analysis on the segmented leaf.

    Returns:
        severity_pct: percentage of diseased pixels (0-100)
        color_map: 3-class image (0=background, 1=healthy, 2=diseased)
        stats_dict: detailed pixel statistics
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Only consider pixels inside the leaf mask
    leaf_pixels = leaf_mask > 0
    total_leaf = np.sum(leaf_pixels)

    if total_leaf == 0:
        return 0.0, np.zeros_like(leaf_mask), {'total_leaf': 0, 'healthy': 0, 'diseased': 0}

    # Healthy (green) pixels within leaf
    healthy_mask = (leaf_pixels &
                    (h >= GREEN_H_LOW) & (h <= GREEN_H_HIGH) &
                    (s >= GREEN_S_MIN) & (v >= GREEN_V_MIN))

    # Diseased: brown/yellow pixels within leaf
    brown_mask = (leaf_pixels &
                  (((h < BROWN_H_HIGH) & (h >= BROWN_H_LOW)) | (h > 150)) &
                  (s >= BROWN_S_MIN) & (v >= BROWN_V_MIN))

    # Necrotic: dark, desaturated pixels within leaf (dead tissue)
    necrotic_mask = (leaf_pixels &
                     (s < NECROTIC_S_MAX) &
                     (v >= NECROTIC_V_LOW) & (v <= NECROTIC_V_HIGH))

    # Combined diseased
    diseased_mask = brown_mask | necrotic_mask

    # Unclassified leaf pixels (neither clearly healthy nor diseased)
    # Assign based on proximity to green vs brown in hue space
    unclassified = leaf_pixels & (~healthy_mask) & (~diseased_mask)
    # For unclassified, use a softer threshold
    unclass_green = unclassified & (h >= 20) & (h <= 90)
    unclass_diseased = unclassified & (~unclass_green)

    healthy_total = np.sum(healthy_mask) + np.sum(unclass_green)
    diseased_total = np.sum(diseased_mask) + np.sum(unclass_diseased)

    severity_pct = (diseased_total / total_leaf) * 100

    # Create 3-class color map for visualization
    color_map = np.zeros_like(leaf_mask, dtype=np.uint8)
    color_map[healthy_mask | unclass_green] = 1  # healthy
    color_map[diseased_mask | unclass_diseased] = 2  # diseased

    stats_dict = {
        'total_leaf': int(total_leaf),
        'healthy': int(healthy_total),
        'diseased': int(diseased_total),
        'severity_pct': float(severity_pct),
    }
    return severity_pct, color_map, stats_dict


def severity_to_class(severity_pct):
    """Map severity percentage to class label (S0-S4).

    Thresholds based on standard phytopathological severity scales:
      S0: 0-5%    (healthy or trace)
      S1: 5-15%   (slight infection)
      S2: 15-35%  (moderate infection)
      S3: 35-60%  (severe infection)
      S4: >60%    (very severe / nearly dead)
    """
    if severity_pct < 5:
        return 0
    elif severity_pct < 15:
        return 1
    elif severity_pct < 35:
        return 2
    elif severity_pct < 60:
        return 3
    else:
        return 4


# ============================================================
# Visualization Functions
# ============================================================
def create_severity_visualization(image_bgr, leaf_mask, color_map, severity_pct,
                                   pred_class, true_class, save_path):
    """Create a 4-panel visualization of the severity estimation."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original image
    axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Leaf mask
    axes[1].imshow(leaf_mask, cmap='gray')
    axes[1].set_title('Leaf Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Color-coded severity map
    cmap = ListedColormap(['black', '#2ecc71', '#e74c3c'])  # bg, green, red
    axes[2].imshow(color_map, cmap=cmap, vmin=0, vmax=2)
    axes[2].set_title('Severity Map\n(Green=Healthy, Red=Diseased)',
                      fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # Panel 4: Overlay on original
    overlay = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
    overlay[color_map == 1] = (overlay[color_map == 1] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
    overlay[color_map == 2] = (overlay[color_map == 2] * 0.5 + np.array([200, 0, 0]) * 0.5).astype(np.uint8)
    axes[3].imshow(overlay)
    match = "CORRECT" if pred_class == true_class else "WRONG"
    color = '#2ecc71' if pred_class == true_class else '#e74c3c'
    axes[3].set_title(f'Severity: {severity_pct:.1f}%\n'
                      f'Pred: {CLASS_NAMES[pred_class]} | True: {CLASS_NAMES[true_class]} ({match})',
                      fontsize=11, fontweight='bold', color=color)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_severity_distribution(results, save_path):
    """Plot severity percentage distribution per true class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data_by_class = {c: [] for c in range(5)}
    for r in results:
        data_by_class[r['true_class']].append(r['severity_pct'])

    positions = list(range(5))
    box_data = [data_by_class[c] for c in range(5)]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)

    colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Draw threshold lines
    thresholds = [5, 15, 35, 60]
    for t in thresholds:
        ax.axhline(y=t, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(4.6, t, f'{t}%', va='center', fontsize=9, color='gray')

    ax.set_xticklabels(CLASS_NAMES, fontsize=12, fontweight='bold')
    ax.set_xlabel('True Severity Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Estimated Severity (%)', fontsize=13, fontweight='bold')
    ax.set_title('HSV-Based Severity Distribution by Class', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved severity distribution: {save_path.name}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix for severity classification."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Severity Classification\n(Segmentation + HSV)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix: {save_path.name}")


def plot_correlation(results, save_path):
    """Plot correlation between true class and estimated severity."""
    true_classes = [r['true_class'] for r in results]
    severities = [r['severity_pct'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    # Jitter x for visibility
    jitter = np.random.uniform(-0.15, 0.15, len(true_classes))
    ax.scatter(np.array(true_classes) + jitter, severities, alpha=0.5, s=30,
               c=[['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'][t] for t in true_classes])

    # Trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_classes, severities)
    x_line = np.linspace(0, 4, 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2,
            label=f'R$^2$={r_value**2:.3f}, p={p_value:.2e}')

    # Spearman correlation
    rho, p_spear = stats.spearmanr(true_classes, severities)

    ax.set_xticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, fontsize=12, fontweight='bold')
    ax.set_xlabel('True Severity Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Estimated Severity (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Severity Correlation (Spearman rho={rho:.3f})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved correlation plot: {save_path.name}")
    return r_value**2, rho


# ============================================================
# Threshold Optimization
# ============================================================
def optimize_thresholds(results):
    """Find optimal severity-to-class thresholds using grid search."""
    severities = np.array([r['severity_pct'] for r in results])
    true_labels = np.array([r['true_class'] for r in results])

    best_acc = 0
    best_thresholds = (5, 15, 35, 60)

    # Grid search over threshold boundaries
    for t1 in range(2, 12, 1):
        for t2 in range(t1 + 3, 30, 2):
            for t3 in range(t2 + 5, 55, 3):
                for t4 in range(t3 + 5, 80, 3):
                    preds = np.zeros_like(true_labels)
                    preds[severities >= t1] = 1
                    preds[severities >= t2] = 2
                    preds[severities >= t3] = 3
                    preds[severities >= t4] = 4
                    acc = np.mean(preds == true_labels)
                    if acc > best_acc:
                        best_acc = acc
                        best_thresholds = (t1, t2, t3, t4)

    print(f"\n  Optimized thresholds: {best_thresholds}")
    print(f"  Optimized accuracy:  {best_acc*100:.2f}%")
    return best_thresholds, best_acc


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("=" * 60)
    print("PAPER 3: HSV-BASED SEVERITY ESTIMATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vis_dir = RESULTS_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Step 1: Load segmentation model
    print("\n--- Step 1: Loading U-Net segmentation model ---")
    seg_model = load_segmentation_model()

    # Step 2: Process all original images
    print("\n--- Step 2: Processing all images ---")
    results = []
    vis_count = {c: 0 for c in range(5)}

    for folder, true_class in CLASS_MAP.items():
        cls_dir = ORIGINAL_DIR / folder
        images = sorted(cls_dir.glob('*.jpg'))
        print(f"\n  Processing {CLASS_NAMES[true_class]} ({folder}): {len(images)} images")

        for img_path in tqdm(images, desc=f"  {CLASS_NAMES[true_class]}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Stage 1: Segment leaf
            leaf_mask = segment_leaf(seg_model, img)

            # Stage 2: HSV severity estimation
            severity_pct, color_map, pixel_stats = estimate_severity_hsv(img, leaf_mask)
            pred_class = severity_to_class(severity_pct)

            results.append({
                'image': img_path.name,
                'true_class': true_class,
                'pred_class': pred_class,
                'severity_pct': severity_pct,
                'pixels': pixel_stats,
            })

            # Save sample visualizations (3 per class)
            if vis_count[true_class] < 3:
                save_path = vis_dir / f"severity_{CLASS_NAMES[true_class]}_{vis_count[true_class]}.png"
                create_severity_visualization(img, leaf_mask, color_map,
                                              severity_pct, pred_class, true_class, save_path)
                vis_count[true_class] += 1

    # Step 3: Results with default thresholds
    print("\n\n--- Step 3: Results (Default Thresholds) ---")
    y_true = [r['true_class'] for r in results]
    y_pred = [r['pred_class'] for r in results]

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"\n  Accuracy:      {acc*100:.2f}%")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")

    # Step 4: Optimize thresholds
    print("\n--- Step 4: Optimizing Severity Thresholds ---")
    opt_thresholds, opt_acc = optimize_thresholds(results)

    # Re-classify with optimized thresholds
    t1, t2, t3, t4 = opt_thresholds
    y_pred_opt = []
    for r in results:
        s = r['severity_pct']
        if s < t1:
            c = 0
        elif s < t2:
            c = 1
        elif s < t3:
            c = 2
        elif s < t4:
            c = 3
        else:
            c = 4
        y_pred_opt.append(c)
        r['pred_class_opt'] = c

    kappa_opt = cohen_kappa_score(y_true, y_pred_opt)
    print(f"\n  Optimized Accuracy:      {opt_acc*100:.2f}%")
    print(f"  Optimized Cohen's Kappa: {kappa_opt:.4f}")
    print(f"\n{classification_report(y_true, y_pred_opt, target_names=CLASS_NAMES)}")

    # Step 5: Generate all plots
    print("\n--- Step 5: Generating Publication Figures ---")
    plot_severity_distribution(results, RESULTS_DIR / "severity_distribution.png")
    plot_confusion_matrix(y_true, y_pred_opt, RESULTS_DIR / "severity_confusion_matrix.png")
    r2, rho = plot_correlation(results, RESULTS_DIR / "severity_correlation.png")

    # Step 6: Save metrics
    metrics = {
        'pipeline': 'U-Net Segmentation + HSV Severity Estimation',
        'total_images': len(results),
        'default_thresholds': [5, 15, 35, 60],
        'default_accuracy': float(acc),
        'default_kappa': float(kappa),
        'optimized_thresholds': list(opt_thresholds),
        'optimized_accuracy': float(opt_acc),
        'optimized_kappa': float(kappa_opt),
        'r_squared': float(r2),
        'spearman_rho': float(rho),
        'per_class_severity_mean': {
            CLASS_NAMES[c]: float(np.mean([r['severity_pct'] for r in results if r['true_class'] == c]))
            for c in range(5)
        },
        'timestamp': datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "severity_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  All results saved to {RESULTS_DIR}")
    print("=" * 60)
    print("SEVERITY ESTIMATION PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
