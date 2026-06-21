#!/usr/bin/env python3
"""
Script 06: Hybrid Severity Estimation (Segmentation + Feature Extraction + ML)
===============================================================================
Combines U-Net leaf segmentation with color/texture feature extraction and
ML classifiers for accurate severity classification.

Pipeline:
  1. U-Net segments leaf from background (Script 04 model)
  2. Extract features from segmented leaf region:
     - HSV color histograms & statistics
     - Green/brown/necrotic pixel ratios
     - GLCM texture features on leaf region
     - Leaf area ratio, perimeter complexity
  3. Train ML classifiers (RF, SVM, GB, XGBoost) on extracted features
  4. Compare: direct classification vs hybrid segmentation-based approach

Author: Kartik E. Cholachgudda (R18PEC20)
"""

import os, sys, json, random, warnings
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
ORIGINAL_DIR = Path(r"d:\Projects\AgRECA\PhD\PhD2\04_Dataset\Downy_Mildew\Original")
SEG_MODEL_PATH = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper3_segmentation\best_unet_resnet34.pt")
RESULTS_DIR = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results\paper3_hybrid_severity")
IMAGE_SIZE = 512
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_MAP = {'O_0': 0, 'O_1': 1, 'O_2': 2, 'O_3': 3, 'O_4': 4}
CLASS_NAMES = ['S0', 'S1', 'S2', 'S3', 'S4']

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# U-Net Model (identical to Script 04)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class UNetResNet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet34(weights=None)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool1 = resnet.maxpool
        self.enc2, self.enc3, self.enc4, self.enc5 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.up5 = nn.ConvTranspose2d(512,256,2,stride=2); self.dec5 = ConvBlock(512,256)
        self.up4 = nn.ConvTranspose2d(256,128,2,stride=2); self.dec4 = ConvBlock(256,128)
        self.up3 = nn.ConvTranspose2d(128,64,2,stride=2);  self.dec3 = ConvBlock(128,64)
        self.up2 = nn.ConvTranspose2d(64,64,2,stride=2);   self.dec2 = ConvBlock(128,64)
        self.up1 = nn.ConvTranspose2d(64,32,2,stride=2);   self.dec1 = ConvBlock(32,32)
        self.final = nn.Conv2d(32, num_classes, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(e2); e4 = self.enc4(e3); e5 = self.enc5(e4)
        d5 = self.dec5(torch.cat([self.up5(e5),e4],1))
        d4 = self.dec4(torch.cat([self.up4(d5),e3],1))
        d3 = self.dec3(torch.cat([self.up3(d4),e2],1))
        d2 = self.dec2(torch.cat([self.up2(d3),e1],1))
        return self.final(self.dec1(self.up1(d2)))

def load_seg_model():
    model = UNetResNet34(num_classes=1, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(SEG_MODEL_PATH, weights_only=True, map_location=DEVICE))
    model.eval()
    return model

def segment_leaf(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                   A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                   ToTensorV2()])
    tensor = t(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = (torch.sigmoid(model(tensor)) > 0.5).float().cpu().squeeze().numpy()
    h, w = img_bgr.shape[:2]
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

# ============================================================
# Feature Extraction (on segmented leaf)
# ============================================================
def extract_features(img_bgr, leaf_mask):
    """Extract 60+ features from the segmented leaf region."""
    features = {}
    h_img, w_img = img_bgr.shape[:2]
    total_pixels = h_img * w_img
    leaf_pixels = leaf_mask > 0
    leaf_area = np.sum(leaf_pixels)

    if leaf_area < 100:
        return None  # skip if segmentation failed

    # --- 1. Leaf morphology features (3) ---
    features['leaf_area_ratio'] = leaf_area / total_pixels
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        area = cv2.contourArea(largest)
        features['compactness'] = (perimeter ** 2) / (4 * np.pi * max(area, 1))
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = area / max(hull_area, 1)
    else:
        features['compactness'] = 0
        features['solidity'] = 0

    # --- 2. HSV color statistics on leaf (18) ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    for name, ch in [('h', h_ch), ('s', s_ch), ('v', v_ch)]:
        vals = ch[leaf_pixels].astype(np.float64)
        features[f'{name}_mean'] = np.mean(vals)
        features[f'{name}_std'] = np.std(vals)
        features[f'{name}_median'] = np.median(vals)
        features[f'{name}_skew'] = float(stats.skew(vals)) if len(vals) > 2 else 0
        features[f'{name}_kurtosis'] = float(stats.kurtosis(vals)) if len(vals) > 2 else 0
        features[f'{name}_range'] = float(np.max(vals) - np.min(vals))

    # --- 3. HSV color histograms on leaf (30: 10 bins x 3 channels) ---
    for name, ch, max_val in [('h', h_ch, 180), ('s', s_ch, 256), ('v', v_ch, 256)]:
        vals = ch[leaf_pixels]
        hist, _ = np.histogram(vals, bins=10, range=(0, max_val), density=True)
        for i, hv in enumerate(hist):
            features[f'{name}_hist_{i}'] = float(hv)

    # --- 4. Color ratio features (6) ---
    green_mask = leaf_pixels & (h_ch >= 25) & (h_ch <= 85) & (s_ch >= 30) & (v_ch >= 30)
    brown_mask = leaf_pixels & (((h_ch < 25) & (h_ch >= 0)) | (h_ch > 150)) & (s_ch >= 20) & (v_ch >= 30)
    necrotic_mask = leaf_pixels & (s_ch < 40) & (v_ch >= 25) & (v_ch <= 120)
    yellow_mask = leaf_pixels & (h_ch >= 15) & (h_ch <= 35) & (s_ch >= 40) & (v_ch >= 50)

    features['green_ratio'] = np.sum(green_mask) / leaf_area
    features['brown_ratio'] = np.sum(brown_mask) / leaf_area
    features['necrotic_ratio'] = np.sum(necrotic_mask) / leaf_area
    features['yellow_ratio'] = np.sum(yellow_mask) / leaf_area
    features['diseased_ratio'] = (np.sum(brown_mask) + np.sum(necrotic_mask)) / leaf_area
    features['healthy_diseased_ratio'] = np.sum(green_mask) / max(np.sum(brown_mask) + np.sum(necrotic_mask), 1)

    # --- 5. Lab color space features on leaf (6) ---
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)
    for name, ch in [('lab_l', l_ch), ('lab_a', a_ch), ('lab_b', b_ch)]:
        vals = ch[leaf_pixels].astype(np.float64)
        features[f'{name}_mean'] = np.mean(vals)
        features[f'{name}_std'] = np.std(vals)

    # --- 6. GLCM texture features on leaf (8) ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Crop to leaf bounding box for GLCM efficiency
    ys, xs = np.where(leaf_pixels)
    if len(ys) > 0:
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = gray[y1:y2+1, x1:x2+1]
        crop_mask = leaf_mask[y1:y2+1, x1:x2+1]
        # Apply mask (set background to 0)
        crop_masked = crop * crop_mask
        # Reduce to 32 levels for GLCM speed
        crop_q = (crop_masked // 8).astype(np.uint8)
        try:
            glcm = graycomatrix(crop_q, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                                levels=32, symmetric=True, normed=True)
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                         'correlation', 'ASM']:
                vals = graycoprops(glcm, prop)
                features[f'glcm_{prop}_mean'] = float(np.mean(vals))
                features[f'glcm_{prop}_std'] = float(np.std(vals))
        except Exception:
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                         'correlation', 'ASM']:
                features[f'glcm_{prop}_mean'] = 0
                features[f'glcm_{prop}_std'] = 0

    return features

# ============================================================
# Visualization
# ============================================================
def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, save_path, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color='#3498db', edgecolor='#2c3e50')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Features for Severity Classification', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_comparison(results_dict, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = list(results_dict.keys())
    accs = [results_dict[m]['accuracy'] * 100 for m in methods]
    kappas = [results_dict[m]['kappa'] for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, label='Accuracy (%)', color='#3498db', edgecolor='#2c3e50')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, kappas, w, label="Cohen's Kappa", color='#e74c3c', edgecolor='#2c3e50')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12, color='#3498db')
    ax2.set_ylabel("Cohen's Kappa", fontsize=12, color='#e74c3c')
    ax.set_title('Severity Classification: Method Comparison', fontsize=14, fontweight='bold')

    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, kappas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("PAPER 3: HYBRID SEVERITY ESTIMATION")
    print(f"Device: {DEVICE} | Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load segmentation model
    print("\n--- Step 1: Loading U-Net model ---")
    seg_model = load_seg_model()
    print("  Model loaded.")

    # Step 2: Extract features for all images
    print("\n--- Step 2: Extracting features from segmented leaves ---")
    all_features = []
    all_labels = []
    all_names = []

    for folder, label in CLASS_MAP.items():
        cls_dir = ORIGINAL_DIR / folder
        images = sorted(cls_dir.glob('*.jpg'))
        print(f"  {CLASS_NAMES[label]} ({folder}): {len(images)} images")

        for img_path in tqdm(images, desc=f"  {CLASS_NAMES[label]}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            leaf_mask = segment_leaf(seg_model, img)
            feats = extract_features(img, leaf_mask)
            if feats is not None:
                all_features.append(feats)
                all_labels.append(label)
                all_names.append(img_path.name)

    # Convert to arrays
    feature_names = list(all_features[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in all_features])
    y = np.array(all_labels)
    print(f"\n  Total samples: {len(X)}, Features: {X.shape[1]}")
    print(f"  Feature names: {feature_names[:10]}...")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 3: Train and evaluate classifiers with 5-fold CV
    print("\n--- Step 3: Training ML classifiers (5-fold CV) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=15,
                                                 min_samples_split=5, random_state=SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                         learning_rate=0.1, random_state=SEED),
        'SVM-RBF': SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED),
        'SVM-Linear': SVC(kernel='linear', C=1, random_state=SEED),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results_dict = {}
    best_acc = 0
    best_name = None
    best_preds = None

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        X_use = X_scaled if 'SVM' in name else X
        y_pred = cross_val_predict(clf, X_use, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        kappa = cohen_kappa_score(y, y_pred)

        results_dict[name] = {'accuracy': acc, 'f1': f1, 'kappa': kappa}
        print(f"    Accuracy: {acc*100:.2f}%  F1: {f1*100:.2f}%  Kappa: {kappa:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_preds = y_pred

    # Step 4: Train best model on full data for feature importance
    print(f"\n--- Step 4: Best model = {best_name} ({best_acc*100:.2f}%) ---")

    # Detailed report for best model
    print(f"\n{classification_report(y, best_preds, target_names=CLASS_NAMES)}")

    # Train RF on full data for feature importance plot
    rf_full = RandomForestClassifier(n_estimators=300, max_depth=15,
                                      min_samples_split=5, random_state=SEED, n_jobs=-1)
    rf_full.fit(X, y)

    # Step 5: Generate figures
    print("\n--- Step 5: Generating publication figures ---")
    plot_confusion(y, best_preds, f'Hybrid Severity Classification\n({best_name}, 5-fold CV)',
                   RESULTS_DIR / "hybrid_confusion_matrix.png")
    print(f"  Saved: hybrid_confusion_matrix.png")

    plot_feature_importance(rf_full, feature_names, RESULTS_DIR / "feature_importance.png")
    print(f"  Saved: feature_importance.png")

    # Add HSV-only and DL classification baselines
    results_dict['HSV-Only (Script 05)'] = {'accuracy': 0.2089, 'f1': 0.18, 'kappa': 0.053}
    results_dict['EfficientNet-B0 (Script 03)'] = {'accuracy': 0.9257, 'f1': 0.9245, 'kappa': 0.907}

    plot_comparison(results_dict, RESULTS_DIR / "method_comparison.png")
    print(f"  Saved: method_comparison.png")

    # Correlation analysis
    # Use RF predicted severity (as proxy) vs true label
    rf_proba = rf_full.predict_proba(X)
    # Weighted severity index
    severity_index = np.sum(rf_proba * np.arange(5), axis=1)
    rho, p_val = stats.spearmanr(y, severity_index)

    fig, ax = plt.subplots(figsize=(8, 6))
    jitter = np.random.uniform(-0.15, 0.15, len(y))
    colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
    ax.scatter(y + jitter, severity_index, alpha=0.4, s=25,
               c=[colors[t] for t in y])
    slope, intercept, r_value, _, _ = stats.linregress(y, severity_index)
    x_line = np.linspace(0, 4, 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2,
            label=f'R$^2$={r_value**2:.3f}, Spearman rho={rho:.3f}')
    ax.set_xticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, fontsize=12, fontweight='bold')
    ax.set_xlabel('True Severity Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Severity Index', fontsize=13, fontweight='bold')
    ax.set_title('Hybrid Model: Severity Correlation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: hybrid_correlation.png")

    # Step 6: Save metrics
    metrics = {
        'pipeline': 'U-Net Segmentation + Feature Extraction + ML Classification',
        'total_samples': int(len(X)),
        'num_features': int(X.shape[1]),
        'feature_names': feature_names,
        'cv_folds': 5,
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results_dict.items()},
        'best_model': best_name,
        'best_accuracy': float(best_acc),
        'best_kappa': float(cohen_kappa_score(y, best_preds)),
        'best_f1': float(f1_score(y, best_preds, average='weighted')),
        'spearman_rho': float(rho),
        'r_squared': float(r_value**2),
        'timestamp': datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "hybrid_severity_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"HYBRID SEVERITY ESTIMATION COMPLETE")
    print(f"Best: {best_name} -> {best_acc*100:.2f}% accuracy, {cohen_kappa_score(y, best_preds):.4f} kappa")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
