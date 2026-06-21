#!/usr/bin/env python3
"""
00_clean_split_and_augment.py
==============================
CLEAN Data Splitting & Augmentation Pipeline (v2)
Fixes augmentation leakage identified in AIIA review.

CRITICAL CHANGE vs v1 (01_augment_and_balance.py):
  v1: Augment ALL 920 → 1750, THEN split → leakage
  v2: Split 920 originals FIRST, THEN augment ONLY training set → clean

Pipeline:
  1. Load 920 originals from O_0..O_4
  2. Stratified split at original-image level (80/10/10, seed=42)
  3. Test and val contain ONLY originals (naturally imbalanced)
  4. Augment ONLY training partition to balance classes
  5. Save split_manifest.csv + augmentation_provenance.csv
  6. Verify zero leakage

Author: Kartik E. Cholachgudda
Date: May 2026 (AIIA review revision)
"""

import os
import sys
import random
import shutil

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import cv2
import albumentations as A
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
DATASET_ROOT = Path(r"d:\Projects\AgRECA\PhD\PhD2\04_Dataset\Downy_Mildew\Original")
OUTPUT_ROOT = Path(r"d:\Projects\AgRECA\PhD\PhD2\04_Dataset\Balanced_Dataset_v2")
RESULTS_DIR = Path(r"d:\Projects\AgRECA\PhD\PhD2\03_Experiments\results")

CLASS_DIRS = {
    "S0": "O_0",  # Healthy      (99 images)
    "S1": "O_1",  # Slight       (409 images)
    "S2": "O_2",  # Moderate     (172 images)
    "S3": "O_3",  # Severe       (132 images)
    "S4": "O_4",  # Very Severe  (108 images)
}

RANDOM_SEED = 42
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# Augmentation Pipeline (identical transforms to v1)
# ============================================================
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    ], p=0.6),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.GaussNoise(p=1.0),
    ], p=0.2),
    A.Affine(translate_percent=(-0.06, 0.06), scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
])


def get_image_files(directory: Path) -> list:
    """Get all image files from a directory, sorted for reproducibility."""
    files = []
    for f in directory.iterdir():
        if f.is_file() and f.suffix in IMAGE_EXTENSIONS:
            files.append(f)
    return sorted(files)


def stratified_split(class_files: dict, train_r=0.8, val_r=0.1, test_r=0.1, seed=42):
    """
    Split original files per class into train/val/test at the IMAGE level.
    Maintains proportional class distribution in val and test (naturally imbalanced).
    
    Returns: dict with 'train', 'val', 'test' keys, each mapping class -> list of paths
    """
    rng = random.Random(seed)
    
    splits = {'train': {}, 'val': {}, 'test': {}}
    
    for cls, files in class_files.items():
        files = list(files)  # copy
        rng.shuffle(files)
        
        n = len(files)
        n_val = max(1, round(n * val_r))    # at least 1 per class
        n_test = max(1, round(n * test_r))  # at least 1 per class
        n_train = n - n_val - n_test
        
        assert n_train > 0, f"Class {cls} has too few images ({n}) for the requested split"
        
        splits['train'][cls] = files[:n_train]
        splits['val'][cls] = files[n_train:n_train + n_val]
        splits['test'][cls] = files[n_train + n_val:]
    
    return splits


def augment_image(image_path: Path) -> np.ndarray:
    """Load and augment a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = augmentation_pipeline(image=img_rgb)
    return cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)


def copy_originals(files: list, dest_dir: Path, class_label: str) -> list:
    """Copy original images to destination, return list of (dest_name, src_name) tuples."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, f in enumerate(files):
        dst_name = f"{class_label}_orig_{i:04d}{f.suffix}"
        dst_path = dest_dir / dst_name
        shutil.copy2(str(f), str(dst_path))
        manifest.append((dst_name, f.name, str(f)))
    return manifest


def augment_training_class(files: list, target_count: int, dest_dir: Path, 
                           class_label: str, start_idx: int) -> list:
    """
    Augment a training class to reach target_count total images.
    Returns list of (dest_name, source_original_name, source_original_path) provenance tuples.
    """
    augment_needed = target_count - len(files)
    if augment_needed <= 0:
        return []  # No augmentation needed (majority class)
    
    provenance = []
    rng = random.Random(RANDOM_SEED + hash(class_label))  # class-specific seed
    aug_count = 0
    max_attempts = augment_needed * 5
    attempts = 0
    
    while aug_count < augment_needed and attempts < max_attempts:
        src_file = rng.choice(files)
        try:
            aug_img = augment_image(src_file)
            dst_name = f"{class_label}_aug_{aug_count:04d}.jpg"
            dst_path = dest_dir / dst_name
            cv2.imwrite(str(dst_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            provenance.append((dst_name, src_file.name, str(src_file)))
            aug_count += 1
        except Exception as e:
            print(f"  Warning: augmentation failed for {src_file.name}: {e}")
        attempts += 1
    
    print(f"  {class_label}: augmented {aug_count} images (from {len(files)} originals → {len(files) + aug_count} total)")
    return provenance


def verify_no_leakage(splits: dict, manifest_data: dict):
    """Verify that no original image appears in more than one split."""
    print("\n" + "=" * 60)
    print("LEAKAGE VERIFICATION")
    print("=" * 60)
    
    # Collect all original source filenames per split
    for cls in splits['train']:
        train_sources = set()
        val_sources = set()
        test_sources = set()
        
        for entry in manifest_data.get(('train', cls), []):
            # entry = (dest_name, src_name, src_path)
            # For originals, src_name is the original filename
            # For augmented, we need to check provenance
            train_sources.add(entry[2])  # source path
        
        for entry in manifest_data.get(('val', cls), []):
            val_sources.add(entry[2])
        
        for entry in manifest_data.get(('test', cls), []):
            test_sources.add(entry[2])
        
        # Check overlaps
        tv_overlap = train_sources & val_sources
        tt_overlap = train_sources & test_sources
        vt_overlap = val_sources & test_sources
        
        if tv_overlap or tt_overlap or vt_overlap:
            print(f"  ❌ LEAKAGE DETECTED in class {cls}!")
            if tv_overlap:
                print(f"    Train-Val overlap: {len(tv_overlap)} images")
            if tt_overlap:
                print(f"    Train-Test overlap: {len(tt_overlap)} images")
            if vt_overlap:
                print(f"    Val-Test overlap: {len(vt_overlap)} images")
            return False
        else:
            print(f"  ✅ Class {cls}: No leakage (train={len(train_sources)}, val={len(val_sources)}, test={len(test_sources)})")
    
    print("  ✅ ALL CLASSES PASS — Zero leakage confirmed")
    return True


def plot_split_distribution(split_stats: dict, output_path: Path):
    """Generate distribution plot showing train/val/test per class."""
    classes = list(CLASS_DIRS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = {'S0': '#FF6B6B', 'S1': '#4ECDC4', 'S2': '#45B7D1', 'S3': '#96CEB4', 'S4': '#FFEAA7'}
    
    for ax, split_name in zip(axes, ['train', 'val', 'test']):
        counts = [split_stats[split_name].get(c, 0) for c in classes]
        bars = ax.bar(classes, counts, color=[colors[c] for c in classes], 
                     edgecolor='#2C3E50', linewidth=1.2)
        ax.set_title(f'{split_name.capitalize()} Set', fontsize=14, fontweight='bold')
        ax.set_xlabel('Severity Class', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        total = sum(counts)
        aug_note = " (balanced, originals+augmented)" if split_name == 'train' else " (originals only)"
        ax.set_title(f'{split_name.capitalize()} (n={total}){aug_note}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Distribution plot saved: {output_path}")


def main():
    print("=" * 60)
    print("CLEAN SPLIT & AUGMENT PIPELINE (v2 — AIIA review fix)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Source: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Split: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
    print(f"Seed: {RANDOM_SEED}")
    print("=" * 60)
    
    # ── Step 1: Load all originals ──────────────────────────────
    print("\n── Step 1: Loading original images ──")
    class_files = {}
    for cls, dir_name in CLASS_DIRS.items():
        src_dir = DATASET_ROOT / dir_name
        files = get_image_files(src_dir)
        class_files[cls] = files
        print(f"  {cls} ({dir_name}): {len(files)} images")
    
    total_originals = sum(len(v) for v in class_files.values())
    print(f"  Total originals: {total_originals}")
    
    # ── Step 2: Stratified split of originals FIRST ─────────────
    print("\n── Step 2: Stratified split of 920 originals ──")
    splits = stratified_split(class_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
    
    for split_name in ['train', 'val', 'test']:
        counts = {cls: len(files) for cls, files in splits[split_name].items()}
        total = sum(counts.values())
        detail = ", ".join([f"{c}:{n}" for c, n in sorted(counts.items())])
        print(f"  {split_name:>5}: {total:>4} images  ({detail})")
    
    # ── Step 3: Clean output directory ──────────────────────────
    if OUTPUT_ROOT.exists():
        print(f"\nRemoving existing output: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    
    split_dir = OUTPUT_ROOT / "splits"
    
    # ── Step 4: Copy val and test (originals ONLY) ──────────────
    print("\n── Step 4: Copying val and test sets (originals only) ──")
    all_manifest = {}  # (split, class) -> [(dest_name, src_name, src_path)]
    
    for split_name in ['val', 'test']:
        for cls in CLASS_DIRS:
            dest = split_dir / split_name / cls
            entries = copy_originals(splits[split_name][cls], dest, cls)
            all_manifest[(split_name, cls)] = entries
            print(f"  {split_name}/{cls}: {len(entries)} originals copied")
    
    # ── Step 5: Copy training originals ─────────────────────────
    print("\n── Step 5: Copying training originals ──")
    train_orig_counts = {}
    for cls in CLASS_DIRS:
        dest = split_dir / 'train' / cls
        entries = copy_originals(splits['train'][cls], dest, cls)
        all_manifest[('train', cls)] = list(entries)  # will extend with augmented
        train_orig_counts[cls] = len(entries)
        print(f"  train/{cls}: {len(entries)} originals copied")
    
    # ── Step 6: Augment ONLY training set to balance ────────────
    print("\n── Step 6: Augmenting training set only ──")
    # Target: match the majority class count in training
    majority_count = max(train_orig_counts.values())
    print(f"  Majority class in train: {majority_count} (S1)")
    print(f"  Balancing all training classes to {majority_count}")
    
    augmentation_provenance = []  # Global provenance log
    
    for cls in CLASS_DIRS:
        dest = split_dir / 'train' / cls
        n_orig = train_orig_counts[cls]
        
        if n_orig >= majority_count:
            print(f"  {cls}: already at {n_orig} (majority class, no augmentation)")
            continue
        
        provenance = augment_training_class(
            splits['train'][cls], majority_count, dest, cls, n_orig
        )
        
        # Add augmented provenance to manifest (source is the ORIGINAL, not the augmented file)
        # But for leakage check, we need to track that augmented images source from train originals
        for entry in provenance:
            all_manifest[('train', cls)].append(entry)
            augmentation_provenance.append({
                'augmented_file': entry[0],
                'source_original': entry[1],
                'source_path': entry[2],
                'class': cls,
                'partition': 'train'
            })
    
    # ── Step 7: Verify zero leakage ─────────────────────────────
    passed = verify_no_leakage(splits, all_manifest)
    if not passed:
        print("\n❌ ABORTING — Leakage detected. This should never happen with the v2 pipeline.")
        sys.exit(1)
    
    # Also verify no augmented images in val/test
    print("\n  Checking val/test for augmented files...")
    for split_name in ['val', 'test']:
        for cls in CLASS_DIRS:
            split_path = split_dir / split_name / cls
            if split_path.exists():
                aug_files = [f for f in split_path.iterdir() if '_aug_' in f.name]
                if aug_files:
                    print(f"  ❌ FOUND {len(aug_files)} augmented files in {split_name}/{cls}!")
                    sys.exit(1)
    print("  ✅ Val and test contain zero augmented images")
    
    # ── Step 8: Compute final statistics ────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SPLIT STATISTICS")
    print("=" * 60)
    
    split_stats = {}
    for split_name in ['train', 'val', 'test']:
        split_stats[split_name] = {}
        for cls in CLASS_DIRS:
            path = split_dir / split_name / cls
            if path.exists():
                count = len([f for f in path.iterdir() if f.is_file()])
                split_stats[split_name][cls] = count
    
    print(f"\n{'':>10} {'S0':>6} {'S1':>6} {'S2':>6} {'S3':>6} {'S4':>6} {'Total':>8}")
    print("-" * 54)
    for split_name in ['train', 'val', 'test']:
        counts = split_stats[split_name]
        total = sum(counts.values())
        row = f"{split_name:>10}"
        for cls in CLASS_DIRS:
            row += f" {counts.get(cls, 0):>6}"
        row += f" {total:>8}"
        print(row)
    
    grand_total = sum(sum(v.values()) for v in split_stats.values())
    print(f"\n  Grand total: {grand_total} images")
    print(f"  Val+Test: originals only (naturally proportional)")
    print(f"  Train: balanced via augmentation of training originals only")
    
    # ── Step 9: Save manifests ──────────────────────────────────
    print("\n── Step 9: Saving manifests and metadata ──")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Split manifest CSV
    manifest_path = OUTPUT_ROOT / "split_manifest.csv"
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['partition', 'class', 'filename', 'source_original', 'source_path', 'is_augmented'])
        for (split_name, cls), entries in sorted(all_manifest.items()):
            for entry in entries:
                is_aug = '_aug_' in entry[0]
                writer.writerow([split_name, cls, entry[0], entry[1], entry[2], is_aug])
    print(f"  ✅ Split manifest: {manifest_path}")
    
    # Augmentation provenance CSV
    prov_path = OUTPUT_ROOT / "augmentation_provenance.csv"
    with open(prov_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['augmented_file', 'source_original', 'source_path', 'class', 'partition'])
        writer.writeheader()
        writer.writerows(augmentation_provenance)
    print(f"  ✅ Augmentation provenance: {prov_path}")
    
    # Metadata JSON
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "v2_clean_split",
        "description": "Split originals FIRST, augment training ONLY. Fixes AIIA review §3.1 leakage.",
        "source": str(DATASET_ROOT),
        "output": str(OUTPUT_ROOT),
        "random_seed": RANDOM_SEED,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "original_counts": {cls: len(files) for cls, files in class_files.items()},
        "split_stats": split_stats,
        "augmentation_config": {
            "HorizontalFlip": 0.5,
            "VerticalFlip": 0.3,
            "RandomRotate90": 0.5,
            "ColorJitter/BrightnessContrast/CLAHE": 0.6,
            "GaussianBlur/GaussNoise": 0.2,
            "ShiftScaleRotate": 0.3,
        },
        "leakage_check": "PASSED",
        "val_test_augmented_count": 0,
        "total_augmented_in_train": len(augmentation_provenance),
    }
    
    meta_path = RESULTS_DIR / 'clean_split_metadata_v2.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Metadata: {meta_path}")
    
    # ── Step 10: Distribution plot ──────────────────────────────
    plot_path = RESULTS_DIR / 'clean_split_distribution_v2.png'
    plot_split_distribution(split_stats, plot_path)
    
    print("\n" + "=" * 60)
    print("✅ CLEAN SPLIT PIPELINE COMPLETE")
    print(f"   Output: {OUTPUT_ROOT}")
    print(f"   Splits: {split_dir}")
    print(f"   Leakage: NONE (verified)")
    print(f"   Val/Test: Originals only")
    print("=" * 60)


if __name__ == '__main__':
    main()
