"""
=============================================================================
AusculTek - Multi-Approach Model Comparison with Full Analysis
=============================================================================

Runs 5 approaches on the same patient-level split, generates per-approach
confusion matrices, ROC curves, per-class recall charts, binary analysis,
feature importance (RF/GB), CNN training curves, and a summary dashboard.

  Approach 1: Random Forest on MFCC features
  Approach 2: SVM on MFCC features
  Approach 3: Small CNN on reduced mel spectrograms (32 bands, 3s)
  Approach 4: Small CNN + SpecAugment
  Approach 5: Gradient Boosting on MFCC features

Usage:
  python train_multi_approach.py --data_dir data/ICBHI_final_database
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = ["Normal", "Crackle", "Wheeze"]
NUM_CLASSES = 3
RANDOM_SEED = 42

# Audio
TARGET_SR = 8000
FILTER_LOWCUT = 50
FILTER_HIGHCUT = 2000
FILTER_ORDER = 5

# Colors for plots
CLASS_COLORS = {"Normal": "#2ecc71", "Crackle": "#e74c3c", "Wheeze": "#3498db"}
APPROACH_COLORS = [
    "#2ecc71",  # RF
    "#3498db",  # SVM
    "#e74c3c",  # SmallCNN
    "#9b59b6",  # SpecAugment
    "#f39c12",  # GradientBoosting
]


# ============================================================================
# AUDIO UTILITIES
# ============================================================================

def bandpass_filter(audio, sr):
    nyquist = 0.5 * sr
    low = FILTER_LOWCUT / nyquist
    high = FILTER_HIGHCUT / nyquist
    b, a = butter(FILTER_ORDER, [low, high], btype="band")
    return filtfilt(b, a, audio)


def cyclic_pad(audio, target_length):
    if len(audio) >= target_length:
        return audio[:target_length]
    repeats = int(np.ceil(target_length / len(audio)))
    return np.tile(audio, repeats)[:target_length]


def parse_annotations(txt_path):
    cycles = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
                crackle, wheeze = int(parts[2]), int(parts[3])
            except ValueError:
                continue
            if crackle == 0 and wheeze == 0:
                label = 0
            elif wheeze == 1 and crackle == 0:
                label = 2
            else:
                label = 1
            cycles.append((start, end, label))
    return cycles


def extract_patient_id(filename):
    return filename.split("_")[0]


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_raw_cycles(data_dir):
    data_dir = Path(data_dir)
    wav_files = sorted(data_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} .wav files")

    all_cycles = []
    skipped = 0

    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            skipped += 1
            continue

        cycles = parse_annotations(txt_path)
        if not cycles:
            continue

        try:
            audio_full, sr = librosa.load(wav_path, sr=None)
        except Exception:
            skipped += 1
            continue

        patient_id = extract_patient_id(wav_path.stem)

        for start, end, label in cycles:
            s = int(start * sr)
            e = int(end * sr)
            if e <= s or e > len(audio_full):
                continue
            segment = audio_full[s:e]
            if len(segment) < int(0.15 * sr):
                continue

            all_cycles.append({
                "audio": segment,
                "sr": sr,
                "label": label,
                "patient_id": patient_id,
                "source": wav_path.stem,
            })

    print(f"Loaded {len(all_cycles)} cycles (skipped {skipped} files)")
    return all_cycles


# ============================================================================
# PATIENT-LEVEL SPLIT
# ============================================================================

def patient_split(cycles, train_ratio=0.70, val_ratio=0.15):
    rng = np.random.RandomState(RANDOM_SEED)
    patients = list(set(c["patient_id"] for c in cycles))
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    train = [c for c in cycles if c["patient_id"] in train_p]
    val = [c for c in cycles if c["patient_id"] in val_p]
    test = [c for c in cycles if c["patient_id"] in test_p]

    print(f"  Patients - train:{len(train_p)} val:{len(val_p)} test:{len(test_p)}")
    print(f"  Cycles   - train:{len(train)} val:{len(val)} test:{len(test)}")
    return train, val, test


# ============================================================================
# FEATURE EXTRACTION: MFCCs
# ============================================================================

def extract_mfcc_features(audio, sr, n_mfcc=20, target_duration=3.0):
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    audio = bandpass_filter(audio, sr)
    target_len = int(target_duration * sr)
    audio = cyclic_pad(audio, target_len)

    features = []
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    delta = librosa.feature.delta(mfccs)
    features.extend(np.mean(delta, axis=1))
    features.extend(np.std(delta, axis=1))

    delta2 = librosa.feature.delta(mfccs, order=2)
    features.extend(np.mean(delta2, axis=1))
    features.extend(np.std(delta2, axis=1))

    for func in [librosa.feature.spectral_centroid,
                 librosa.feature.spectral_bandwidth,
                 librosa.feature.spectral_rolloff]:
        vals = func(y=audio, sr=sr)[0]
        features.extend([np.mean(vals), np.std(vals)])

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.extend([np.mean(zcr), np.std(zcr)])
    rms = librosa.feature.rms(y=audio)[0]
    features.extend([np.mean(rms), np.std(rms)])

    return np.array(features, dtype=np.float32)


def build_mfcc_dataset(cycles):
    X, y = [], []
    for c in cycles:
        try:
            feat = extract_mfcc_features(c["audio"], c["sr"])
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                continue
            X.append(feat)
            y.append(c["label"])
        except Exception:
            continue
    return np.array(X), np.array(y)


# ============================================================================
# FEATURE EXTRACTION: Small mel spectrograms
# ============================================================================

def extract_small_mel(audio, sr, n_mels=32, target_duration=3.0):
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    audio = bandpass_filter(audio, sr)
    target_len = int(target_duration * sr)
    audio = cyclic_pad(audio, target_len)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=512, hop_length=512, fmax=2000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)


def build_mel_dataset(cycles, n_mels=32, target_duration=3.0):
    X, y = [], []
    for c in cycles:
        try:
            mel = extract_small_mel(c["audio"], c["sr"], n_mels, target_duration)
            X.append(mel)
            y.append(c["label"])
        except Exception:
            continue
    return np.array(X), np.array(y)


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def binary_metrics(y_true, y_pred):
    """Compute Normal vs Abnormal binary metrics."""
    yt = (y_true > 0).astype(int)
    yp = (y_pred > 0).astype(int)
    tn = np.sum((yt == 0) & (yp == 0))
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    tp = np.sum((yt == 1) & (yp == 1))
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def feature_names_130():
    names = []
    for stat in ["mean", "std"]:
        for i in range(20):
            names.append(f"MFCC_{i+1}_{stat}")
    for stat in ["mean", "std"]:
        for i in range(20):
            names.append(f"dMFCC_{i+1}_{stat}")
    for stat in ["mean", "std"]:
        for i in range(20):
            names.append(f"d2MFCC_{i+1}_{stat}")
    for feat in ["spec_centroid", "spec_bandwidth",
                  "spec_rolloff", "zcr", "rms"]:
        names.append(f"{feat}_mean")
        names.append(f"{feat}_std")
    return names


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, title, filepath):
    """Single confusion matrix heatmap with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)

    # Overlay count + percentage
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            text_color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color=text_color, fontweight="bold")

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_all_confusion_matrices(results_list, y_tests, output_dir):
    """Grid of all confusion matrices side by side."""
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r, yt in zip(axes, results_list, y_tests):
        cm = confusion_matrix(yt, r["y_pred"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm, annot=False, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j + 0.5, i + 0.5,
                        f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)",
                        ha="center", va="center", fontsize=9,
                        color=color, fontweight="bold")
        ax.set_title(r["name"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices - All Approaches", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "all_confusion_matrices.png", dpi=150)
    plt.close()


def plot_per_class_recall(results_list, y_tests, output_dir):
    """Grouped bar chart comparing per-class recall across approaches."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(NUM_CLASSES)
    width = 0.15
    n = len(results_list)

    for idx, (r, yt) in enumerate(zip(results_list, y_tests)):
        report = classification_report(yt, r["y_pred"], target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        recalls = [report[name]["recall"] * 100 for name in CLASS_NAMES]
        offset = (idx - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=r["name"],
                      color=APPROACH_COLORS[idx], alpha=0.85)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.set_title("Per-Class Recall Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_recall.png", dpi=150)
    plt.close()


def plot_per_class_precision(results_list, y_tests, output_dir):
    """Grouped bar chart comparing per-class precision across approaches."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(NUM_CLASSES)
    width = 0.15
    n = len(results_list)

    for idx, (r, yt) in enumerate(zip(results_list, y_tests)):
        report = classification_report(yt, r["y_pred"], target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        precisions = [report[name]["precision"] * 100 for name in CLASS_NAMES]
        offset = (idx - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, precisions, width, label=r["name"],
                      color=APPROACH_COLORS[idx], alpha=0.85)
        for bar, val in zip(bars, precisions):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("Per-Class Precision Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_precision.png", dpi=150)
    plt.close()


def plot_binary_analysis(results_list, y_tests, output_dir):
    """Compare sensitivity vs specificity across approaches (Normal vs Abnormal)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["name"] for r in results_list]
    sensitivities = []
    specificities = []

    for r, yt in zip(results_list, y_tests):
        bm = binary_metrics(yt, r["y_pred"])
        sensitivities.append(bm["sensitivity"] * 100)
        specificities.append(bm["specificity"] * 100)

    # Sensitivity
    bars = axes[0].barh(names, sensitivities, color=APPROACH_COLORS[:len(names)], alpha=0.85)
    axes[0].set_xlabel("Sensitivity (%)")
    axes[0].set_title("Sensitivity\n(Abnormal Detection Rate)", fontweight="bold")
    axes[0].set_xlim(0, 110)
    for bar, val in zip(bars, sensitivities):
        axes[0].text(val + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)

    # Specificity
    bars = axes[1].barh(names, specificities, color=APPROACH_COLORS[:len(names)], alpha=0.85)
    axes[1].set_xlabel("Specificity (%)")
    axes[1].set_title("Specificity\n(Normal Confirmation Rate)", fontweight="bold")
    axes[1].set_xlim(0, 110)
    for bar, val in zip(bars, specificities):
        axes[1].text(val + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)

    plt.suptitle("Binary Analysis: Normal vs Abnormal", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "binary_analysis.png", dpi=150)
    plt.close()


def plot_roc_curves(results_list, y_tests, output_dir):
    """Per-class ROC curves for each approach (only for models with predict_proba)."""
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(6 * NUM_CLASSES, 5))

    for class_idx in range(NUM_CLASSES):
        ax = axes[class_idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

        for idx, (r, yt) in enumerate(zip(results_list, y_tests)):
            if "y_proba" not in r or r["y_proba"] is None:
                continue
            y_bin = (yt == class_idx).astype(int)
            scores = r["y_proba"][:, class_idx]
            fpr, tpr, _ = roc_curve(y_bin, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=APPROACH_COLORS[idx], linewidth=2,
                    label=f"{r['name']} (AUC={roc_auc:.2f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC - {CLASS_NAMES[class_idx]}", fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    plt.suptitle("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close()


def plot_overfitting_comparison(results_list, output_dir):
    """Train vs Test accuracy showing overfitting gap."""
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r["name"] for r in results_list]
    x = np.arange(len(names))
    width = 0.35

    train_accs = [r["train_acc"] * 100 for r in results_list]
    test_accs = [r["test_acc"] * 100 for r in results_list]

    bars1 = ax.bar(x - width / 2, train_accs, width, label="Train",
                   color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, test_accs, width, label="Test",
                   color="#e74c3c", alpha=0.8)

    # Add gap annotations
    for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
        gap = tr - te
        color = "#e74c3c" if gap > 20 else ("#f39c12" if gap > 10 else "#2ecc71")
        ax.annotate(f"Gap: {gap:.1f}%", xy=(i, max(tr, te) + 2),
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overfitting Analysis: Train vs Test Accuracy", fontsize=14,
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "overfitting_comparison.png", dpi=150)
    plt.close()


def plot_summary_dashboard(results_list, y_tests, output_dir):
    """4-panel summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    names = [r["name"] for r in results_list]
    n = len(names)

    # Panel 1: Test Accuracy vs Balanced Accuracy
    ax = axes[0, 0]
    x = np.arange(n)
    w = 0.35
    test_accs = [r["test_acc"] * 100 for r in results_list]
    bal_accs = [r["bal_acc"] * 100 for r in results_list]
    ax.bar(x - w / 2, test_accs, w, label="Test Acc", color="#3498db", alpha=0.85)
    ax.bar(x + w / 2, bal_accs, w, label="Balanced Acc", color="#2ecc71", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test Accuracy vs Balanced Accuracy", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Sensitivity vs Specificity scatter
    ax = axes[0, 1]
    for idx, (r, yt) in enumerate(zip(results_list, y_tests)):
        bm = binary_metrics(yt, r["y_pred"])
        ax.scatter(bm["specificity"] * 100, bm["sensitivity"] * 100,
                   s=200, c=APPROACH_COLORS[idx], label=r["name"],
                   edgecolors="black", zorder=3)
    ax.set_xlabel("Specificity (%) - Normal Confirmation")
    ax.set_ylabel("Sensitivity (%) - Abnormal Detection")
    ax.set_title("Sensitivity vs Specificity Tradeoff", fontweight="bold")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # Panel 3: Per-class F1 scores
    ax = axes[1, 0]
    x = np.arange(NUM_CLASSES)
    w = 0.15
    for idx, (r, yt) in enumerate(zip(results_list, y_tests)):
        report = classification_report(yt, r["y_pred"], target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        f1s = [report[name]["f1-score"] * 100 for name in CLASS_NAMES]
        offset = (idx - n / 2 + 0.5) * w
        ax.bar(x + offset, f1s, w, label=r["name"],
               color=APPROACH_COLORS[idx], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Per-Class F1 Score Comparison", fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Overfitting gap
    ax = axes[1, 1]
    gaps = [(r["train_acc"] - r["test_acc"]) * 100 for r in results_list]
    colors = ["#e74c3c" if g > 20 else ("#f39c12" if g > 10 else "#2ecc71")
              for g in gaps]
    bars = ax.barh(names, gaps, color=colors, alpha=0.85)
    ax.set_xlabel("Train-Test Gap (%)")
    ax.set_title("Overfitting Gap (lower = better)", fontweight="bold")
    for bar, val in zip(bars, gaps):
        ax.text(max(val + 1, 1), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.axvline(x=10, color="green", linestyle="--", alpha=0.5, label="Good (<10%)")
    ax.axvline(x=20, color="orange", linestyle="--", alpha=0.5, label="Moderate (<20%)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle("AusculTek Model Comparison Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "summary_dashboard.png", dpi=150)
    plt.close()


def plot_feature_importance(model, model_name, output_dir):
    """Plot top 20 feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    names = feature_names_130()
    top_idx = np.argsort(importances)[-20:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [names[i] if i < len(names) else f"feat_{i}" for i in top_idx],
        importances[top_idx],
        color="#3498db", alpha=0.85,
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 20 Features - {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_{model_name}.png", dpi=150)
    plt.close()


def plot_cnn_training_curves(history, model_name, output_dir):
    """Loss and accuracy curves for CNN models."""
    if history is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val", linewidth=2)
    axes[0].set_title(f"{model_name} - Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val", linewidth=2)
    axes[1].set_title(f"{model_name} - Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"training_curves_{model_name}.png", dpi=150)
    plt.close()


def plot_class_distribution(y_train, y_val, y_test, output_dir):
    """Show class distribution across splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(axes, [y_train, y_val, y_test],
                                ["Train", "Validation", "Test"]):
        counts = [np.sum(data == i) for i in range(NUM_CLASSES)]
        bars = ax.bar(CLASS_NAMES, counts,
                      color=[CLASS_COLORS[n] for n in CLASS_NAMES])
        ax.set_title(f"{title} (n={len(data)})", fontweight="bold")
        ax.set_ylabel("Count")
        for bar, c in zip(bars, counts):
            pct = c / len(data) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, c + 5,
                    f"{c}\n({pct:.0f}%)", ha="center", fontsize=9)
    plt.suptitle("Class Distribution by Split", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 70)
    print("APPROACH 1: Random Forest on MFCC Features")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=10,
        min_samples_split=20, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)
    y_proba = rf.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, rf.predict(X_train_s))

    _print_results("Random Forest", train_acc, acc, bal_acc, y_test, y_pred)

    return {
        "name": "RandomForest", "model": rf, "scaler": scaler,
        "test_acc": acc, "bal_acc": bal_acc, "train_acc": train_acc,
        "y_pred": y_pred, "y_proba": y_proba, "approach": "mfcc",
        "history": None,
    }


def train_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 70)
    print("APPROACH 2: SVM on MFCC Features")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
              random_state=RANDOM_SEED, probability=True)
    svm.fit(X_train_s, y_train)

    y_pred = svm.predict(X_test_s)
    y_proba = svm.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, svm.predict(X_train_s))

    _print_results("SVM", train_acc, acc, bal_acc, y_test, y_pred)

    return {
        "name": "SVM", "model": svm, "scaler": scaler,
        "test_acc": acc, "bal_acc": bal_acc, "train_acc": train_acc,
        "y_pred": y_pred, "y_proba": y_proba, "approach": "mfcc",
        "history": None,
    }


def build_small_cnn(input_shape, num_classes=3):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])


def train_small_cnn(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 70)
    print("APPROACH 3: Small CNN (32 mel bands, 3s)")
    print("=" * 70)

    Xt = X_train[..., np.newaxis]
    Xv = X_val[..., np.newaxis]
    Xte = X_test[..., np.newaxis]

    model = build_small_cnn(Xt.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    history = model.fit(Xt, y_train, validation_data=(Xv, y_val),
                        epochs=50, batch_size=32, callbacks=cb, verbose=0)

    y_proba = model.predict(Xte, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    train_acc = history.history["accuracy"][-1]

    _print_results("Small CNN", train_acc, acc, bal_acc, y_test, y_pred)

    return {
        "name": "SmallCNN", "model": model,
        "test_acc": acc, "bal_acc": bal_acc, "train_acc": train_acc,
        "y_pred": y_pred, "y_proba": y_proba, "approach": "mel_small",
        "history": history,
    }


class SpecAugment(layers.Layer):
    """Randomly mask frequency bands and time steps during training."""

    def __init__(self, freq_mask_param=4, time_mask_param=8,
                 n_freq_masks=1, n_time_masks=1, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def call(self, inputs, training=None):
        if not training:
            return inputs
        x = inputs
        shape = tf.shape(x)
        freq_dim = shape[1]
        time_dim = shape[2]

        for _ in range(self.n_freq_masks):
            f = tf.random.uniform([], 1, self.freq_mask_param + 1, dtype=tf.int32)
            f = tf.minimum(f, freq_dim)
            f0 = tf.random.uniform([], 0, freq_dim - f, dtype=tf.int32)
            mask = tf.concat([tf.ones([f0]), tf.zeros([f]),
                              tf.ones([freq_dim - f0 - f])], axis=0)
            x = x * tf.reshape(mask, [1, -1, 1, 1])

        for _ in range(self.n_time_masks):
            t = tf.random.uniform([], 1, self.time_mask_param + 1, dtype=tf.int32)
            t = tf.minimum(t, time_dim)
            t0 = tf.random.uniform([], 0, time_dim - t, dtype=tf.int32)
            mask = tf.concat([tf.ones([t0]), tf.zeros([t]),
                              tf.ones([time_dim - t0 - t])], axis=0)
            x = x * tf.reshape(mask, [1, 1, -1, 1])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "n_freq_masks": self.n_freq_masks,
            "n_time_masks": self.n_time_masks,
        })
        return config


def train_specaugment_cnn(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 70)
    print("APPROACH 4: Small CNN + SpecAugment")
    print("=" * 70)

    Xt = X_train[..., np.newaxis]
    Xv = X_val[..., np.newaxis]
    Xte = X_test[..., np.newaxis]

    model = models.Sequential([
        layers.Input(shape=Xt.shape[1:]),
        SpecAugment(freq_mask_param=4, time_mask_param=6),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    history = model.fit(Xt, y_train, validation_data=(Xv, y_val),
                        epochs=60, batch_size=32, callbacks=cb, verbose=0)

    y_proba = model.predict(Xte, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    train_acc = history.history["accuracy"][-1]

    _print_results("SpecAugment CNN", train_acc, acc, bal_acc, y_test, y_pred)

    return {
        "name": "SpecAugmentCNN", "model": model,
        "test_acc": acc, "bal_acc": bal_acc, "train_acc": train_acc,
        "y_pred": y_pred, "y_proba": y_proba, "approach": "mel_small",
        "history": history,
    }


def train_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 70)
    print("APPROACH 5: Gradient Boosting on MFCC Features")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=15, min_samples_split=25, subsample=0.8,
        random_state=RANDOM_SEED,
    )
    sample_weights = compute_sample_weight("balanced", y_train)
    gb.fit(X_train_s, y_train, sample_weight=sample_weights)

    y_pred = gb.predict(X_test_s)
    y_proba = gb.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, gb.predict(X_train_s))

    _print_results("Gradient Boosting", train_acc, acc, bal_acc, y_test, y_pred)

    return {
        "name": "GradientBoosting", "model": gb, "scaler": scaler,
        "test_acc": acc, "bal_acc": bal_acc, "train_acc": train_acc,
        "y_pred": y_pred, "y_proba": y_proba, "approach": "mfcc",
        "history": None,
    }


def _print_results(name, train_acc, test_acc, bal_acc, y_test, y_pred):
    """Standardized console output for each approach."""
    bm = binary_metrics(y_test, y_pred)
    print(f"\n  Train Accuracy:    {train_acc * 100:.1f}%")
    print(f"  Test Accuracy:     {test_acc * 100:.1f}%")
    print(f"  Balanced Accuracy: {bal_acc * 100:.1f}%")
    print(f"  Overfit gap:       {(train_acc - test_acc) * 100:.1f}%")
    print(f"  Sensitivity:       {bm['sensitivity'] * 100:.1f}%")
    print(f"  Specificity:       {bm['specificity'] * 100:.1f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)}")


# ============================================================================
# COMPARISON & OUTPUT
# ============================================================================

def print_comparison_table(results_list, y_tests):
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Approach':<22s} {'Test':>6s} {'Bal':>6s} {'Train':>6s} "
          f"{'Gap':>6s} {'Sens':>6s} {'Spec':>6s} {'F1-N':>5s} {'F1-C':>5s} {'F1-W':>5s}")
    print(f"  {'-' * 88}")

    for r, yt in zip(results_list, y_tests):
        bm = binary_metrics(yt, r["y_pred"])
        gap = (r["train_acc"] - r["test_acc"]) * 100
        report = classification_report(yt, r["y_pred"], target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        f1s = [report[n]["f1-score"] * 100 for n in CLASS_NAMES]

        print(f"  {r['name']:<22s} {r['test_acc']*100:5.1f}% {r['bal_acc']*100:5.1f}% "
              f"{r['train_acc']*100:5.1f}% {gap:5.1f}% "
              f"{bm['sensitivity']*100:5.1f}% {bm['specificity']*100:5.1f}% "
              f"{f1s[0]:4.1f}% {f1s[1]:4.1f}% {f1s[2]:4.1f}%")

    best_bal = max(results_list, key=lambda r: r["bal_acc"])
    best_acc = max(results_list, key=lambda r: r["test_acc"])
    print(f"\n  Best balanced accuracy: {best_bal['name']} ({best_bal['bal_acc']*100:.1f}%)")
    print(f"  Best test accuracy:     {best_acc['name']} ({best_acc['test_acc']*100:.1f}%)")


def save_results_json(results_list, y_tests, output_dir):
    output = []
    for r, yt in zip(results_list, y_tests):
        bm = binary_metrics(yt, r["y_pred"])
        report = classification_report(yt, r["y_pred"], target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        output.append({
            "name": r["name"],
            "test_accuracy": float(r["test_acc"]),
            "balanced_accuracy": float(r["bal_acc"]),
            "train_accuracy": float(r["train_acc"]),
            "overfit_gap": float(r["train_acc"] - r["test_acc"]),
            "sensitivity": float(bm["sensitivity"]),
            "specificity": float(bm["specificity"]),
            "ppv": float(bm["ppv"]),
            "npv": float(bm["npv"]),
            "per_class": {
                name: {
                    "precision": float(report[name]["precision"]),
                    "recall": float(report[name]["recall"]),
                    "f1": float(report[name]["f1-score"]),
                    "support": int(report[name]["support"]),
                } for name in CLASS_NAMES
            },
            "confusion_matrix": confusion_matrix(yt, r["y_pred"]).tolist(),
        })

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults JSON saved to {output_dir / 'comparison_results.json'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AusculTek multi-approach comparison")
    parser.add_argument("--data_dir", type=str, default="data/ICBHI_final_database")
    parser.add_argument("--output_dir", type=str, default="results/comparison")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    # ---- Load ----
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    all_cycles = load_raw_cycles(data_dir)

    labels = np.array([c["label"] for c in all_cycles])
    print(f"\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(labels == i)
        print(f"  {name:10s} {count:5d} ({count / len(labels) * 100:.1f}%)")

    # ---- Split ----
    print(f"\n{'=' * 70}")
    print("PATIENT-LEVEL SPLIT")
    print("=" * 70)
    train_cycles, val_cycles, test_cycles = patient_split(all_cycles)

    # ---- Extract features ----
    print(f"\n{'=' * 70}")
    print("EXTRACTING MFCC FEATURES")
    print("=" * 70)
    X_train_mfcc, y_train_mfcc = build_mfcc_dataset(train_cycles)
    X_val_mfcc, y_val_mfcc = build_mfcc_dataset(val_cycles)
    X_test_mfcc, y_test_mfcc = build_mfcc_dataset(test_cycles)
    print(f"  MFCC features: {X_train_mfcc.shape[1]} per sample")

    print(f"\n{'=' * 70}")
    print("EXTRACTING SMALL MEL SPECTROGRAMS (32 bands, 3s)")
    print("=" * 70)
    X_train_mel, y_train_mel = build_mel_dataset(train_cycles)
    X_val_mel, y_val_mel = build_mel_dataset(val_cycles)
    X_test_mel, y_test_mel = build_mel_dataset(test_cycles)
    print(f"  Mel shape: {X_train_mel.shape[1:]}")

    # Class distribution plot
    plot_class_distribution(y_train_mfcc, y_val_mfcc, y_test_mfcc, output_dir)

    # ---- Train all approaches ----
    results = []
    y_tests = []  # track which y_test goes with which result

    r = train_random_forest(X_train_mfcc, y_train_mfcc, X_val_mfcc, y_val_mfcc,
                            X_test_mfcc, y_test_mfcc)
    results.append(r)
    y_tests.append(y_test_mfcc)

    r = train_svm(X_train_mfcc, y_train_mfcc, X_val_mfcc, y_val_mfcc,
                  X_test_mfcc, y_test_mfcc)
    results.append(r)
    y_tests.append(y_test_mfcc)

    r = train_small_cnn(X_train_mel, y_train_mel, X_val_mel, y_val_mel,
                        X_test_mel, y_test_mel)
    results.append(r)
    y_tests.append(y_test_mel)

    r = train_specaugment_cnn(X_train_mel, y_train_mel, X_val_mel, y_val_mel,
                              X_test_mel, y_test_mel)
    results.append(r)
    y_tests.append(y_test_mel)

    r = train_gradient_boosting(X_train_mfcc, y_train_mfcc, X_val_mfcc, y_val_mfcc,
                                X_test_mfcc, y_test_mfcc)
    results.append(r)
    y_tests.append(y_test_mfcc)

    # ---- Comparison table ----
    print_comparison_table(results, y_tests)

    # ---- Generate all plots ----
    print(f"\n{'=' * 70}")
    print("GENERATING ANALYSIS PLOTS")
    print("=" * 70)

    # Individual confusion matrices
    for r, yt in zip(results, y_tests):
        plot_confusion_matrix(
            yt, r["y_pred"],
            f"Confusion Matrix - {r['name']}",
            output_dir / f"cm_{r['name']}.png",
        )
        print(f"  Saved cm_{r['name']}.png")

    # All confusion matrices in one figure
    plot_all_confusion_matrices(results, y_tests, output_dir)
    print("  Saved all_confusion_matrices.png")

    # Per-class recall and precision
    plot_per_class_recall(results, y_tests, output_dir)
    print("  Saved per_class_recall.png")

    plot_per_class_precision(results, y_tests, output_dir)
    print("  Saved per_class_precision.png")

    # Binary analysis
    plot_binary_analysis(results, y_tests, output_dir)
    print("  Saved binary_analysis.png")

    # ROC curves
    plot_roc_curves(results, y_tests, output_dir)
    print("  Saved roc_curves.png")

    # Overfitting comparison
    plot_overfitting_comparison(results, output_dir)
    print("  Saved overfitting_comparison.png")

    # Summary dashboard
    plot_summary_dashboard(results, y_tests, output_dir)
    print("  Saved summary_dashboard.png")

    # Feature importance for tree models
    for r in results:
        if hasattr(r.get("model"), "feature_importances_"):
            plot_feature_importance(r["model"], r["name"], output_dir)
            print(f"  Saved feature_importance_{r['name']}.png")

    # CNN training curves
    for r in results:
        if r.get("history") is not None:
            plot_cnn_training_curves(r["history"], r["name"], output_dir)
            print(f"  Saved training_curves_{r['name']}.png")

    # Save JSON
    save_results_json(results, y_tests, output_dir)

    # ---- File listing ----
    print(f"\n{'=' * 70}")
    print("ALL OUTPUTS")
    print("=" * 70)
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size
        unit = "KB" if size > 1024 else "B"
        size_val = size / 1024 if size > 1024 else size
        print(f"  {f.name:<45s} {size_val:6.1f} {unit}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()