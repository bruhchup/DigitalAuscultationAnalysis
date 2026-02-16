"""
Trains the deployment model: Random Forest on 130 MFCC + spectral features.

Selected over CNN approaches because:
  - Best balanced accuracy (47.6%) across all 3 classes
  - Best sensitivity for abnormality detection (58.6%)
  - Reasonable specificity (74.0%)
  - Detects crackles at 56% recall (vs CNN's 33%)
  - Fast inference, no GPU needed

Pipeline:
  1. Load ICBHI 2017 annotations -> extract individual respiratory cycles
  2. For each cycle: resample -> bandpass filter -> pad to 3s -> extract 130 features
  3. Patient-level train/val/test split (70/15/15)
  4. Train Random Forest with balanced subsample weighting
  5. Evaluate and save model + scaler + config

Usage:
  python train_final_model.py --data_dir data/ICBHI_final_database
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = ["Normal", "Crackle", "Wheeze"]
NUM_CLASSES = 3
RANDOM_SEED = 42

# Audio
TARGET_SR = 8000
TARGET_DURATION = 3.0
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)  # 24,000 samples
FILTER_LOWCUT = 50
FILTER_HIGHCUT = 2000
FILTER_ORDER = 5

# MFCC extraction
N_MFCC = 20

# Random Forest
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_LEAF = 10
RF_MIN_SAMPLES_SPLIT = 20

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


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


# ============================================================================
# ANNOTATION PARSING
# ============================================================================

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
                label = 0  # Normal
            elif wheeze == 1 and crackle == 0:
                label = 2  # Wheeze
            else:
                label = 1  # Crackle (or both)
            cycles.append((start, end, label))
    return cycles


def extract_patient_id(filename):
    return filename.split("_")[0]


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(audio, sr):
    """Extract 130 features: MFCCs + deltas + spectral descriptors.

    Feature breakdown:
      - 20 MFCC means + 20 MFCC stds           = 40
      - 20 delta-MFCC means + 20 stds           = 40
      - 20 delta-delta-MFCC means + 20 stds     = 40
      - spectral centroid mean/std               = 2
      - spectral bandwidth mean/std              = 2
      - spectral rolloff mean/std                = 2
      - zero crossing rate mean/std              = 2
      - RMS energy mean/std                      = 2
      Total                                      = 130
    """
    # Resample
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Bandpass filter
    audio = bandpass_filter(audio, sr)

    # Pad / truncate
    audio = cyclic_pad(audio, TARGET_LENGTH)

    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    # Delta MFCCs
    delta = librosa.feature.delta(mfccs)
    features.extend(np.mean(delta, axis=1))
    features.extend(np.std(delta, axis=1))

    # Delta-delta MFCCs
    delta2 = librosa.feature.delta(mfccs, order=2)
    features.extend(np.mean(delta2, axis=1))
    features.extend(np.std(delta2, axis=1))

    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.extend([np.mean(spec_cent), np.std(spec_cent)])

    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features.extend([np.mean(spec_bw), np.std(spec_bw)])

    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features.extend([np.mean(spec_rolloff), np.std(spec_rolloff)])

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.extend([np.mean(zcr), np.std(zcr)])

    rms = librosa.feature.rms(y=audio)[0]
    features.extend([np.mean(rms), np.std(rms)])

    return np.array(features, dtype=np.float32)


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(data_dir):
    data_dir = Path(data_dir)
    wav_files = sorted(data_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} .wav files in {data_dir}")

    all_features = []
    all_labels = []
    all_patients = []
    skipped = 0

    for i, wav_path in enumerate(wav_files):
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

            try:
                feat = extract_features(segment, sr)
                if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                    continue
                all_features.append(feat)
                all_labels.append(label)
                all_patients.append(patient_id)
            except Exception:
                continue

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(wav_files)} files "
                  f"({len(all_features)} cycles extracted)")

    print(f"\nDataset loaded:")
    print(f"  Total files: {len(wav_files)}, Skipped: {skipped}")
    print(f"  Cycles extracted: {len(all_features)}")

    return np.array(all_features), np.array(all_labels), all_patients


# ============================================================================
# PATIENT-LEVEL SPLIT
# ============================================================================

def patient_level_split(X, y, patient_ids):
    patient_ids = np.array(patient_ids)
    unique = np.unique(patient_ids)
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique)

    n = len(unique)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_p = set(unique[:n_train])
    val_p = set(unique[n_train:n_train + n_val])
    test_p = set(unique[n_train + n_val:])

    train_mask = np.array([p in train_p for p in patient_ids])
    val_mask = np.array([p in val_p for p in patient_ids])
    test_mask = np.array([p in test_p for p in patient_ids])

    print(f"  Patients  - train:{len(train_p)} val:{len(val_p)} test:{len(test_p)}")
    print(f"  Cycles    - train:{train_mask.sum()} val:{val_mask.sum()} test:{test_mask.sum()}")

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
    )


# ============================================================================
# TRAINING
# ============================================================================

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train
    print(f"\nTraining Random Forest (n={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH})...")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    # Predictions
    y_pred_train = rf.predict(X_train_s)
    y_pred_val = rf.predict(X_val_s)
    y_pred_test = rf.predict(X_test_s)
    y_proba_test = rf.predict_proba(X_test_s)

    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)

    # ---- Print results ----
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print(f"\n  Train Accuracy:    {train_acc * 100:.1f}%")
    print(f"  Val Accuracy:      {val_acc * 100:.1f}%")
    print(f"  Test Accuracy:     {test_acc * 100:.1f}%")
    print(f"  Balanced Accuracy: {bal_acc * 100:.1f}%")
    print(f"  Overfit gap:       {(train_acc - test_acc) * 100:.1f}%")

    print(f"\nPer-Class Recall:")
    report = classification_report(y_test, y_pred_test, target_names=CLASS_NAMES,
                                   output_dict=True)
    for name in CLASS_NAMES:
        print(f"  {name:10s} {report[name]['recall'] * 100:.1f}%")

    print(f"\nConfusion Matrix:")
    header = "            " + "".join(f"{n:>10s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:12s}" + "".join(f"{cm[i, j]:10d}" for j in range(NUM_CLASSES))
        print(row)

    print(f"\n{classification_report(y_test, y_pred_test, target_names=CLASS_NAMES)}")

    # ---- Binary analysis ----
    y_test_bin = (y_test > 0).astype(int)
    y_pred_bin = (y_pred_test > 0).astype(int)
    tn = np.sum((y_test_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_test_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_test_bin == 1) & (y_pred_bin == 0))
    tp = np.sum((y_test_bin == 1) & (y_pred_bin == 1))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print("=" * 70)
    print("BINARY ANALYSIS (Normal vs Abnormal)")
    print("=" * 70)
    print(f"\n  Sensitivity (Detecting Abnormal): {sensitivity * 100:.1f}%")
    print(f"  Specificity (Detecting Normal):   {specificity * 100:.1f}%")
    print(f"  Positive Predictive Value:        {ppv * 100:.1f}%")
    print(f"  Negative Predictive Value:        {npv * 100:.1f}%")

    # ---- Feature importance ----
    importances = rf.feature_importances_
    top_10_idx = np.argsort(importances)[-10:][::-1]
    feature_labels = _feature_names()
    print(f"\nTop 10 Features:")
    for idx in top_10_idx:
        name = feature_labels[idx] if idx < len(feature_labels) else f"feature_{idx}"
        print(f"  {name:35s} {importances[idx]:.4f}")

    # ---- Save model artifacts ----
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nModel saved to {output_dir / 'model.pkl'}")
    print(f"Scaler saved to {output_dir / 'scaler.pkl'}")

    # Save config for inference
    config = {
        "model_type": "sklearn",
        "model_name": "RandomForest_MFCC",
        "TARGET_SR": TARGET_SR,
        "TARGET_DURATION": TARGET_DURATION,
        "TARGET_LENGTH": TARGET_LENGTH,
        "FILTER_LOWCUT": FILTER_LOWCUT,
        "FILTER_HIGHCUT": FILTER_HIGHCUT,
        "FILTER_ORDER": FILTER_ORDER,
        "N_MFCC": N_MFCC,
        "CLASS_NAMES": CLASS_NAMES,
        "NUM_CLASSES": NUM_CLASSES,
    }
    with open(output_dir / "preprocessing_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save results
    results = {
        "model_name": "RandomForest_MFCC",
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "balanced_accuracy": float(bal_acc),
        "overfit_gap": float(train_acc - test_acc),
        "per_class_recall": {
            name: float(report[name]["recall"]) for name in CLASS_NAMES
        },
        "binary_sensitivity": float(sensitivity),
        "binary_specificity": float(specificity),
        "binary_ppv": float(ppv),
        "binary_npv": float(npv),
        "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Plots ----
    _save_confusion_matrix(cm, output_dir)
    _save_class_distribution(y_train, y_val, y_test, output_dir)
    _save_feature_importance(importances, feature_labels, output_dir)

    return rf, scaler, results


def _feature_names():
    """Generate descriptive names for all 130 features."""
    names = []
    for stat in ["mean", "std"]:
        for i in range(N_MFCC):
            names.append(f"MFCC_{i+1}_{stat}")
    for stat in ["mean", "std"]:
        for i in range(N_MFCC):
            names.append(f"delta_MFCC_{i+1}_{stat}")
    for stat in ["mean", "std"]:
        for i in range(N_MFCC):
            names.append(f"delta2_MFCC_{i+1}_{stat}")
    for feat in ["spectral_centroid", "spectral_bandwidth",
                  "spectral_rolloff", "zero_crossing_rate", "rms_energy"]:
        names.append(f"{feat}_mean")
        names.append(f"{feat}_std")
    return names


# ============================================================================
# PLOTS
# ============================================================================

def _save_confusion_matrix(cm, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Random Forest (MFCC Features)")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def _save_class_distribution(y_train, y_val, y_test, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(
        axes, [y_train, y_val, y_test], ["Train", "Validation", "Test"]
    ):
        counts = [np.sum(data == i) for i in range(NUM_CLASSES)]
        ax.bar(CLASS_NAMES, counts, color=["#2ecc71", "#e74c3c", "#3498db"])
        ax.set_title(f"{title} (n={len(data)})")
        ax.set_ylabel("Count")
        for i, c in enumerate(counts):
            ax.text(i, c + 5, str(c), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()


def _save_feature_importance(importances, names, output_dir):
    top_20_idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [names[i] if i < len(names) else f"feat_{i}" for i in top_20_idx],
        importances[top_20_idx],
        color="#3498db",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train AusculTek final model (Random Forest + MFCC)"
    )
    parser.add_argument("--data_dir", type=str, default="data/ICBHI_final_database")
    parser.add_argument("--output_dir", type=str, default="models/best_model")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    # Load
    print("=" * 70)
    print("STEP 1: Loading and extracting features")
    print("=" * 70)
    X, y, patients = load_dataset(data_dir)

    print(f"\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y == i)
        print(f"  {name:10s} {count:5d} ({count / len(y) * 100:.1f}%)")

    # Split
    print(f"\n{'=' * 70}")
    print("STEP 2: Patient-level split")
    print("=" * 70)
    X_train, y_train, X_val, y_val, X_test, y_test = patient_level_split(
        X, y, patients
    )

    # Train + evaluate
    print(f"\n{'=' * 70}")
    print("STEP 3: Training and evaluation")
    print("=" * 70)
    rf, scaler, results = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test, output_dir
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nOutputs in {output_dir}/:")
    print(f"  model.pkl                  <- Random Forest model")
    print(f"  scaler.pkl                 <- StandardScaler for features")
    print(f"  preprocessing_config.json  <- params for inference.py")
    print(f"  model_results.json         <- evaluation metrics")
    print(f"  confusion_matrix.png")
    print(f"  class_distribution.png")
    print(f"  feature_importance.png")


if __name__ == "__main__":
    main()