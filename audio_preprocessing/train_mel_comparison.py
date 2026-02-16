"""
Every model uses mel spectrogram-derived features exclusively.

  Classical ML (RF, SVM, GB):  138-dim summary vector
    - 32 mel bands x 4 stats (mean, std, max, skew) = 128
    - Spectral centroid, bandwidth, rolloff, ZCR, RMS: mean & std = 10
    
  CNNs (SmallCNN, SpecAugmentCNN):  Full mel spectrogram as 2D input
    - (32 x time_frames x 1), dB-scaled and z-score normalized

All models evaluated on the SAME test samples for fair comparison.

Usage:
  python train_mel_comparison.py --data_dir data/ICBHI_final_database

Output: results/mel_comparison/
"""

import os, sys, json, argparse, pickle
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import skew as scipy_skew
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    confusion_matrix, classification_report,
    balanced_accuracy_score, accuracy_score, roc_curve, auc,
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
# CONFIG
# ============================================================================

CLASS_NAMES = ["Normal", "Crackle", "Wheeze"]
NUM_CLASSES = 3
RANDOM_SEED = 42
TARGET_SR = 8000
TARGET_DURATION = 3.0
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)
FILTER_LOW = 50
FILTER_HIGH = 2000
FILTER_ORDER = 5
N_MELS = 32

FEATURE_TAG = "Mel Spectrogram"
COLORS_APPROACH = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]
COLORS_CLASS = {"Normal": "#2ecc71", "Crackle": "#e74c3c", "Wheeze": "#3498db"}


# ============================================================================
# AUDIO
# ============================================================================

def bandpass(audio, sr):
    ny = 0.5 * sr
    b, a = butter(FILTER_ORDER, [FILTER_LOW / ny, FILTER_HIGH / ny], btype="band")
    return filtfilt(b, a, audio)

def pad(audio, length):
    if len(audio) >= length:
        return audio[:length]
    return np.tile(audio, int(np.ceil(length / len(audio))))[:length]

def preprocess(audio, sr):
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return pad(bandpass(audio, TARGET_SR), TARGET_LENGTH)

def parse_annotations(txt):
    out = []
    with open(txt) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 4: continue
            try:
                s, e, cr, wh = float(p[0]), float(p[1]), int(p[2]), int(p[3])
            except ValueError: continue
            lab = 0 if (cr == 0 and wh == 0) else (2 if (wh == 1 and cr == 0) else 1)
            out.append((s, e, lab))
    return out


# ============================================================================
# DATASET
# ============================================================================

def load_cycles(data_dir):
    data_dir = Path(data_dir)
    wavs = sorted(data_dir.glob("*.wav"))
    print(f"Found {len(wavs)} .wav files")
    cycles = []
    for wp in wavs:
        tp = wp.with_suffix(".txt")
        if not tp.exists(): continue
        anns = parse_annotations(tp)
        if not anns: continue
        try:
            audio, sr = librosa.load(wp, sr=None)
        except Exception: continue
        pid = wp.stem.split("_")[0]
        for start, end, lab in anns:
            s, e = int(start * sr), int(end * sr)
            if e <= s or e > len(audio): continue
            seg = audio[s:e]
            if len(seg) < int(0.15 * sr): continue
            cycles.append({"audio": seg, "sr": sr, "label": lab, "pid": pid})
    print(f"Loaded {len(cycles)} cycles")
    return cycles

def split_patients(cycles):
    rng = np.random.RandomState(RANDOM_SEED)
    pids = list(set(c["pid"] for c in cycles))
    rng.shuffle(pids)
    n = len(pids)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    tr_p = set(pids[:n_tr])
    va_p = set(pids[n_tr:n_tr + n_va])
    te_p = set(pids[n_tr + n_va:])
    tr = [c for c in cycles if c["pid"] in tr_p]
    va = [c for c in cycles if c["pid"] in va_p]
    te = [c for c in cycles if c["pid"] in te_p]
    print(f"  Patients - train:{len(tr_p)} val:{len(va_p)} test:{len(te_p)}")
    print(f"  Cycles   - train:{len(tr)} val:{len(va)} test:{len(te)}")
    return tr, va, te


# ============================================================================
# MEL SPECTROGRAM EXTRACTION (both representations per cycle)
# ============================================================================

def compute_mel_spectrogram(audio_pp):
    """Compute dB-scaled mel spectrogram (32 x time_frames)."""
    mel = librosa.feature.melspectrogram(
        y=audio_pp, sr=TARGET_SR, n_mels=N_MELS,
        n_fft=512, hop_length=512, fmax=FILTER_HIGH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def extract_mel_vector(audio_pp):
    """138-dim summary vector for classical ML.
    
    Feature breakdown:
      - 32 mel bands x mean     = 32
      - 32 mel bands x std      = 32
      - 32 mel bands x max      = 32
      - 32 mel bands x skew     = 32
      - spectral centroid  mean/std = 2
      - spectral bandwidth mean/std = 2
      - spectral rolloff   mean/std = 2
      - ZCR mean/std                = 2
      - RMS mean/std                = 2
      Total                        = 138
    """
    mel_db = compute_mel_spectrogram(audio_pp)

    features = []
    # Per-band statistics across time
    features.extend(np.mean(mel_db, axis=1))          # 32
    features.extend(np.std(mel_db, axis=1))            # 32
    features.extend(np.max(mel_db, axis=1))            # 32
    features.extend(scipy_skew(mel_db, axis=1))        # 32

    # Spectral descriptors (computed from raw audio, not mel)
    for func in [librosa.feature.spectral_centroid,
                 librosa.feature.spectral_bandwidth,
                 librosa.feature.spectral_rolloff]:
        v = func(y=audio_pp, sr=TARGET_SR)[0]
        features.extend([np.mean(v), np.std(v)])
    zcr = librosa.feature.zero_crossing_rate(audio_pp)[0]
    features.extend([np.mean(zcr), np.std(zcr)])
    rms = librosa.feature.rms(y=audio_pp)[0]
    features.extend([np.mean(rms), np.std(rms)])

    return np.array(features, dtype=np.float32)


def extract_mel_matrix(audio_pp):
    """Normalized mel spectrogram (32 x time_frames) for CNN input."""
    mel_db = compute_mel_spectrogram(audio_pp)
    return ((mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)).astype(np.float32)


def build_datasets(cycles):
    """Extract BOTH vector and matrix for each cycle simultaneously.
    
    Ensures all models evaluate on the exact same samples.
    """
    vectors, matrices, labels = [], [], []
    for c in cycles:
        try:
            pp = preprocess(c["audio"], c["sr"])
            vec = extract_mel_vector(pp)
            mat = extract_mel_matrix(pp)
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                continue
            if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
                continue
            vectors.append(vec)
            matrices.append(mat)
            labels.append(c["label"])
        except Exception:
            continue
    return np.array(vectors), np.array(matrices), np.array(labels)


# ============================================================================
# MODELS
# ============================================================================

def train_rf(Xtr, ytr, Xte, yte):
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)
    m = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=10,
                                min_samples_split=20, max_features="sqrt",
                                class_weight="balanced_subsample",
                                random_state=RANDOM_SEED, n_jobs=-1)
    m.fit(Xtr_s, ytr)
    yp = m.predict(Xte_s)
    ypr = m.predict_proba(Xte_s)
    return {"name": "RandomForest", "y_pred": yp, "y_proba": ypr,
            "train_acc": accuracy_score(ytr, m.predict(Xtr_s)),
            "test_acc": accuracy_score(yte, yp),
            "bal_acc": balanced_accuracy_score(yte, yp),
            "model": m, "history": None}

def train_svm(Xtr, ytr, Xte, yte):
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)
    m = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
            random_state=RANDOM_SEED, probability=True)
    m.fit(Xtr_s, ytr)
    yp = m.predict(Xte_s)
    ypr = m.predict_proba(Xte_s)
    return {"name": "SVM", "y_pred": yp, "y_proba": ypr,
            "train_acc": accuracy_score(ytr, m.predict(Xtr_s)),
            "test_acc": accuracy_score(yte, yp),
            "bal_acc": balanced_accuracy_score(yte, yp),
            "model": m, "history": None}

def train_gb(Xtr, ytr, Xte, yte):
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)
    m = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                    max_depth=5, min_samples_leaf=15,
                                    min_samples_split=25, subsample=0.8,
                                    random_state=RANDOM_SEED)
    sw = compute_sample_weight("balanced", ytr)
    m.fit(Xtr_s, ytr, sample_weight=sw)
    yp = m.predict(Xte_s)
    ypr = m.predict_proba(Xte_s)
    return {"name": "GradientBoosting", "y_pred": yp, "y_proba": ypr,
            "train_acc": accuracy_score(ytr, m.predict(Xtr_s)),
            "test_acc": accuracy_score(yte, yp),
            "bal_acc": balanced_accuracy_score(yte, yp),
            "model": m, "history": None}


class SpecAugment(layers.Layer):
    def __init__(self, freq_mask=4, time_mask=6, **kw):
        super().__init__(**kw)
        self.fm, self.tm = freq_mask, time_mask
    def call(self, x, training=None):
        if not training: return x
        sh = tf.shape(x)
        f = tf.minimum(tf.random.uniform([], 1, self.fm+1, tf.int32), sh[1])
        f0 = tf.random.uniform([], 0, sh[1]-f, tf.int32)
        x = x * tf.reshape(tf.concat([tf.ones([f0]), tf.zeros([f]),
                                       tf.ones([sh[1]-f0-f])], 0), [1,-1,1,1])
        t = tf.minimum(tf.random.uniform([], 1, self.tm+1, tf.int32), sh[2])
        t0 = tf.random.uniform([], 0, sh[2]-t, tf.int32)
        return x * tf.reshape(tf.concat([tf.ones([t0]), tf.zeros([t]),
                                          tf.ones([sh[2]-t0-t])], 0), [1,1,-1,1])
    def get_config(self):
        return {**super().get_config(), "freq_mask": self.fm, "time_mask": self.tm}


def _build_cnn(input_shape, use_specaug=False):
    lyrs = [layers.Input(shape=input_shape)]
    if use_specaug:
        lyrs.append(SpecAugment(freq_mask=4, time_mask=6))
    lyrs += [
        layers.Conv2D(16, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(), layers.MaxPooling2D((2,2)), layers.Dropout(0.3),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(), layers.MaxPooling2D((2,2)), layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"), layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
    return models.Sequential(lyrs)


def train_cnn(Xtr, ytr, Xva, yva, Xte, yte, name, use_specaug=False):
    Xt = Xtr[..., np.newaxis]; Xv = Xva[..., np.newaxis]; Xe = Xte[..., np.newaxis]
    model = _build_cnn(Xt.shape[1:], use_specaug)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    patience = 12 if use_specaug else 10
    epochs = 60 if use_specaug else 50
    cb = [callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                  restore_best_weights=True),
          callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)]
    hist = model.fit(Xt, ytr, validation_data=(Xv, yva),
                     epochs=epochs, batch_size=32, callbacks=cb, verbose=0)
    ypr = model.predict(Xe, verbose=0)
    yp = np.argmax(ypr, axis=1)
    return {"name": name, "y_pred": yp, "y_proba": ypr,
            "train_acc": hist.history["accuracy"][-1],
            "test_acc": accuracy_score(yte, yp),
            "bal_acc": balanced_accuracy_score(yte, yp),
            "model": model, "history": hist}


# ============================================================================
# ANALYSIS + PLOTTING
# ============================================================================

def binary_metrics(yt, yp):
    yt_b = (yt > 0).astype(int); yp_b = (yp > 0).astype(int)
    tn = np.sum((yt_b==0)&(yp_b==0)); fp = np.sum((yt_b==0)&(yp_b==1))
    fn = np.sum((yt_b==1)&(yp_b==0)); tp = np.sum((yt_b==1)&(yp_b==1))
    return {"sensitivity": tp/(tp+fn) if tp+fn else 0,
            "specificity": tn/(tn+fp) if tn+fp else 0,
            "ppv": tp/(tp+fp) if tp+fp else 0,
            "npv": tn/(tn+fn) if tn+fn else 0}


def mel_feature_names():
    """Generate descriptive names for all 138 mel vector features."""
    names = []
    for stat in ["mean", "std", "max", "skew"]:
        for i in range(N_MELS):
            names.append(f"mel_{i+1}_{stat}")
    for feat in ["spec_centroid", "spec_bandwidth", "spec_rolloff", "zcr", "rms"]:
        names.append(f"{feat}_mean")
        names.append(f"{feat}_std")
    return names


def _print(name, r, yte):
    bm = binary_metrics(yte, r["y_pred"])
    gap = (r["train_acc"] - r["test_acc"]) * 100
    print(f"\n  Train: {r['train_acc']*100:.1f}%  Test: {r['test_acc']*100:.1f}%  "
          f"Balanced: {r['bal_acc']*100:.1f}%  Gap: {gap:.1f}%")
    print(f"  Sensitivity: {bm['sensitivity']*100:.1f}%  Specificity: {bm['specificity']*100:.1f}%")
    print(classification_report(yte, r["y_pred"], target_names=CLASS_NAMES, zero_division=0))


def plot_cm(yt, yp, title, path):
    cm = confusion_matrix(yt, yp)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(cm, annot=False, cmap="Oranges", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    for i in range(3):
        for j in range(3):
            c = "white" if cm[i,j] > cm.max()*0.5 else "black"
            ax.text(j+.5, i+.5, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=11, color=c, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_all_cms(results, yt, out):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(yt, r["y_pred"])
        pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm, annot=False, cmap="Oranges", xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES, ax=ax)
        for i in range(3):
            for j in range(3):
                c = "white" if cm[i,j] > cm.max()*0.5 else "black"
                ax.text(j+.5,i+.5, f"{cm[i,j]}\n({pct[i,j]:.0f}%)",
                        ha="center",va="center",fontsize=8,color=c,fontweight="bold")
        ax.set_title(r["name"], fontsize=10, fontweight="bold")
    plt.suptitle(f"Confusion Matrices - {FEATURE_TAG}", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(out/"all_confusion_matrices.png", dpi=150); plt.close()

def plot_class_bars(results, yt, metric, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(3); w = 0.15; n = len(results)
    for idx, r in enumerate(results):
        rpt = classification_report(yt, r["y_pred"], target_names=CLASS_NAMES,
                                     output_dict=True, zero_division=0)
        vals = [rpt[c][metric]*100 for c in CLASS_NAMES]
        off = (idx - n/2 + 0.5) * w
        bars = ax.bar(x+off, vals, w, label=r["name"], color=COLORS_APPROACH[idx], alpha=.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{v:.0f}%",
                    ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.set_ylim(0,110); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_binary(results, yt, out):
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    names = [r["name"] for r in results]
    sens = [binary_metrics(yt, r["y_pred"])["sensitivity"]*100 for r in results]
    spec = [binary_metrics(yt, r["y_pred"])["specificity"]*100 for r in results]
    for ax, vals, lab in zip(axes, [sens, spec],
                              ["Sensitivity\n(Abnormal Detection)",
                               "Specificity\n(Normal Confirmation)"]):
        bars = ax.barh(names, vals, color=COLORS_APPROACH[:len(names)], alpha=.85)
        ax.set_xlabel(f"{lab.split(chr(10))[0]} (%)"); ax.set_title(lab, fontweight="bold")
        ax.set_xlim(0,110)
        for b, v in zip(bars, vals):
            ax.text(v+1, b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center", fontsize=9)
    plt.suptitle(f"Binary Analysis ({FEATURE_TAG})", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(out/"binary_analysis.png", dpi=150); plt.close()

def plot_roc(results, yt, out):
    fig, axes = plt.subplots(1,3,figsize=(18,5))
    for ci in range(3):
        ax = axes[ci]; ax.plot([0,1],[0,1],"k--",alpha=.3)
        for idx, r in enumerate(results):
            if r["y_proba"] is None: continue
            yb = (yt==ci).astype(int)
            fpr, tpr, _ = roc_curve(yb, r["y_proba"][:,ci])
            ax.plot(fpr, tpr, color=COLORS_APPROACH[idx], lw=2,
                    label=f"{r['name']} ({auc(fpr,tpr):.2f})")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC - {CLASS_NAMES[ci]}", fontweight="bold")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.suptitle(f"ROC Curves ({FEATURE_TAG})", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(out/"roc_curves.png", dpi=150); plt.close()

def plot_overfit(results, out):
    fig, ax = plt.subplots(figsize=(12,6))
    names = [r["name"] for r in results]; x = np.arange(len(names)); w=.35
    tr = [r["train_acc"]*100 for r in results]; te = [r["test_acc"]*100 for r in results]
    ax.bar(x-w/2, tr, w, label="Train", color="#3498db", alpha=.8)
    ax.bar(x+w/2, te, w, label="Test", color="#e74c3c", alpha=.8)
    for i in range(len(names)):
        g = tr[i]-te[i]; c = "#e74c3c" if g>20 else ("#f39c12" if g>10 else "#2ecc71")
        ax.annotate(f"Gap: {g:.1f}%", xy=(i, max(tr[i],te[i])+2),
                    ha="center", fontsize=9, color=c, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,110)
    ax.set_title(f"Overfitting ({FEATURE_TAG})", fontweight="bold")
    ax.legend(); ax.grid(axis="y",alpha=.3)
    plt.tight_layout(); plt.savefig(out/"overfitting.png", dpi=150); plt.close()

def plot_dashboard(results, yt, out):
    fig, axes = plt.subplots(2,2,figsize=(16,12))
    names = [r["name"] for r in results]; n = len(names)
    # Accuracy
    ax=axes[0,0]; x=np.arange(n); w=.35
    ax.bar(x-w/2, [r["test_acc"]*100 for r in results], w, label="Test", color="#3498db")
    ax.bar(x+w/2, [r["bal_acc"]*100 for r in results], w, label="Balanced", color="#2ecc71")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Test vs Balanced Accuracy", fontweight="bold")
    ax.legend(); ax.set_ylim(0,80); ax.grid(axis="y",alpha=.3)
    # Sens vs Spec
    ax=axes[0,1]
    for i,r in enumerate(results):
        bm=binary_metrics(yt,r["y_pred"])
        ax.scatter(bm["specificity"]*100, bm["sensitivity"]*100, s=200,
                   c=COLORS_APPROACH[i], label=r["name"], edgecolors="black", zorder=3)
    ax.set_xlabel("Specificity (%)"); ax.set_ylabel("Sensitivity (%)")
    ax.set_title("Sens vs Spec Tradeoff", fontweight="bold")
    ax.set_xlim(0,105); ax.set_ylim(0,105)
    ax.axhline(50,color="gray",ls="--",alpha=.3); ax.axvline(50,color="gray",ls="--",alpha=.3)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    # F1
    ax=axes[1,0]; x2=np.arange(3); w2=.15
    for i,r in enumerate(results):
        rpt=classification_report(yt,r["y_pred"],target_names=CLASS_NAMES,output_dict=True,zero_division=0)
        ax.bar(x2+(i-n/2+.5)*w2, [rpt[c]["f1-score"]*100 for c in CLASS_NAMES],
               w2, label=r["name"], color=COLORS_APPROACH[i], alpha=.85)
    ax.set_xticks(x2); ax.set_xticklabels(CLASS_NAMES); ax.set_ylabel("F1 (%)")
    ax.set_title("Per-Class F1", fontweight="bold"); ax.legend(fontsize=7)
    ax.set_ylim(0,100); ax.grid(axis="y",alpha=.3)
    # Gap
    ax=axes[1,1]
    gaps=[(r["train_acc"]-r["test_acc"])*100 for r in results]
    cs=["#e74c3c" if g>20 else ("#f39c12" if g>10 else "#2ecc71") for g in gaps]
    ax.barh(names, gaps, color=cs, alpha=.85)
    ax.set_xlabel("Gap (%)"); ax.set_title("Overfitting Gap", fontweight="bold")
    ax.axvline(10,color="green",ls="--",alpha=.5); ax.axvline(20,color="orange",ls="--",alpha=.5)
    ax.grid(axis="x",alpha=.3)
    plt.suptitle(f"AusculTek Dashboard - {FEATURE_TAG}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,.96])
    plt.savefig(out/"summary_dashboard.png", dpi=150); plt.close()

def plot_cnn_curves(hist, name, out):
    if hist is None: return
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(hist.history["accuracy"],label="Train",lw=2)
    axes[0].plot(hist.history["val_accuracy"],label="Val",lw=2)
    axes[0].set_title(f"{name} Accuracy",fontweight="bold"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(hist.history["loss"],label="Train",lw=2)
    axes[1].plot(hist.history["val_loss"],label="Val",lw=2)
    axes[1].set_title(f"{name} Loss",fontweight="bold"); axes[1].legend(); axes[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out/f"training_curves_{name}.png", dpi=150); plt.close()

def plot_feat_importance(model, name, out, n_features):
    if not hasattr(model, "feature_importances_"): return
    fi = model.feature_importances_
    fn = mel_feature_names() if len(fi) == 138 else [f"f{i}" for i in range(len(fi))]
    top = np.argsort(fi)[-20:]
    fig, ax = plt.subplots(figsize=(10,8))
    ax.barh([fn[i] if i < len(fn) else f"f{i}" for i in top], fi[top], color="#f39c12")
    ax.set_xlabel("Importance"); ax.set_title(f"Top 20 Features - {name}", fontweight="bold")
    plt.tight_layout(); plt.savefig(out/f"feature_importance_{name}.png", dpi=150); plt.close()

def plot_class_dist(ytr, yva, yte, out):
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax,d,t in zip(axes,[ytr,yva,yte],["Train","Val","Test"]):
        cts = [np.sum(d==i) for i in range(3)]
        bars = ax.bar(CLASS_NAMES, cts, color=[COLORS_CLASS[n] for n in CLASS_NAMES])
        ax.set_title(f"{t} (n={len(d)})", fontweight="bold"); ax.set_ylabel("Count")
        for b,c in zip(bars,cts):
            ax.text(b.get_x()+b.get_width()/2, c+5, f"{c}\n({c/len(d)*100:.0f}%)",
                    ha="center", fontsize=9)
    plt.suptitle("Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(out/"class_distribution.png", dpi=150); plt.close()


def save_json(results, yt, out):
    data = []
    for r in results:
        bm = binary_metrics(yt, r["y_pred"])
        rpt = classification_report(yt, r["y_pred"], target_names=CLASS_NAMES,
                                     output_dict=True, zero_division=0)
        data.append({
            "name": r["name"], "features": FEATURE_TAG,
            "test_accuracy": float(r["test_acc"]),
            "balanced_accuracy": float(r["bal_acc"]),
            "train_accuracy": float(r["train_acc"]),
            "sensitivity": float(bm["sensitivity"]),
            "specificity": float(bm["specificity"]),
            "per_class": {c: {"precision": float(rpt[c]["precision"]),
                               "recall": float(rpt[c]["recall"]),
                               "f1": float(rpt[c]["f1-score"])}
                          for c in CLASS_NAMES},
            "confusion_matrix": confusion_matrix(yt, r["y_pred"]).tolist(),
        })
    with open(out/"results.json","w") as f: json.dump(data, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ICBHI_final_database")
    parser.add_argument("--output_dir", default="results/mel_comparison")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED); tf.random.set_seed(RANDOM_SEED)
    data_dir = Path(args.data_dir); out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists(): print(f"ERROR: {data_dir}"); sys.exit(1)

    # Load
    print("="*70 + f"\nLOADING DATASET\n" + "="*70)
    cycles = load_cycles(data_dir)
    labels = np.array([c["label"] for c in cycles])
    for i,n in enumerate(CLASS_NAMES):
        c=np.sum(labels==i); print(f"  {n:10s} {c:5d} ({c/len(labels)*100:.1f}%)")

    # Split
    print(f"\n{'='*70}\nPATIENT-LEVEL SPLIT\n{'='*70}")
    tr_cyc, va_cyc, te_cyc = split_patients(cycles)

    # Extract BOTH representations simultaneously
    print(f"\n{'='*70}\nEXTRACTING MEL SPECTROGRAM FEATURES\n{'='*70}")
    print("  Building vectors (138-dim) + matrices (32 x time) per cycle...")
    Xtr_v, Xtr_m, ytr = build_datasets(tr_cyc)
    Xva_v, Xva_m, yva = build_datasets(va_cyc)
    Xte_v, Xte_m, yte = build_datasets(te_cyc)
    print(f"  Vector shape: {Xtr_v.shape[1]}  |  Matrix shape: {Xtr_m.shape[1:]}")
    print(f"  Train: {len(ytr)}  Val: {len(yva)}  Test: {len(yte)}")

    plot_class_dist(ytr, yva, yte, out)

    # Train
    results = []
    for name, func in [
        ("RandomForest", lambda: train_rf(Xtr_v, ytr, Xte_v, yte)),
        ("SVM", lambda: train_svm(Xtr_v, ytr, Xte_v, yte)),
        ("SmallCNN", lambda: train_cnn(Xtr_m, ytr, Xva_m, yva, Xte_m, yte, "SmallCNN", False)),
        ("SpecAugmentCNN", lambda: train_cnn(Xtr_m, ytr, Xva_m, yva, Xte_m, yte, "SpecAugmentCNN", True)),
        ("GradientBoosting", lambda: train_gb(Xtr_v, ytr, Xte_v, yte)),
    ]:
        print(f"\n{'='*70}\nAPPROACH: {name} (Mel Spectrogram)\n{'='*70}")
        r = func()
        _print(name, r, yte)
        results.append(r)

    # Summary
    print(f"\n{'='*70}\nCOMPARISON ({FEATURE_TAG})\n{'='*70}")
    print(f"\n  {'Model':<22s} {'Test':>6s} {'Bal':>6s} {'Train':>6s} "
          f"{'Gap':>6s} {'Sens':>6s} {'Spec':>6s}")
    print(f"  {'-'*70}")
    for r in results:
        bm = binary_metrics(yte, r["y_pred"])
        g = (r["train_acc"]-r["test_acc"])*100
        print(f"  {r['name']:<22s} {r['test_acc']*100:5.1f}% {r['bal_acc']*100:5.1f}% "
              f"{r['train_acc']*100:5.1f}% {g:5.1f}% "
              f"{bm['sensitivity']*100:5.1f}% {bm['specificity']*100:5.1f}%")
    best = max(results, key=lambda r: r["bal_acc"])
    print(f"\n  >>> Best: {best['name']} (balanced: {best['bal_acc']*100:.1f}%)")

    # Plots
    print(f"\n{'='*70}\nGENERATING PLOTS\n{'='*70}")
    for r in results:
        plot_cm(yte, r["y_pred"], f"{r['name']} ({FEATURE_TAG})", out/f"cm_{r['name']}.png")
    plot_all_cms(results, yte, out)
    plot_class_bars(results, yte, "recall", "Recall (%)",
                    f"Per-Class Recall ({FEATURE_TAG})", out/"per_class_recall.png")
    plot_class_bars(results, yte, "precision", "Precision (%)",
                    f"Per-Class Precision ({FEATURE_TAG})", out/"per_class_precision.png")
    plot_class_bars(results, yte, "f1-score", "F1 (%)",
                    f"Per-Class F1 ({FEATURE_TAG})", out/"per_class_f1.png")
    plot_binary(results, yte, out)
    plot_roc(results, yte, out)
    plot_overfit(results, out)
    plot_dashboard(results, yte, out)
    for r in results:
        plot_cnn_curves(r.get("history"), r["name"], out)
        plot_feat_importance(r.get("model"), r["name"], out, Xtr_v.shape[1])
    save_json(results, yte, out)

    print(f"\nAll outputs in {out}/:")
    for f in sorted(out.glob("*")): print(f"  {f.name}")
    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
