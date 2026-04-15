"""
=============================================================================
Inference Pipeline
=============================================================================

Supports two model backends:
  - Random Forest + MFCC (original, model_type="sklearn")
  - PyTorch CNN6 + Mel Spectrogram (SCL, model_type="pytorch")

The load_classifier() factory reads preprocessing_config.json to determine
which backend to use. Both return the same dict format from classify().

Usage (standalone - test mode, picks random file from dataset):
  python inference.py

Usage (standalone - specific file):
  python inference.py --audio path/to/file.wav

Usage (imported):
  from inference import load_classifier
  clf = load_classifier("models/scl_model")
  results = clf.classify("upload.wav")

Author: Hayden Banks
Date: February 2026
"""

import os
import json
import argparse
import pickle
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import butter, filtfilt


class AudioClassifier:
    """Wraps model loading, preprocessing, and prediction."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)

        # Load config
        config_path = model_dir / "preprocessing_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path) as f:
            self.cfg = json.load(f)

        # Load model + scaler
        model_path = model_dir / "model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.class_names = self.cfg["CLASS_NAMES"]
        self.target_sr = self.cfg["TARGET_SR"]
        self.target_length = self.cfg["TARGET_LENGTH"]
        print(f"Classifier loaded ({self.cfg.get('model_name', 'RF')})")

    # ---- Full-recording preprocessing (applied ONCE) ----

    def _preprocess_full_recording(self, audio, sr):
        """Resample and bandpass filter the FULL recording before segmentation.

        Filtering on the full signal avoids edge artifacts that occur when
        applying Butterworth filters to short (<1s) clips.
        """
        # Resample
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

        # Bandpass filter
        nyquist = 0.5 * self.target_sr
        low = self.cfg["FILTER_LOWCUT"] / nyquist
        high = self.cfg["FILTER_HIGHCUT"] / nyquist
        b, a = butter(self.cfg["FILTER_ORDER"], [low, high], btype="band")
        audio = filtfilt(b, a, audio)

        return audio

    # ---- Per-segment feature extraction ----

    def _cyclic_pad(self, audio):
        """Pad or truncate a segment to the target length (3s)."""
        if len(audio) >= self.target_length:
            return audio[:self.target_length]
        repeats = int(np.ceil(self.target_length / len(audio)))
        return np.tile(audio, repeats)[:self.target_length]

    def _extract_features(self, segment):
        """Extract 130-dim MFCC feature vector from a preprocessed segment.

        The segment has ALREADY been resampled and filtered. This method
        only pads to 3s and computes features.
        """
        audio = self._cyclic_pad(segment)
        sr = self.target_sr
        n_mfcc = self.cfg.get("N_MFCC", 20)

        features = []

        # MFCCs + deltas
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))

        delta = librosa.feature.delta(mfccs)
        features.extend(np.mean(delta, axis=1))
        features.extend(np.std(delta, axis=1))

        delta2 = librosa.feature.delta(mfccs, order=2)
        features.extend(np.mean(delta2, axis=1))
        features.extend(np.std(delta2, axis=1))

        # Spectral descriptors
        for func in [
            librosa.feature.spectral_centroid,
            librosa.feature.spectral_bandwidth,
            librosa.feature.spectral_rolloff,
        ]:
            vals = func(y=audio, sr=sr)[0]
            features.extend([np.mean(vals), np.std(vals)])

        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([np.mean(zcr), np.std(zcr)])

        rms = librosa.feature.rms(y=audio)[0]
        features.extend([np.mean(rms), np.std(rms)])

        return np.array(features, dtype=np.float32)

    # ---- Segmentation ----

    def _segment_with_annotations(self, audio, annotation_path):
        """Segment using ICBHI .txt annotation file (cycle boundaries)."""
        sr = self.target_sr
        segments = []
        with open(annotation_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    start, end = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                s, e = int(start * sr), int(end * sr)
                if e > s and e <= len(audio):
                    segments.append({"audio": audio[s:e], "start": start, "end": end})
        return segments

    def _segment_sliding_window(self, audio):
        """Segment using a 3s sliding window with 1.5s hop."""
        sr = self.target_sr
        duration = len(audio) / sr
        window_samples = self.target_length  # 3s worth of samples
        hop_samples = window_samples // 2     # 1.5s hop

        segments = []

        if len(audio) <= window_samples:
            segments.append({"audio": audio, "start": 0.0, "end": duration})
        else:
            pos = 0
            while pos + window_samples <= len(audio):
                segments.append({
                    "audio": audio[pos:pos + window_samples],
                    "start": pos / sr,
                    "end": (pos + window_samples) / sr,
                })
                pos += hop_samples
            # Catch remaining audio if longer than 0.5s
            if pos < len(audio) and (len(audio) - pos) > int(0.5 * sr):
                segments.append({
                    "audio": audio[pos:],
                    "start": pos / sr,
                    "end": duration,
                })

        return segments

    # ---- Public API ----

    def classify(self, audio_path, annotation_path=None):
        """Classify a .wav file.

        Pipeline:
          1. Load full recording
          2. Preprocess full recording (resample + bandpass filter)
          3. Segment (annotations or sliding window)
          4. Extract features per segment
          5. Classify and aggregate

        Returns dict with per-cycle results and overall assessment.
        """
        # Step 1: Load raw audio
        audio_raw, sr_original = librosa.load(audio_path, sr=None)

        # Step 2: Preprocess FULL recording (resample + filter)
        audio = self._preprocess_full_recording(audio_raw, sr_original)
        duration = len(audio) / self.target_sr

        # Step 3: Segment
        if annotation_path and Path(annotation_path).exists():
            segments = self._segment_with_annotations(audio, annotation_path)
        else:
            segments = self._segment_sliding_window(audio)

        if not segments:
            return {
                "overall_label": "Unknown",
                "overall_confidence": 0.0,
                "cycles": [],
                "duration_sec": round(duration, 2),
                "total_cycles": 0,
                "abnormal_cycles": 0,
                "normal_cycles": 0,
                "error": "No valid audio segments found",
            }

        # Step 4: Extract features per segment
        feature_rows = []
        valid_segments = []
        for seg in segments:
            try:
                feat = self._extract_features(seg["audio"])
                if not (np.any(np.isnan(feat)) or np.any(np.isinf(feat))):
                    feature_rows.append(feat)
                    valid_segments.append(seg)
            except Exception:
                continue

        if not feature_rows:
            return {
                "overall_label": "Unknown",
                "overall_confidence": 0.0,
                "cycles": [],
                "duration_sec": round(duration, 2),
                "total_cycles": 0,
                "abnormal_cycles": 0,
                "normal_cycles": 0,
                "error": "Feature extraction failed for all segments",
            }

        X = np.array(feature_rows)
        X_scaled = self.scaler.transform(X)

        # Step 5: Classify
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)

        # Build per-cycle results
        cycle_results = []
        for i, seg in enumerate(valid_segments):
            cycle_results.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "label": self.class_names[preds[i]],
                "confidence": round(float(probs[i, preds[i]]), 4),
                "probabilities": {
                    name: round(float(probs[i, j]), 4)
                    for j, name in enumerate(self.class_names)
                },
            })

        # Aggregate: confidence-weighted majority vote
        label_scores = {name: 0.0 for name in self.class_names}
        for cr in cycle_results:
            label_scores[cr["label"]] += cr["confidence"]

        overall_label = max(label_scores, key=label_scores.get)
        total = sum(label_scores.values())
        overall_conf = label_scores[overall_label] / total if total > 0 else 0.0

        abnormal_count = sum(1 for c in cycle_results if c["label"] != "Normal")

        return {
            "overall_label": overall_label,
            "overall_confidence": round(overall_conf, 4),
            "cycles": cycle_results,
            "duration_sec": round(duration, 2),
            "total_cycles": len(cycle_results),
            "abnormal_cycles": abnormal_count,
            "normal_cycles": len(cycle_results) - abnormal_count,
            "error": None,
        }


class CNNClassifier:
    """Wraps PyTorch CNN6/CNN10/CNN14 model loading, preprocessing, and prediction."""

    def __init__(self, model_dir):
        import torch
        import torchaudio.transforms as T

        model_dir = Path(model_dir)

        config_path = model_dir / "preprocessing_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.class_names = self.cfg["CLASS_NAMES"]
        self.target_sr = self.cfg["TARGET_SR"]
        self.target_length = self.cfg["TARGET_LENGTH"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model architecture
        from audio_preprocessing.scl.models import CNN6, CNN10, CNN14, LinearClassifier

        backbone = self.cfg["backbone"]
        num_classes = self.cfg["NUM_CLASSES"]
        embed_only = self.cfg.get("embed_only", True)
        method = self.cfg.get("method", "hybrid")

        if backbone == 'cnn6':
            self.encoder = CNN6(num_classes=num_classes, embed_only=embed_only,
                                from_scratch=True, device=self.device)
        elif backbone == 'cnn10':
            self.encoder = CNN10(num_classes=num_classes, embed_only=embed_only,
                                 from_scratch=True, device=self.device)
        elif backbone == 'cnn14':
            self.encoder = CNN14(num_classes=num_classes, embed_only=embed_only,
                                 from_scratch=True, device=self.device)

        # Load saved weights
        state_path = model_dir / "model_state.pth"
        if not state_path.exists():
            raise FileNotFoundError(f"Model weights not found: {state_path}")

        checkpoint = torch.load(state_path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.encoder.eval()

        # For methods that have a separate classifier (scl, mscl, hybrid)
        if "classifier" in checkpoint:
            self.classifier = LinearClassifier(name=backbone, num_classes=num_classes, device=self.device)
            self.classifier.load_state_dict(checkpoint["classifier"])
            self.classifier.eval()
            self._use_classifier = True
        else:
            self._use_classifier = False

        # Build mel spectrogram transform
        from audio_preprocessing.scl.utils import Normalize, Standardize

        melspec = T.MelSpectrogram(
            n_fft=self.cfg["N_FFT"], n_mels=self.cfg["N_MELS"],
            win_length=self.cfg.get("WIN_LENGTH", self.cfg["N_FFT"]),
            hop_length=self.cfg["HOP_LENGTH"],
            f_min=self.cfg["FILTER_LOWCUT"], f_max=self.cfg["FILTER_HIGHCUT"],
        ).to(self.device)

        self.transform = torch.nn.Sequential(
            melspec, Normalize(), Standardize(device=self.device)
        ).to(self.device)

        self._torch = torch
        print(f"CNN Classifier loaded ({self.cfg.get('model_name', backbone)}) on {self.device}")

    def _preprocess_full_recording(self, audio, sr):
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        return audio

    def _circular_pad(self, audio_np):
        if len(audio_np) >= self.target_length:
            return audio_np[:self.target_length]
        repeats = int(np.ceil(self.target_length / len(audio_np)))
        padded = np.tile(audio_np, repeats)[:self.target_length]
        # Apply fade out at the end
        fade_len = self.target_sr // 16
        if fade_len > 0 and len(padded) > fade_len:
            fade = np.linspace(1.0, 0.0, fade_len)
            padded[-fade_len:] *= fade
        return padded

    def _segment_with_annotations(self, audio, annotation_path):
        sr = self.target_sr
        segments = []
        with open(annotation_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    start, end = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                s, e = int(start * sr), int(end * sr)
                if e > s and e <= len(audio):
                    segments.append({"audio": audio[s:e], "start": start, "end": end})
        return segments

    def _segment_sliding_window(self, audio):
        sr = self.target_sr
        duration = len(audio) / sr
        window_samples = self.target_length  # 8s
        hop_samples = window_samples // 2     # 4s hop

        segments = []
        if len(audio) <= window_samples:
            segments.append({"audio": audio, "start": 0.0, "end": duration})
        else:
            pos = 0
            while pos + window_samples <= len(audio):
                segments.append({
                    "audio": audio[pos:pos + window_samples],
                    "start": pos / sr,
                    "end": (pos + window_samples) / sr,
                })
                pos += hop_samples
            if pos < len(audio) and (len(audio) - pos) > int(0.5 * sr):
                segments.append({
                    "audio": audio[pos:],
                    "start": pos / sr,
                    "end": duration,
                })
        return segments

    def classify(self, audio_path, annotation_path=None):
        torch = self._torch

        audio_raw, sr_original = librosa.load(audio_path, sr=None)
        audio = self._preprocess_full_recording(audio_raw, sr_original)
        duration = len(audio) / self.target_sr

        if annotation_path and Path(annotation_path).exists():
            segments = self._segment_with_annotations(audio, annotation_path)
        else:
            segments = self._segment_sliding_window(audio)

        if not segments:
            return {
                "overall_label": "Unknown",
                "overall_confidence": 0.0,
                "cycles": [],
                "duration_sec": round(duration, 2),
                "total_cycles": 0,
                "abnormal_cycles": 0,
                "normal_cycles": 0,
                "error": "No valid audio segments found",
            }

        cycle_results = []

        with torch.no_grad():
            for seg in segments:
                padded = self._circular_pad(seg["audio"])
                # Shape: (1, 1, target_length) - batch, channel, time
                tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

                mel = self.transform(tensor)

                if self._use_classifier:
                    features = self.encoder(mel)
                    logits = self.classifier(features)
                else:
                    logits = self.encoder(mel)

                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))

                cycle_results.append({
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "label": self.class_names[pred_idx],
                    "confidence": round(float(probs[pred_idx]), 4),
                    "probabilities": {
                        name: round(float(probs[j]), 4)
                        for j, name in enumerate(self.class_names)
                    },
                })

        # Aggregate: confidence-weighted majority vote
        label_scores = {name: 0.0 for name in self.class_names}
        for cr in cycle_results:
            label_scores[cr["label"]] += cr["confidence"]

        overall_label = max(label_scores, key=label_scores.get)
        total = sum(label_scores.values())
        overall_conf = label_scores[overall_label] / total if total > 0 else 0.0

        abnormal_count = sum(1 for c in cycle_results if c["label"] != "Normal")

        return {
            "overall_label": overall_label,
            "overall_confidence": round(overall_conf, 4),
            "cycles": cycle_results,
            "duration_sec": round(duration, 2),
            "total_cycles": len(cycle_results),
            "abnormal_cycles": abnormal_count,
            "normal_cycles": len(cycle_results) - abnormal_count,
            "error": None,
        }


def load_classifier(model_dir):
    """Factory: load the appropriate classifier based on config's model_type."""
    model_dir = Path(model_dir)
    config_path = model_dir / "preprocessing_config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("model_type") == "pytorch":
        return CNNClassifier(model_dir)
    else:
        return AudioClassifier(model_dir)


# ============================================================================
# CLI
# ============================================================================

def main():
    # Resolve paths relative to this script's location, not the working directory
    SCRIPT_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Respiratory sound inference")
    parser.add_argument("--model_dir", type=str, default=str(SCRIPT_DIR / ".." / "models" / "best_model"))
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to .wav file. If omitted, picks a random file from --data_dir.")
    parser.add_argument("--data_dir", type=str, default=str(SCRIPT_DIR / ".." / "data" / "ICBHI_final_database"),
                        help="Path to ICBHI dataset (used for test mode when --audio is omitted).")
    parser.add_argument("--annotations", type=str, default=None)
    args = parser.parse_args()

    # If no audio file given, pick a random .wav from the dataset for testing
    if args.audio is None:
        import glob
        import random
        wav_files = glob.glob(os.path.join(args.data_dir, "*.wav"))
        if not wav_files:
            print(f"No .wav files found in {args.data_dir}")
            print("Provide --audio or --data_dir with ICBHI files.")
            return
        args.audio = random.choice(wav_files)
        print(f"[TEST MODE] Randomly selected: {Path(args.audio).name}")

    clf = load_classifier(args.model_dir)
    results = clf.classify(args.audio, args.annotations)

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"File:       {args.audio}")
    print(f"Duration:   {results['duration_sec']}s")
    print(f"Segments:   {results['total_cycles']}")
    print(f"\nOverall:    {results['overall_label']} "
          f"(confidence: {results['overall_confidence']:.1%})")
    print(f"Normal:     {results['normal_cycles']}")
    print(f"Abnormal:   {results['abnormal_cycles']}")

    # Build header dynamically based on class names
    class_names = clf.class_names
    header_parts = [f"{'Start':>6s}", f"{'End':>6s}", f"{'Label':>8s}", f"{'Conf':>6s}"]
    header_parts.extend(f"{name:>8s}" for name in class_names)
    print(f"\nPer-Segment Breakdown:")
    print(f"  {'  '.join(header_parts)}")
    print(f"  {'-' * (14 + 10 * len(class_names) + 16)}")
    for c in results["cycles"]:
        p = c["probabilities"]
        parts = [
            f"  {c['start']:6.2f}",
            f"{c['end']:6.2f}",
            f"{c['label']:>8s}",
            f"{c['confidence']:5.1%}",
        ]
        parts.extend(f"{p.get(name, 0.0):7.1%}" for name in class_names)
        print("  ".join(parts))


if __name__ == "__main__":
    main()