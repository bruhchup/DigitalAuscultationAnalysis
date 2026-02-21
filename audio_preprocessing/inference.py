"""
=============================================================================
Inference Pipeline (Random Forest + MFCC)
=============================================================================

Loads the trained RF model + scaler + config and classifies a raw .wav file.
Designed to be imported by the FastAPI web app or run standalone for testing.

Pipeline:
  1. Load full recording
  2. Resample to 8kHz + bandpass filter (50-2000Hz) on FULL recording
  3. Segment into 3s windows (or use ICBHI annotations if provided)
  4. Extract MFCC features per segment
  5. Classify each segment, aggregate results

Usage (standalone - test mode, picks random file from dataset):
  python inference.py

Usage (standalone - specific file):
  python inference.py --audio path/to/file.wav

Usage (imported):
  from inference import AudioClassifier
  clf = AudioClassifier("models/best_model")
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

    clf = AudioClassifier(args.model_dir)
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

    print(f"\nPer-Segment Breakdown:")
    print(f"  {'Start':>6s}  {'End':>6s}  {'Label':>8s}  {'Conf':>6s}  "
          f"{'Normal':>7s}  {'Crackle':>8s}  {'Wheeze':>7s}")
    print(f"  {'-' * 58}")
    for c in results["cycles"]:
        p = c["probabilities"]
        print(
            f"  {c['start']:6.2f}  {c['end']:6.2f}  {c['label']:>8s}  "
            f"{c['confidence']:5.1%}  {p['Normal']:6.1%}  "
            f"{p['Crackle']:7.1%}  {p['Wheeze']:6.1%}"
        )


if __name__ == "__main__":
    main()