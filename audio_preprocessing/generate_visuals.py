"""
Generate MFCC and Mel Spectrogram visualizations for presentation slides.
Picks a sample from each class (Normal, Crackle, Wheeze, Both) and produces
publication-quality plots.

Usage:
    python generate_visuals.py
    python generate_visuals.py --data_dir path/to/ICBHI_final_database --output_dir path/to/output
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy.signal import butter, sosfilt
import argparse
import random

matplotlib.use('Agg')

# --- Preprocessing (matches inference.py pipeline) ---

def bandpass_filter(audio, sr, low=50, high=2000, order=5):
    nyq = sr / 2
    sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    return sosfilt(sos, audio)


def preprocess_audio(audio, sr, target_sr=8000):
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = bandpass_filter(audio, target_sr)
    return audio, target_sr


def get_segment(audio, sr, start, end):
    s_start = int(start * sr)
    s_end = int(end * sr)
    segment = audio[s_start:s_end]
    # Pad to 3 seconds if short
    target_len = int(3.0 * sr)
    if len(segment) < target_len:
        segment = np.pad(segment, (0, target_len - len(segment)), mode='wrap')
    elif len(segment) > target_len:
        segment = segment[:target_len]
    return segment


# --- Parsing ---

def parse_annotations(txt_path):
    """Parse ICBHI annotation file. Returns list of (start, end, crackle, wheeze)."""
    cycles = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                start, end = float(parts[0]), float(parts[1])
                crackle, wheeze = int(parts[2]), int(parts[3])
                cycles.append((start, end, crackle, wheeze))
    return cycles


def cycle_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return "Normal"
    elif crackle == 1 and wheeze == 0:
        return "Crackle"
    elif crackle == 0 and wheeze == 1:
        return "Wheeze"
    else:
        return "Both"


def find_samples(data_dir):
    """Find one good sample per class. Prefers longer cycles."""
    data_dir = Path(data_dir)
    txt_files = sorted(data_dir.glob("*.txt"))

    candidates = {"Normal": [], "Crackle": [], "Wheeze": []}

    for txt_path in txt_files:
        wav_path = txt_path.with_suffix('.wav')
        if not wav_path.exists():
            continue

        cycles = parse_annotations(txt_path)
        for start, end, crackle, wheeze in cycles:
            label = cycle_label(crackle, wheeze)
            duration = end - start
            if label in candidates and duration >= 2.0:  # prefer cycles at least 2s
                candidates[label].append((wav_path, start, end, duration))

    # Pick one per class, preferring longer duration
    selected = {}
    for label, items in candidates.items():
        if items:
            items.sort(key=lambda x: x[3], reverse=True)
            # Pick from top 10 randomly for variety
            pick = random.choice(items[:min(10, len(items))])
            selected[label] = pick
        else:
            print(f"Warning: No suitable {label} samples found")

    return selected


# --- Visualization ---

def plot_mel_spectrogram(segment, sr, title, output_path):
    """Generate a mel spectrogram plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, fmax=sr // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
                                    ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Power (dB)')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_mfcc(segment, sr, title, output_path):
    """Generate an MFCC visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)

    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
    fig.colorbar(img, ax=ax, label='Value')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('MFCC', fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_comparison_grid(segments, sr, output_path):
    """4x2 grid: mel spectrogram + MFCC for each class."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    labels = ["Normal", "Crackle", "Wheeze"]

    for i, label in enumerate(labels):
        if label not in segments:
            continue
        segment = segments[label]

        # Top row: Mel spectrogram
        S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, fmax=sr // 2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
                                  ax=axes[0, i], cmap='magma')
        axes[0, i].set_title(label, fontsize=14, fontweight='bold')
        if i == 0:
            axes[0, i].set_ylabel('Mel Spectrogram\nFrequency (Hz)', fontsize=11)
        else:
            axes[0, i].set_ylabel('')

        # Bottom row: MFCC
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1, i], cmap='coolwarm')
        if i == 0:
            axes[1, i].set_ylabel('MFCC', fontsize=11)
        else:
            axes[1, i].set_ylabel('')

    plt.suptitle('Feature Representations Across Respiratory Sound Classes',
                  fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_waveform_comparison(segments, sr, output_path):
    """Waveform comparison across all 4 classes."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=True)

    labels = ["Normal", "Crackle", "Wheeze"]
    colors = ["#0891B2", "#F97316", "#EF4444"]

    for i, label in enumerate(labels):
        if label not in segments:
            continue
        segment = segments[label]
        time_axis = np.linspace(0, len(segment) / sr, len(segment))
        axes[i].plot(time_axis, segment, color=colors[i], linewidth=0.5, alpha=0.85)
        axes[i].set_title(label, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Time (s)', fontsize=11)
        if i == 0:
            axes[i].set_ylabel('Amplitude', fontsize=11)
        axes[i].set_xlim(0, time_axis[-1])

    plt.suptitle('Waveform Comparison Across Classes',
                  fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate MFCC and Mel Spectrogram visuals')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to ICBHI_final_database')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for images')
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir.parent / 'data' / 'ICBHI_final_database'
    output_dir = Path(args.output_dir) if args.output_dir else script_dir.parent / 'visuals'

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Use --data_dir to specify the path to ICBHI_final_database")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print()

    # Find samples
    print("Finding samples per class...")
    selected = find_samples(data_dir)

    for label, (wav_path, start, end, dur) in selected.items():
        print(f"  {label}: {wav_path.name} [{start:.1f}s - {end:.1f}s] ({dur:.1f}s)")

    # Load and preprocess
    print("\nPreprocessing audio...")
    processed_segments = {}

    for label, (wav_path, start, end, _) in selected.items():
        audio, sr = librosa.load(wav_path, sr=None)
        audio, sr = preprocess_audio(audio, sr)
        segment = get_segment(audio, sr, start, end)
        processed_segments[label] = segment
        print(f"  {label}: {len(segment)} samples @ {sr} Hz")

    # Generate individual plots
    print("\nGenerating individual plots...")
    for label, segment in processed_segments.items():
        label_lower = label.lower()
        plot_mel_spectrogram(segment, sr,
                              f'Mel Spectrogram — {label}',
                              output_dir / f'mel_{label_lower}.png')
        plot_mfcc(segment, sr,
                   f'MFCC — {label}',
                   output_dir / f'mfcc_{label_lower}.png')

    # Generate comparison grids
    print("\nGenerating comparison grids...")
    plot_comparison_grid(processed_segments, sr, output_dir / 'comparison_grid.png')
    plot_waveform_comparison(processed_segments, sr, output_dir / 'waveform_comparison.png')

    print(f"\nDone! {len(list(output_dir.glob('*.png')))} images saved to {output_dir}")


if __name__ == '__main__':
    main()