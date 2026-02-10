"""
Split ICBHI 2017 Dataset by Category
Extracts individual respiratory cycles and organizes them into:
- normal/ (no crackles, no wheezes)
- crackle/ (crackles only)
- wheeze/ (wheezes only)
- both/ (both crackles and wheezes)
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import pandas as pd
from collections import Counter
from tqdm import tqdm


def parse_annotation_file(txt_path):
    """
    Parse annotation file to extract respiratory cycles
    
    Args:
        txt_path: Path to .txt annotation file
    
    Returns:
        List of dictionaries with cycle information
    """
    cycles = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                start = float(parts[0])
                end = float(parts[1])
                crackles = int(parts[2])
                wheezes = int(parts[3])
                
                # Determine category
                if crackles == 1 and wheezes == 1:
                    category = 'both'
                elif crackles == 1:
                    category = 'crackle'
                elif wheezes == 1:
                    category = 'wheeze'
                else:
                    category = 'normal'
                
                cycles.append({
                    'start': start,
                    'end': end,
                    'crackles': crackles,
                    'wheezes': wheezes,
                    'category': category,
                    'duration': end - start
                })
    
    return cycles


def extract_and_save_cycle(audio, sr, cycle_info, output_path, cycle_idx, source_filename):
    """
    Extract a single cycle from audio and save it
    
    Args:
        audio: Full audio array
        sr: Sample rate
        cycle_info: Dictionary with start/end times
        output_path: Directory to save the file
        cycle_idx: Index of this cycle in the recording
        source_filename: Original filename (without extension)
    
    Returns:
        Path to saved file
    """
    # Calculate sample indices
    start_sample = int(cycle_info['start'] * sr)
    end_sample = int(cycle_info['end'] * sr)
    
    # Extract cycle
    cycle_audio = audio[start_sample:end_sample]
    
    # Create filename
    # Format: originalname_cycle###_duration.wav
    duration_str = f"{cycle_info['duration']:.2f}s"
    output_filename = f"{source_filename}_cycle{cycle_idx:03d}_{duration_str}.wav"
    output_filepath = os.path.join(output_path, output_filename)
    
    # Save audio file
    sf.write(output_filepath, cycle_audio, sr)
    
    return output_filepath


def split_dataset_by_category(data_dir, output_dir, resample_rate=None):
    """
    Main function to split dataset into categories
    
    Args:
        data_dir: Directory containing ICBHI .wav and .txt files
        output_dir: Directory where categorized cycles will be saved
        resample_rate: Target sample rate (None = keep original)
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Check if data directory exists
    if not data_path.exists():
        print(f"ERROR: Data directory does not exist: {data_dir}")
        print(f"Absolute path checked: {data_path.absolute()}")
        return None
    
    # Create output directories
    categories = ['normal', 'crackle', 'wheeze', 'both']
    category_paths = {}
    
    for category in categories:
        cat_path = output_path / category
        cat_path.mkdir(parents=True, exist_ok=True)
        category_paths[category] = cat_path
    
    print("="*70)
    print("SPLITTING ICBHI 2017 DATASET BY CATEGORY")
    print("="*70)
    print(f"\nSource directory: {data_dir}")
    print(f"Absolute path: {data_path.absolute()}")
    print(f"Output directory: {output_dir}")
    print(f"Output absolute path: {output_path.absolute()}")
    if resample_rate:
        print(f"Resampling to: {resample_rate} Hz")
    print()
    
    # Get all annotation files
    txt_files = sorted(data_path.glob('*.txt'))
    
    # Debug: List what we found
    print(f"Searching for .txt files in: {data_path}")
    print(f"Found {len(txt_files)} annotation files")
    
    if len(txt_files) == 0:
        print("\nERROR: No .txt files found!")
        print("\nTroubleshooting:")
        print("1. Check if the data directory path is correct")
        print("2. List all files in the directory:")
        
        all_files = list(data_path.glob('*'))
        print(f"\n   Total files in directory: {len(all_files)}")
        
        if len(all_files) > 0:
            print("\n   First 10 files found:")
            for f in all_files[:10]:
                print(f"   - {f.name}")
        
        return None
    
    print(f"\nFirst 5 files to process:")
    for f in txt_files[:5]:
        print(f"  - {f.name}")
    print()
    
    # Statistics
    stats = {
        'total_recordings': 0,
        'total_cycles': 0,
        'cycles_per_category': Counter(),
        'duration_per_category': {cat: 0.0 for cat in categories},
        'files_processed': 0,
        'files_skipped': 0,
        'errors': []
    }
    
    # Process each recording
    print("Processing recordings...")
    for txt_file in tqdm(txt_files, desc="Extracting cycles"):
        wav_file = txt_file.with_suffix('.wav')
        
        # Check if corresponding wav file exists
        if not wav_file.exists():
            stats['files_skipped'] += 1
            stats['errors'].append(f"Missing .wav for {txt_file.name}")
            continue
        
        try:
            # Load audio
            audio, sr = librosa.load(wav_file, sr=resample_rate)
            
            # Parse annotations
            cycles = parse_annotation_file(txt_file)
            
            # Extract source filename (without extension)
            source_filename = txt_file.stem
            
            # Extract and save each cycle
            for idx, cycle in enumerate(cycles, start=1):
                category = cycle['category']
                
                # Save cycle
                extract_and_save_cycle(
                    audio=audio,
                    sr=sr,
                    cycle_info=cycle,
                    output_path=category_paths[category],
                    cycle_idx=idx,
                    source_filename=source_filename
                )
                
                # Update statistics
                stats['cycles_per_category'][category] += 1
                stats['duration_per_category'][category] += cycle['duration']
            
            stats['total_recordings'] += 1
            stats['total_cycles'] += len(cycles)
            stats['files_processed'] += 1
            
        except Exception as e:
            stats['files_skipped'] += 1
            stats['errors'].append(f"Error processing {txt_file.name}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE - SUMMARY")
    print("="*70)
    print(f"\nTotal recordings processed: {stats['files_processed']}")
    print(f"Total recordings skipped: {stats['files_skipped']}")
    print(f"Total cycles extracted: {stats['total_cycles']}")
    
    if stats['total_cycles'] > 0:
        print(f"\n{'Category':<15} {'Cycles':>10} {'Percentage':>12} {'Total Duration':>15}")
        print("-"*55)
        
        for category in categories:
            count = stats['cycles_per_category'][category]
            percentage = (count / stats['total_cycles'] * 100) if stats['total_cycles'] > 0 else 0
            duration = stats['duration_per_category'][category]
            print(f"{category.capitalize():<15} {count:>10} {percentage:>11.2f}% {duration:>14.2f}s")
        
        print(f"\n{'Total':<15} {stats['total_cycles']:>10} {'100.00%':>12}")
        
        # Print category directories
        print(f"\n{'='*70}")
        print("OUTPUT DIRECTORIES")
        print("="*70)
        for category in categories:
            cat_path = category_paths[category]
            num_files = len(list(cat_path.glob('*.wav')))
            print(f"{category.capitalize():<15} {num_files:>6} files -> {cat_path}")
    
    # Print errors if any
    if stats['errors']:
        print(f"\n{'='*70}")
        print("ERRORS AND WARNINGS")
        print("="*70)
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  * {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    # Save metadata CSV
    if stats['total_cycles'] > 0:
        metadata_path = output_path / 'split_metadata.csv'
        save_metadata(category_paths, metadata_path, stats)
        print(f"\nMetadata saved to: {metadata_path}")
    
    return stats


def save_metadata(category_paths, output_file, stats):
    """
    Save metadata about the split dataset to CSV
    """
    records = []
    
    for category, cat_path in category_paths.items():
        wav_files = list(cat_path.glob('*.wav'))
        
        for wav_file in wav_files:
            # Parse filename
            # Format: originalname_cycle###_duration.wav
            parts = wav_file.stem.split('_cycle')
            source_file = parts[0] if len(parts) > 1 else wav_file.stem
            
            # Extract duration from filename
            duration_str = wav_file.stem.split('_')[-1].replace('s', '')
            try:
                duration = float(duration_str)
            except:
                duration = None
            
            # Get audio info
            try:
                audio_info = sf.info(wav_file)
                sr = audio_info.samplerate
                actual_duration = audio_info.duration
            except:
                sr = None
                actual_duration = None
            
            records.append({
                'filename': wav_file.name,
                'category': category,
                'source_recording': source_file,
                'duration_seconds': actual_duration,
                'sample_rate': sr,
                'filepath': str(wav_file)
            })
    
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    
    print(f"\nTotal cycle files in metadata: {len(df)}")


def analyze_category_distribution(output_dir):
    """
    Analyze the distribution of cycles across categories
    """
    output_path = Path(output_dir)
    metadata_file = output_path / 'split_metadata.csv'
    
    if not metadata_file.exists():
        print("Metadata file not found. Run split_dataset_by_category first.")
        return
    
    df = pd.DataFrame(pd.read_csv(metadata_file))
    
    print("\n" + "="*70)
    print("CATEGORY DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Overall statistics
    print(f"\nTotal cycles: {len(df)}")
    print(f"Total duration: {df['duration_seconds'].sum():.2f} seconds")
    print(f"Average cycle duration: {df['duration_seconds'].mean():.2f} seconds")
    
    # Per-category statistics
    print("\nPer-Category Statistics:")
    print(f"{'Category':<15} {'Count':>8} {'%':>8} {'Min (s)':>10} {'Max (s)':>10} {'Mean (s)':>10}")
    print("-"*70)
    
    for category in ['normal', 'crackle', 'wheeze', 'both']:
        cat_df = df[df['category'] == category]
        if len(cat_df) > 0:
            count = len(cat_df)
            percentage = (count / len(df)) * 100
            min_dur = cat_df['duration_seconds'].min()
            max_dur = cat_df['duration_seconds'].max()
            mean_dur = cat_df['duration_seconds'].mean()
            
            print(f"{category.capitalize():<15} {count:>8} {percentage:>7.2f}% {min_dur:>10.2f} {max_dur:>10.2f} {mean_dur:>10.2f}")
    
    # Sample rate distribution
    print("\nSample Rate Distribution:")
    sr_counts = df['sample_rate'].value_counts()
    for sr, count in sr_counts.items():
        print(f"  {int(sr)} Hz: {count} files")
    
    # Duration distribution
    print("\nDuration Distribution:")
    bins = [0, 1, 2, 3, 5, 10, float('inf')]
    labels = ['0-1s', '1-2s', '2-3s', '3-5s', '5-10s', '10s+']
    df['duration_bin'] = pd.cut(df['duration_seconds'], bins=bins, labels=labels)
    
    for label in labels:
        count = len(df[df['duration_bin'] == label])
        percentage = (count / len(df)) * 100
        print(f"  {label:8} {count:>6} cycles ({percentage:>5.2f}%)")


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Build relative paths from script location
    # Script is in: audio_preprocessing/
    # Data is in: data/ICBHI_final_database/
    # So we need to go up one level (..) then into data/
    
    DATA_DIR = script_dir.parent / "data" / "ICBHI_final_database"
    OUTPUT_DIR = script_dir.parent / "data" / "categorized_cycles"
    RESAMPLE_RATE = 10000  # 10 kHz (set to None to keep original)
    
    print("Starting dataset split process...")
    print(f"Script location: {script_dir}")
    print(f"Looking for data in: {DATA_DIR}")
    print()
    
    # Verify path exists before starting
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory does not exist!")
        print(f"Expected path: {DATA_DIR.absolute()}")
        print()
        print("Please ensure your directory structure is:")
        print("  DigitalAuscultationAnalysis/")
        print("    audio_preprocessing/")
        print("      split_by_category.py  <- this script")
        print("    data/")
        print("      ICBHI_final_database/")
        print("        *.txt and *.wav files")
        exit(1)
    
    # Run the split
    stats = split_dataset_by_category(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        resample_rate=RESAMPLE_RATE
    )
    
    # Only analyze if we successfully processed files
    if stats and stats['total_cycles'] > 0:
        # Analyze the results
        analyze_category_distribution(OUTPUT_DIR)
    else:
        print("\nNo files were processed. Please check the data directory path.")
