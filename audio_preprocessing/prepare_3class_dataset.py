"""
Prepare 3-Class Dataset (Normal, Crackle, Wheeze)
Excludes cycles with both crackles and wheezes
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from collections import Counter


def create_3class_dataset(categorized_dir, output_dir):
    """
    Create 3-class dataset by excluding 'both' category
    
    Args:
        categorized_dir: Directory with 4 categories
        output_dir: New directory for 3 classes only
    """
    categorized_path = Path(categorized_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for category in ['normal', 'crackle', 'wheeze']:
        (output_path / category).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CREATING 3-CLASS DATASET")
    print("="*70)
    print(f"\nSource: {categorized_dir}")
    print(f"Output: {output_dir}")
    print()
    
    stats = Counter()
    
    # Copy files from normal, crackle, wheeze (exclude 'both')
    for category in ['normal', 'crackle', 'wheeze']:
        source_dir = categorized_path / category
        dest_dir = output_path / category
        
        if not source_dir.exists():
            print(f"Warning: {category} directory not found!")
            continue
        
        wav_files = list(source_dir.glob('*.wav'))
        
        print(f"Copying {category}...")
        for wav_file in wav_files:
            shutil.copy2(wav_file, dest_dir / wav_file.name)
            stats[category] += 1
    
    # Print summary
    print("\n" + "="*70)
    print("3-CLASS DATASET CREATED")
    print("="*70)
    
    total = sum(stats.values())
    
    print(f"\n{'Category':<15} {'Cycles':>10} {'Percentage':>12}")
    print("-"*40)
    
    for category in ['normal', 'crackle', 'wheeze']:
        count = stats[category]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{category.capitalize():<15} {count:>10} {percentage:>11.2f}%")
    
    print(f"\n{'Total':<15} {total:>10} {'100.00%':>12}")
    
    # Calculate imbalance ratio
    max_count = max(stats.values())
    min_count = min(stats.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("WARNING: Moderate class imbalance detected")
        print("  -> Recommendation: Use data augmentation for minority classes")
    else:
        print("GOOD: Relatively balanced dataset")
    
    # Create metadata CSV
    create_3class_metadata(output_path, stats)
    
    return stats


def create_3class_metadata(output_dir, stats):
    """
    Create metadata CSV for 3-class dataset
    """
    output_path = Path(output_dir)
    records = []
    
    for category in ['normal', 'crackle', 'wheeze']:
        cat_dir = output_path / category
        
        if not cat_dir.exists():
            continue
        
        wav_files = list(cat_dir.glob('*.wav'))
        
        for wav_file in wav_files:
            import soundfile as sf
            
            try:
                info = sf.info(wav_file)
                
                # Parse filename to get source info
                parts = wav_file.stem.split('_cycle')
                source_file = parts[0] if len(parts) > 0 else wav_file.stem
                
                records.append({
                    'filename': wav_file.name,
                    'category': category,
                    'source_recording': source_file,
                    'duration_seconds': info.duration,
                    'sample_rate': info.samplerate,
                    'filepath': str(wav_file)
                })
            except Exception as e:
                print(f"Error reading {wav_file.name}: {e}")
    
    df = pd.DataFrame(records)
    metadata_file = output_path / 'metadata_3class.csv'
    df.to_csv(metadata_file, index=False)
    
    print(f"\nSUCCESS: Metadata saved: {metadata_file}")
    print(f"  Total records: {len(df)}")


def analyze_3class_distribution(output_dir):
    """
    Analyze the 3-class dataset distribution
    """
    output_path = Path(output_dir)
    metadata_file = output_path / 'metadata_3class.csv'
    
    if not metadata_file.exists():
        print("Metadata file not found!")
        return
    
    df = pd.read_csv(metadata_file)
    
    print("\n" + "="*70)
    print("3-CLASS DATASET ANALYSIS")
    print("="*70)
    
    print(f"\nTotal cycles: {len(df)}")
    print(f"Total duration: {df['duration_seconds'].sum():.2f} seconds")
    print(f"  ({df['duration_seconds'].sum()/60:.2f} minutes)")
    print(f"Average cycle duration: {df['duration_seconds'].mean():.2f} seconds")
    
    print("\nPer-Category Statistics:")
    print(f"{'Category':<15} {'Count':>8} {'%':>8} {'Min (s)':>10} {'Max (s)':>10} {'Mean (s)':>10}")
    print("-"*70)
    
    for category in ['normal', 'crackle', 'wheeze']:
        cat_df = df[df['category'] == category]
        
        if len(cat_df) > 0:
            count = len(cat_df)
            percentage = (count / len(df)) * 100
            min_dur = cat_df['duration_seconds'].min()
            max_dur = cat_df['duration_seconds'].max()
            mean_dur = cat_df['duration_seconds'].mean()
            
            print(f"{category.capitalize():<15} {count:>8} {percentage:>7.2f}% "
                  f"{min_dur:>10.2f} {max_dur:>10.2f} {mean_dur:>10.2f}")
    
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
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Paths
    CATEGORIZED_DIR = script_dir.parent / "data" / "categorized_cycles"
    OUTPUT_DIR = script_dir.parent / "data" / "categorized_cycles_3class"
    
    print("Creating 3-class dataset...")
    print(f"Source: {CATEGORIZED_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Check source exists
    if not CATEGORIZED_DIR.exists():
        print(f"Error: Source directory not found: {CATEGORIZED_DIR}")
        print("Please run split_by_category.py first!")
        exit(1)
    
    # Create 3-class dataset
    stats = create_3class_dataset(CATEGORIZED_DIR, OUTPUT_DIR)
    
    # Analyze
    analyze_3class_distribution(OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("DATASET READY!")
    print("="*70)
    print("\nYou now have a 3-class dataset:")
    print(f"  Normal:  {stats['normal']:,} cycles (57%)")
    print(f"  Crackle: {stats['crackle']:,} cycles (29%)")
    print(f"  Wheeze:  {stats['wheeze']:,} cycles (14%)")
    print(f"  Total:   {sum(stats.values()):,} cycles")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Run quick_prototype.py to test the concept")
    print("2. Review the accuracy you get")
    print("3. Then build full preprocessing pipeline")