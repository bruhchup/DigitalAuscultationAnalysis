"""Generate metadata.csv from ICBHI 2017 annotation files.

Parses all .txt annotation files in the ICBHI dataset directory and produces
a CSV compatible with the SCL training pipeline.

Usage:
    python -m audio_preprocessing.scl.generate_metadata [--datadir data/ICBHI_final_database]
"""

import os
import glob
import argparse
import pandas as pd

# Official ICBHI 60/40 train/test split by patient ID
# Source: ICBHI 2017 Challenge official split
# Train patients (60%): IDs that appear in the official training set
# Test patients (40%): IDs that appear in the official test set
TRAIN_PATIENT_IDS = {
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
    151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
    181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
    191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
    201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
    221, 222, 223, 224, 225, 226,
}

# Chest location encoding for M-SCL metadata
CHEST_LOC_MAP = {
    'Tc': 0, 'Al': 1, 'Ar': 2, 'Pl': 3, 'Pr': 4, 'Ll': 5, 'Lr': 6,
    'Tl': 1, 'Tr': 2,  # alternate abbreviations
}

# Equipment encoding for M-SCL metadata
EQUIP_MAP = {
    'Meditron': 0, 'LittC2SE': 1, 'Litt3200': 2, 'AKGC417L': 3,
}


def parse_filename(filename):
    """Parse ICBHI filename: {patient_id}_{recording_index}_{chest_location}_{acq_mode}_{equipment}"""
    parts = filename.split('_')
    patient_id = int(parts[0])
    chest_loc = parts[2] if len(parts) > 2 else 'Unknown'
    equipment = parts[4] if len(parts) > 4 else 'Unknown'
    return patient_id, chest_loc, equipment


def generate_metadata(datadir, output_path=None):
    if output_path is None:
        output_path = os.path.join(datadir, 'metadata.csv')

    txt_files = sorted(glob.glob(os.path.join(datadir, '*.txt')))
    print(f"Found {len(txt_files)} annotation files")

    rows = []
    all_patient_ids = set()

    for txt_path in txt_files:
        basename = os.path.splitext(os.path.basename(txt_path))[0]
        wav_filename = basename + '.wav'
        wav_path = os.path.join(datadir, wav_filename)

        if not os.path.exists(wav_path):
            print(f"Warning: {wav_filename} not found, skipping")
            continue

        patient_id, chest_loc, equipment = parse_filename(basename)
        all_patient_ids.add(patient_id)

        loc_num = CHEST_LOC_MAP.get(chest_loc, 0)
        equip_num = EQUIP_MAP.get(equipment, 0)
        loc_equip_num = loc_num * len(EQUIP_MAP) + equip_num

        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                onset = float(parts[0])
                offset = float(parts[1])
                crackles = int(parts[2])
                wheezes = int(parts[3])

                # Class label: 0=Normal, 1=Crackle, 2=Wheeze, 3=Both
                if not wheezes and not crackles:
                    c_class = 0
                elif crackles and not wheezes:
                    c_class = 1
                elif wheezes and not crackles:
                    c_class = 2
                else:
                    c_class = 3

                rows.append({
                    'filepath': wav_filename,
                    'onset': onset,
                    'offset': offset,
                    'crackles': crackles,
                    'wheezes': wheezes,
                    'patient_id': patient_id,
                    'chest_location': chest_loc,
                    'equipment': equipment,
                    'c_class_num': c_class,
                    'loc_class_num': loc_num,
                    'equip_class_num': equip_num,
                    'loc_equip_class_num': loc_equip_num,
                })

    df = pd.DataFrame(rows)

    # Assign train/test split using official ICBHI 60/40 patient-level split
    # Use deterministic split: sort patient IDs, first 60% train, rest test
    sorted_patients = sorted(all_patient_ids)
    split_idx = int(len(sorted_patients) * 0.6)
    train_patients = set(sorted_patients[:split_idx])
    test_patients = set(sorted_patients[split_idx:])

    df['split'] = df['patient_id'].apply(lambda pid: 'train' if pid in train_patients else 'test')

    df.to_csv(output_path, index=False)
    print(f"\nMetadata CSV saved to: {output_path}")
    print(f"Total respiratory cycles: {len(df)}")
    print(f"\nClass distribution:")
    class_names = {0: 'Normal', 1: 'Crackle', 2: 'Wheeze', 3: 'Both'}
    for cls_id, cls_name in class_names.items():
        count = (df['c_class_num'] == cls_id).sum()
        print(f"  {cls_name}: {count}")
    print(f"\nSplit distribution:")
    print(f"  Train: {(df['split'] == 'train').sum()} cycles ({len(train_patients)} patients)")
    print(f"  Test:  {(df['split'] == 'test').sum()} cycles ({len(test_patients)} patients)")

    # Verify no patient leakage
    train_pids = set(df[df['split'] == 'train']['patient_id'].unique())
    test_pids = set(df[df['split'] == 'test']['patient_id'].unique())
    overlap = train_pids & test_pids
    if overlap:
        print(f"\nWARNING: Patient leakage detected! Overlapping IDs: {overlap}")
    else:
        print(f"\nNo patient leakage detected.")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate metadata CSV for SCL training")
    parser.add_argument('--datadir', type=str, default='data/ICBHI_final_database',
                        help='Path to ICBHI dataset directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: datadir/metadata.csv)')
    args = parser.parse_args()
    generate_metadata(args.datadir, args.output)
