# CLAUDE.md - AusculTek Project Instructions

## What This Project Is
Respiratory sound classification system for CIS 487 capstone. Classifies lung sounds
(Normal/Crackle/Wheeze/Both) from the ICBHI 2017 dataset via a Streamlit web app.

## Current State (as of 2026-03-25)
- **TWO model backends** supported via `load_classifier()` factory in `inference.py`:
  - **Random Forest** (original): 3-class, 8kHz, 3s windows, 130-dim MFCC → `models/best_model/`
  - **PyTorch CNN6** (new): 4-class, 16kHz, 8s windows, mel spectrogram → `models/scl_model/`
- Working Streamlit dashboard in `app.py` (supports both 3-class and 4-class models)
- SCL training package in `audio_preprocessing/scl/` (fully implemented)
- Metadata CSV generated at `data/ICBHI_final_database/metadata.csv` (6,898 cycles)
- Pretrained CNN6 weights at `models/panns_weights/Cnn6_mAP=0.343.pth`
- **Training in progress**: 50-epoch hybrid method on CPU (slow, ~1-3 hours)
  - For full 400-epoch training, use a GPU (Colab recommended)
  - 2-epoch smoke test passed — pipeline verified end-to-end

## What Was Done (SCL Integration)
1. Created `audio_preprocessing/scl/` package (14 files) adapted from ilyassmoummad/scl_icbhi2017
2. Generated `metadata.csv` from ICBHI annotations (60/40 patient-level split, no leakage)
3. Downloaded CNN6 pretrained weights from Zenodo
4. Added `CNNClassifier` class + `load_classifier()` factory to `inference.py`
5. Updated `app.py` for 4-class support (Both=purple #8B5CF6), dynamic sr, factory loading
6. Updated `requirements.txt` with torch, torchaudio, pandas

## What Still Needs to Be Done
- **Complete full training** (50 epochs on CPU finishing, or 400 epochs on GPU)
- **Verify final model accuracy** — target ~55-58% ICBHI score
- **Test Streamlit app** with CNN6 model: `set MODEL_DIR=models/scl_model && streamlit run app.py`
- **Optional**: Try other methods (scl, mscl) and compare results

## Key Technical Facts
- ICBHI data lives in `data/ICBHI_final_database/` (920 .wav + 920 .txt annotation files)
- Annotation format: `{start_time} {end_time} {crackle_0or1} {wheeze_0or1}` per line
- Filename format: `{patient_id}_{recording_index}_{chest_location}_{acq_mode}_{equipment}.wav`
- Patient-level splits are mandatory (prevents data leakage)
- Class imbalance: Normal=3642, Crackle=1864, Wheeze=886, Both=506
- Inverse-frequency class weighting applied in all training methods
- M-SCL uses chest_location/equipment as surrogate metadata (no sex/age data available)

## File Locations

### Original RF Pipeline
- `audio_preprocessing/inference.py` - AudioClassifier + CNNClassifier + load_classifier()
- `audio_preprocessing/train_rf_mfcc.py` - RF training script
- `models/best_model/` - RF model artifacts

### SCL/CNN6 Pipeline
- `audio_preprocessing/scl/` - Full SCL training package:
  - `train.py` - Main entry point
  - `generate_metadata.py` - ICBHI annotations → metadata.csv
  - `models.py` - CNN6/CNN10/CNN14 + Projector + LinearClassifier
  - `dataset.py` - ICBHI PyTorch Dataset
  - `losses.py` - SupConLoss, SupConCELoss
  - `augmentations.py` - SpecAugment
  - `utils.py` - Normalize, Standardize
  - `ce.py`, `scl.py`, `mscl.py`, `hybrid.py` - Training loops
  - `args.py` - CLI argument parser
- `models/scl_model/` - Trained CNN6 model artifacts
- `models/panns_weights/` - Pretrained AudioSet weights
- `data/ICBHI_final_database/metadata.csv` - Generated metadata CSV

### Dashboard & Misc
- `app.py` - Streamlit dashboard (supports both model backends)
- `visuals/` - Generated mel/MFCC comparison images
- `results/` - Training result comparisons
- `context/` - Project documentation

## Commands

### Training
```bash
# Generate metadata (already done)
python -m audio_preprocessing.scl.generate_metadata --datadir data/ICBHI_final_database

# Train hybrid method (recommended)
python -m audio_preprocessing.scl.train --method hybrid --backbone cnn6 --epochs 400 --bs 128 --device cuda:0

# Train on CPU (reduced settings)
python -m audio_preprocessing.scl.train --method hybrid --backbone cnn6 --epochs 50 --bs 32 --device cpu --workers 0

# Other training methods
python -m audio_preprocessing.scl.train --method sl    # Standard cross-entropy
python -m audio_preprocessing.scl.train --method scl   # Supervised contrastive
python -m audio_preprocessing.scl.train --method mscl  # Multi-head contrastive
```

### Inference
```bash
# With RF model (default)
python audio_preprocessing/inference.py

# With CNN6 model
python audio_preprocessing/inference.py --model_dir models/scl_model

# Specific file with annotations
python audio_preprocessing/inference.py --model_dir models/scl_model --audio path/to/file.wav --annotations path/to/file.txt
```

### App
```bash
# With RF model (default)
streamlit run app.py

# With CNN6 model
set MODEL_DIR=models/scl_model && streamlit run app.py
```

## Style Conventions
- Color scheme: dark teal #0C4A6E primary, #0891B2 secondary, #F0F9FF background
- Class colors: Normal=#0891B2, Crackle=#F97316, Wheeze=#EF4444, Both=#8B5CF6
- No emojis in the app
- All Plotly chart text must use explicit dark teal color (#0C4A6E)
- Author: Hayden Banks

## Do NOT
- Use deprecated Plotly APIs (titleside, textfont.color as array)
- Assume Jared has built anything — he hasn't
- Change the 50-2000Hz frequency range — it's clinically motivated
- Use random splits instead of patient-level splits
- Delete the RF model — it's the working fallback
