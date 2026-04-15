# AusculTek - Project Context Document
# Last updated: 2026-03-25
# Purpose: Handoff document for Claude Code CLI / Cowork continuation

## PROJECT OVERVIEW

AusculTek is a CIS 487 capstone project building a respiratory sound classification
system using the ICBHI 2017 dataset. The goal is to classify lung sounds as Normal,
Crackle, Wheeze, and Both via a web-based Streamlit interface.

Team: Hayden Banks (all technical work), Jared Tauler (minimal contribution)
University: Shepherd University, Dept of CS Math and ENGR
Timeline: 8-week capstone, midterm completed 02/25/2026

---

## CURRENT STATE (2026-03-25)

### Phase 1 Complete: Random Forest Baseline
1. **Audio Preprocessing Pipeline** - loads ICBHI 2017, splits into respiratory cycles,
   resamples to 8kHz, applies 50-2000Hz bandpass filter (order 5), patient-level train/test splits
2. **Random Forest Classifier** - 58% test accuracy, 48% balanced accuracy, 130-dim
   MFCC feature vectors (20 MFCCs mean+std + deltas mean+std + delta-deltas mean+std + spectral features)
3. **Inference Pipeline** (`audio_preprocessing/inference.py`) - `AudioClassifier` class
   that accepts a .wav file, segments via sliding window (3s window, 1.5s hop) or ICBHI
   annotations, classifies each segment, returns dict with per-cycle results
4. **Streamlit Dashboard** (`app.py`) - upload .wav files, displays confidence gauge,
   donut chart, annotated waveform, probability heatmap, confidence bar chart, segment
   detail table, session history sidebar

### Phase 2 In Progress: SCL/CNN6 Integration
Replicating the approach from "Pretraining Respiratory Sound Representations using
Metadata and Contrastive Learning" (Moummad & Farrugia, WASPAA 2023).

**What's done:**
- `audio_preprocessing/scl/` package created (14 files) — adapted from ilyassmoummad/scl_icbhi2017
- `data/ICBHI_final_database/metadata.csv` generated (6,898 cycles, 4 classes, 60/40 patient split)
- CNN6 pretrained weights downloaded to `models/panns_weights/Cnn6_mAP=0.343.pth`
- `inference.py` updated with `CNNClassifier` class and `load_classifier()` factory
- `app.py` updated for 4-class support (Both=purple), dynamic sample rate, factory loading
- `requirements.txt` updated with torch, torchaudio, pandas
- 2-epoch smoke test passed — full pipeline verified end-to-end
- 50-epoch hybrid training started on CPU (running, slow without GPU)

**What's remaining:**
- Complete training (50 epochs on CPU, or 400 epochs on GPU for best results)
- Verify final ICBHI score (target: 55-58%)
- Full app testing with CNN6 model
- Optional: compare sl/scl/mscl/hybrid methods

---

## MODEL BACKENDS

### Backend 1: Random Forest (Original)
- Location: `models/best_model/`
- Files: model.pkl, scaler.pkl, preprocessing_config.json, model_results.json
- Config: 8kHz, 3s windows, 130-dim MFCC, 3 classes (Normal/Crackle/Wheeze)
- Accuracy: 58% test, 48% balanced, 90% train (32% overfit gap)
- Per-class recall: Normal=67%, Crackle=55%, Wheeze=21%

### Backend 2: CNN6 + Supervised Contrastive Learning (New)
- Location: `models/scl_model/`
- Files: model_state.pth, preprocessing_config.json, model_results.json
- Config: 16kHz, 8s windows, 64-mel spectrogram, 4 classes (Normal/Crackle/Wheeze/Both)
- Architecture: CNN6 from PANNs (pretrained on AudioSet), ~4.3M params
- Training: Hybrid method (SupConLoss + CrossEntropyLoss, alpha=0.5)
- Target: 55-58% ICBHI score = (Sensitivity + Specificity) / 2

Both backends share the same `classify()` return dict format, allowing seamless
switching via `MODEL_DIR` environment variable.

---

## PROJECT STRUCTURE
```
DigitalAuscultationAnalysis/
├── app.py                          # Streamlit web dashboard (both backends)
├── requirements.txt                # Python dependencies (includes torch)
├── README.md                       # Project readme
├── audio_preprocessing/
│   ├── inference.py                # AudioClassifier + CNNClassifier + load_classifier()
│   ├── train_rf_mfcc.py            # Random Forest MFCC training
│   ├── train_multi_approach.py     # Multi-approach comparison
│   ├── train_mfcc_comparison.py    # MFCC comparison
│   ├── train_mel_comparison.py     # Mel comparison
│   ├── train_original_model.py     # Original CNN model
│   ├── generate_visuals.py         # Mel/MFCC visual generation
│   └── scl/                        # SCL training package (NEW)
│       ├── __init__.py
│       ├── train.py                # Main training entry point
│       ├── generate_metadata.py    # ICBHI annotations → metadata.csv
│       ├── args.py                 # CLI argument parser
│       ├── models.py               # CNN6/CNN10/CNN14 + Projector + LinearClassifier
│       ├── dataset.py              # ICBHI PyTorch Dataset
│       ├── losses.py               # SupConLoss, SupConCELoss
│       ├── augmentations.py        # SpecAugment
│       ├── utils.py                # Normalize, Standardize
│       ├── ce.py                   # Cross-entropy training loop
│       ├── scl.py                  # Supervised contrastive + linear eval
│       ├── mscl.py                 # Multi-head contrastive + linear eval
│       └── hybrid.py               # Joint contrastive + CE training
├── models/
│   ├── best_model/                 # Random Forest model artifacts
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   ├── preprocessing_config.json
│   │   └── model_results.json
│   ├── scl_model/                  # CNN6 model artifacts (NEW)
│   │   ├── model_state.pth
│   │   ├── preprocessing_config.json
│   │   └── model_results.json
│   └── panns_weights/              # Pretrained AudioSet weights (NEW)
│       └── Cnn6_mAP=0.343.pth
├── data/
│   └── ICBHI_final_database/       # Raw ICBHI dataset
│       ├── *.wav (920 files)
│       ├── *.txt (920 annotation files)
│       ├── metadata.csv            # Generated for SCL training (NEW)
│       └── *.pth                   # Cached dataset tensors (auto-generated)
├── visuals/                        # Generated comparison images
├── results/                        # Training result comparisons
├── proposal/                       # Project proposal documents
└── context/                        # Project documentation
    ├── CLAUDE.md
    └── PROJECT_CONTEXT.md
```

---

## DATASET DETAILS: ICBHI 2017

- 920 recordings from 126 patients
- 6,898 respiratory cycles total
- Class distribution: Normal=3642 (53%), Crackle=1864 (27%), Wheeze=886 (13%), Both=506 (7%)
- Recording duration: 10-90 seconds each
- Equipment: AKG C417L, 3M Littmann Classic II SE, Littmann 3200, WelchAllyn Meditron
- Annotation: start_time end_time crackle(0/1) wheeze(0/1)
- Patient diagnoses: COPD, LRTI, URTI, Bronchiectasis, Pneumonia, Bronchiolitis, Asthma, Healthy

### Metadata CSV (data/ICBHI_final_database/metadata.csv)
Generated from annotations. Columns: filepath, onset, offset, crackles, wheezes,
patient_id, chest_location, equipment, c_class_num, loc_class_num, equip_class_num,
loc_equip_class_num, split

Split: 60/40 by patient ID (75 train patients / 51 test patients)
Train: 4,307 cycles | Test: 2,591 cycles | No patient leakage

---

## SCL TRAINING REFERENCE

### Source
Paper: "Pretraining Respiratory Sound Representations using Metadata and Contrastive Learning"
Authors: Moummad & Farrugia, WASPAA 2023
Repo: https://github.com/ilyassmoummad/scl_icbhi2017

### Training Methods
| Method | Description | Published Score |
|--------|-------------|-----------------|
| sl     | Standard cross-entropy | ~52% |
| scl    | Contrastive pretrain → frozen linear eval | ~55% |
| mscl   | Multi-head contrastive (metadata) → linear eval | ~58% |
| hybrid | Joint contrastive + CE (recommended) | ~57% |

### Key Hyperparameters
- Temperature tau=0.06 (very low, tuned)
- Alpha=0.5 (CE vs contrastive tradeoff for hybrid)
- Lambda=0.75 (class vs metadata tradeoff for mscl)
- LR=1e-4 (encoder), LR2=0.1 (linear eval)
- Cosine annealing scheduler, T_max=40, eta_min=1e-6
- Batch size: 128 (GPU), 32 (CPU fallback)

### Adaptations Made
- Relative imports for package structure
- Device passed as parameter (no global args dependency in training loops)
- Best model checkpointing (saves best ICBHI score, not just final state)
- M-SCL metadata uses chest_location/equipment from filenames (no sex/age in our dataset)
- 60/40 deterministic patient-level split (sorted patient IDs)

---

## KEY LEARNINGS & PRINCIPLES

1. **Preprocessing must match data characteristics** - Fava et al. VMD pipeline failed
   catastrophically. Simpler approaches win.
2. **Class imbalance is the core challenge** - Wheeze detection has low recall.
   All training uses inverse-frequency class weighting.
3. **Patient-level splits are mandatory** - Prevents data leakage.
4. **ICBHI benchmarks are hard** - ~58-64% ICBHI score is competitive with SOTA.
5. **Filter on full recording before segmenting** - Butterworth filters have edge artifacts
   on short clips.
6. **Circular padding** is better than zero-padding for respiratory sounds (quasi-periodic).

---

## STREAMLIT APP DETAILS

### Color Scheme:
- Primary: #0C4A6E (dark teal)
- Secondary: #0891B2 (teal)
- Background: #F0F9FF (ice blue)
- Normal: #0891B2 | Crackle: #F97316 | Wheeze: #EF4444 | Both: #8B5CF6

### Known Issues in app.py:
- Session history stores full results dicts with no cap (memory grows unbounded)
- Temp file cleanup is not in a try/finally block (leak on exception)

### Fixed Issues (2026-03-25):
- `sr=8000` hardcoded → now reads from classifier config
- Pie chart only handled 3 classes → now dynamically uses CLASS_COLORS keys
- Segment table hardcoded 3 class columns → now dynamic
- Used AudioClassifier directly → now uses load_classifier() factory

---

## DEPENDENCIES (requirements.txt)
```
streamlit>=1.30.0
plotly>=5.18.0
librosa==0.11.0
matplotlib==3.10.1
numpy==2.3.5
scikit-learn==1.6.1
scipy==1.15.2
seaborn==0.13.2
tensorflow==2.19.0
torch>=2.0.0
torchaudio>=2.0.0
pandas
```

Note: On Windows CPU, install torch with: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`
For CUDA: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`

---

## REMAINING TIMELINE TO FINAL

Weeks 5-6 (current): Complete CNN6 training, verify results, test full pipeline
Weeks 7-8: Final presentation, report, optional app enhancements
Stretch: History tracking, report generation, try mscl/scl methods for comparison
