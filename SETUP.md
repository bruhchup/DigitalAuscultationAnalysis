# AusculTek -- Setup Guide

Instructions for setting up the project on a new machine after cloning.

---

## 1. Clone the repo

```bash
git clone https://github.com/bruhchup/DigitalAuscultationAnalysis.git
cd DigitalAuscultationAnalysis
```

---

## 2. Python environment

Requires Python 3.12+.

```bash
pip install flask flask-cors librosa soundfile torch torchaudio pandas scipy scikit-learn streamlit plotly matplotlib seaborn
```

For CPU-only torch (no GPU):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install flask flask-cors librosa soundfile pandas scipy scikit-learn streamlit plotly matplotlib seaborn
```

---

## 3. Files not in git

The following are in `.gitignore` and must be obtained separately:

### ICBHI dataset

Place the ICBHI 2017 dataset at `data/ICBHI_final_database/`. It should contain 920 `.wav` files and 920 `.txt` annotation files. Generate the metadata CSV if it doesn't exist:

```bash
python -m audio_preprocessing.scl.generate_metadata --datadir data/ICBHI_final_database
```

### Model weights

Copy the `models/` directory from an existing machine or download:

- **CNN6 trained model:** `models/scl_model/model_state.pth`, `model_results.json`, `preprocessing_config.json`
  - If you don't have these, run the Colab training notebook (`notebooks/train_scl_colab_new.ipynb`) and download the resulting `scl_model.zip`
- **PANNs pretrained weights** (only needed for training): `models/panns_weights/Cnn6_mAP=0.343.pth`
  - Downloaded automatically by the Colab notebook, or get from Zenodo record 3987831
- **Random Forest model** (optional fallback): `models/best_model/model.pkl`, `scaler.pkl`, `preprocessing_config.json`, `model_results.json`

---

## 4. Running the Streamlit dashboard

```bash
# With CNN6 model (recommended)
set MODEL_DIR=models/scl_model && streamlit run app.py

# With Random Forest model (fallback)
streamlit run app.py
```

Opens at http://localhost:8501

---

## 5. Running the mobile app

### Start the API server

```bash
python api.py --model_dir models/scl_model
```

This starts a Flask server on port 5000. Note the IP address printed (e.g. `http://192.168.x.x:5000`).

### Update the API address

Edit `mobile/src/services/api.js` and set `API_BASE` to your machine's local IP:

```js
const API_BASE = "http://YOUR_LOCAL_IP:5000";
```

Your phone and your machine must be on the same Wi-Fi network.

### Start Expo

```bash
cd mobile
npm install
npx expo start
```

Scan the QR code with Expo Go on your phone.

---

## 6. Training a new model (Google Colab)

1. Upload `ICBHI_final_database.zip` to Google Drive under `My Drive/data/`
2. Open `notebooks/train_scl_colab_new.ipynb` in Google Colab
3. Set runtime to T4 GPU (Runtime > Change runtime type)
4. Run all cells -- training takes ~1-2 hours for 400 epochs
5. Download the resulting `scl_model.zip` and extract into `models/scl_model/`

---

## 7. Project structure quick reference

```
app.py                    -- Streamlit web dashboard
api.py                    -- Flask REST API for mobile app
audio_preprocessing/
  inference.py            -- AudioClassifier + CNNClassifier + load_classifier()
  scl/                    -- SCL training package (train.py is entry point)
mobile/                   -- React Native (Expo SDK 54) mobile app
  src/services/api.js     -- API_BASE address config
  src/screens/            -- HomeScreen, ResultsScreen, HistoryScreen
models/                   -- (not in git) model weights
data/                     -- (not in git) ICBHI dataset
notebooks/                -- Colab training notebook
```
