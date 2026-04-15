# AusculTek Mobile (React Native / Expo)

Prototype mobile app for respiratory sound classification. Connects to the Flask API backend that wraps the existing inference pipeline.

## Architecture

```
mobile/               React Native (Expo) app
  App.js              Navigation + history state
  src/
    screens/
      HomeScreen      Record audio or upload .wav
      ResultsScreen   Classification results dashboard
      HistoryScreen   Past analysis results
    components/
      ClassBadge      Color-coded label + confidence
      MetricCard      Stat card (segments, duration, etc.)
      SegmentRow      Per-segment result row with bar
    services/
      api.js          API client (fetch wrapper)

api.py                Flask REST API (project root)
```

## Setup

### 1. Backend API

From the project root:

```bash
pip install flask flask-cors
python api.py
# API runs on http://0.0.0.0:5000
```

### 2. Mobile App

```bash
cd mobile
npm install
npx expo start
```

Scan the QR code with Expo Go (Android/iOS) or press `a` for Android emulator / `i` for iOS simulator.

### 3. Configure API URL

Edit `mobile/src/services/api.js` and set `API_BASE`:

| Platform          | URL                              |
|-------------------|----------------------------------|
| Android emulator  | `http://10.0.2.2:5000`           |
| iOS simulator     | `http://localhost:5000`           |
| Physical device   | `http://<your-local-ip>:5000`    |

## Features

- **Record** lung sounds directly from the device microphone
- **Upload** .wav files from device storage
- **View** overall classification (Normal / Crackle / Wheeze / Both)
- **Inspect** per-segment confidence and probability breakdown
- **History** of past analyses within the session
