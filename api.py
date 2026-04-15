"""
=============================================================================
AusculTek - Flask REST API
=============================================================================

Exposes the inference pipeline as a REST endpoint for the React Native app.

Usage:
  python api.py                     # runs on http://0.0.0.0:5000
  python api.py --port 8080         # custom port

Endpoints:
  POST /classify   - Upload a .wav file, returns classification JSON
  GET  /health     - Health check / model status

Author: Hayden Banks
Date: April 2026
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# Make audio_preprocessing importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "audio_preprocessing"))
from inference import load_classifier

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.environ.get("MODEL_DIR", "models/scl_model")
classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        classifier = load_classifier(MODEL_DIR)
    return classifier


@app.route("/health", methods=["GET"])
def health():
    try:
        clf = get_classifier()
        return jsonify({
            "status": "ok",
            "model": clf.cfg.get("model_name", "unknown"),
            "classes": clf.class_names,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503


@app.route("/classify", methods=["POST"])
def classify():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Send a .wav file as 'audio'."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only .wav files are supported."}), 400

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    # Optional annotation file
    annotation_path = None
    if "annotations" in request.files:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_ann:
            request.files["annotations"].save(tmp_ann.name)
            annotation_path = tmp_ann.name

    try:
        clf = get_classifier()
        results = clf.classify(tmp_path, annotation_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)
        if annotation_path:
            os.unlink(annotation_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AusculTek API server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    MODEL_DIR = args.model_dir

    # Pre-load classifier
    print(f"Loading model from {MODEL_DIR}...")
    get_classifier()
    print(f"API ready on http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
