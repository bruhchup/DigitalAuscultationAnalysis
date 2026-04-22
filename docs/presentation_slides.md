# AusculTek -- Final Presentation Slide Outline

Use a simple academic PowerPoint template. Bullet points only, max 6 lines per slide.
Target: 10 minutes (7 content slides + title + outline + summary + Q&A)

---

## SLIDE 1: Title Slide

**AusculTek: Automated Respiratory Sound Classification Using Deep Learning**

Hayden Banks
Dept of CS, Math and ENGR
College of STEM
Shepherd University
Shepherdstown, WV 25443
Email: hbanks01@shepherd.edu
April 29, 2026

---

## SLIDE 2: Presentation Outline

- Introduction and Motivation
- Dataset and Challenge
- Methods: From Baseline to Deep Learning
- System Architecture
- Results
- Demo
- Summary

(Re-show this slide between major sections with completed items dimmed)

---

## SLIDE 3: Introduction and Motivation (~90s)

- Respiratory diseases: 3M+ deaths/year from COPD alone
- Auscultation (stethoscope listening) is subjective
  - Clinician agreement below 75% on lung sound classification
- Digital stethoscopes enable computational analysis
- Goal: automated, objective second opinion for clinicians
- AusculTek classifies lung sounds as Normal, Crackle, Wheeze, or Both

SPEAKER NOTES: Frame the problem -- why this matters clinically. Mention rural/underserved access gap.

---

## SLIDE 4: The ICBHI 2017 Dataset (~90s)

- 920 recordings from 126 patients, 4 stethoscope models
- 6,898 annotated respiratory cycles
- Class distribution (include a simple bar chart or pie):
  - Normal: 3,642 (53%)
  - Crackle: 1,864 (27%)
  - Wheeze: 886 (13%)
  - Both: 506 (7%)
- 60/40 patient-level train/test split (no data leakage)
- ICBHI Score = (Sensitivity + Specificity) / 2

SPEAKER NOTES: Emphasize the class imbalance -- 7:1 ratio between largest and smallest class. Explain why patient-level splits matter.

---

## (Re-show Outline -- dim Introduction, Dataset -- highlight Methods)

---

## SLIDE 5: Methods -- Baseline to Deep Learning (~90s)

- Phase 1: Random Forest with 130-dim MFCC features
  - 58% accuracy, 48% balanced accuracy, heavy overfitting
- Phase 2: CNN6 from PANNs pretrained on AudioSet (2M clips)
  - Transfer learning: general audio knowledge to lung sounds
- Training: Hybrid Supervised Contrastive Learning + Cross-Entropy
  - L = 0.5 * L_SupCon + 0.5 * L_CE
  - Contrastive loss clusters same-class embeddings together
- Trained 400 epochs on Google Colab T4 GPU ($10 budget)

SPEAKER NOTES: Explain contrastive learning simply -- "the model learns that two crackle sounds should look similar in its internal representation, while a crackle and a normal sound should look different." Mention Moummad & Farrugia (2023) as the reference methodology.

---

## SLIDE 6: System Architecture (~90s)

(Include a simple diagram showing the flow)

- Four layers:
  - Training: SCL package on Google Colab (GPU)
  - Inference: Unified API for RF and CNN6 backends
  - API: Flask REST server (POST /classify)
  - Clients: Streamlit web dashboard + React Native mobile app
- Same model powers both web and mobile interfaces
- Mobile records WAV at 16kHz, sends to API over Wi-Fi

SPEAKER NOTES: Emphasize the factory pattern -- both models share the same classify() interface. Mention Expo SDK 54 for mobile.

---

## (Re-show Outline -- dim through Architecture -- highlight Results)

---

## SLIDE 7: Results (~90s)

(Include a comparison table)

| Model              | ICBHI Score | Sensitivity | Specificity |
|--------------------|-------------|-------------|-------------|
| Random Forest      | --          | low         | --          |
| CNN6 (50 epochs)   | 0.5497      | 0.333       | 0.766       |
| CNN6 (400 epochs)  | **0.5923**  | **0.415**   | **0.770**   |

- 24% improvement in sensitivity (abnormal sound detection)
- Specificity held steady at 77%
- Exceeds published results from reference paper (0.55-0.57)
- RF: 32% overfit gap; CNN6: minimal overfitting

SPEAKER NOTES: Explain what sensitivity vs specificity means here -- sensitivity = catching sick patients, specificity = correctly identifying healthy ones. The jump from 0.333 to 0.415 means the model got significantly better at detecting crackles/wheezes.

---

## SLIDE 8: Demo (~2-3 min)

**Live demo -- switch out of slides**

1. Streamlit dashboard: upload an ICBHI WAV file, show the full visualization suite (gauge, donut chart, waveform, heatmap)
2. Mobile app: record or upload on phone, show results coming back from the API

Have a backup plan: pre-recorded screen capture video in case Wi-Fi or setup fails.

---

## (Re-show Outline -- dim through Demo -- highlight Summary)

---

## SLIDE 9: Summary (~60s)

- Built end-to-end respiratory sound classification system
- CNN6 + Supervised Contrastive Learning achieves 0.5923 ICBHI Score
- Multi-platform: Streamlit web + Flask API + React Native mobile
- Exceeds published benchmark results on $10 compute budget
- Future: cloud deployment, larger backbones, clinical validation

SPEAKER NOTES: Tie back to the motivation -- this system could provide an objective second opinion for clinicians, especially in underserved areas.

---

## SLIDE 10: Questions?

(Simple slide with "Questions?" or similar)

---

## PREPARATION CHECKLIST

Before the presentation:
- [ ] Have Streamlit running with CNN6 model loaded
- [ ] Have Flask API running
- [ ] Have mobile app connected and tested
- [ ] Have a backup demo video recorded in case of technical issues
- [ ] Have an ICBHI WAV file ready to upload
- [ ] Practice to fit within 10 minutes
- [ ] Write speaker notes on note cards for each slide
