# AusculTek: Automated Respiratory Sound Classification Using Deep Learning

**Author:** Hayden Banks
**Course:** CIS 487 -- Capstone Project, Spring 2026
**Date:** April 3, 2026

---

## 1. Introduction and Background

Respiratory diseases remain among the leading causes of morbidity and mortality worldwide. According to the World Health Organization, chronic obstructive pulmonary disease (COPD) alone accounts for over 3 million deaths annually, while asthma affects more than 260 million people globally. Early and accurate detection of respiratory abnormalities is critical for timely intervention, yet access to trained pulmonologists remains limited, particularly in rural and underserved communities. The primary diagnostic tool for respiratory assessment---the stethoscope---has remained fundamentally unchanged for over two centuries, relying entirely on the subjective listening ability of the clinician.

Auscultation, the practice of listening to internal body sounds, is the most widely used method for screening respiratory conditions. During a standard examination, a clinician places a stethoscope on the patient's chest and back, listening for adventitious (abnormal) lung sounds such as crackles and wheezes. Crackles are discontinuous, explosive sounds typically associated with conditions like pneumonia, pulmonary fibrosis, and heart failure. Wheezes are continuous, high-pitched sounds commonly linked to asthma, COPD, and bronchitis. The presence, location, and characteristics of these sounds guide clinical decision-making regarding further diagnostic testing and treatment.

However, auscultation is inherently subjective. Studies have shown significant inter-listener variability among clinicians, with agreement rates on adventitious sound classification sometimes falling below 75%. This variability introduces diagnostic uncertainty, particularly for less experienced practitioners such as medical students, nurses, and primary care providers who may encounter respiratory patients infrequently. Furthermore, traditional auscultation produces no permanent record---the sounds are heard in the moment and cannot be revisited, shared for second opinions, or compared over time to track disease progression.

The emergence of digital stethoscopes has opened the door to computational analysis of respiratory sounds. These devices convert acoustic signals into digital waveforms that can be recorded, stored, and processed by machine learning algorithms. This intersection of medical instrumentation and artificial intelligence presents an opportunity to develop automated classification systems that can assist clinicians by providing objective, repeatable assessments of lung sounds.

The AusculTek project addresses this opportunity by developing an end-to-end system for automated respiratory sound classification. The system accepts digital stethoscope recordings as input, processes the audio signal through a deep learning pipeline, and outputs per-segment classifications indicating whether each respiratory cycle contains normal breath sounds, crackles, wheezes, or both. The goal is not to replace clinical judgment but to augment it---providing a consistent second opinion that can flag potential abnormalities for further review.

The project leverages the ICBHI (International Conference on Biomedical and Health Informatics) 2017 Respiratory Sound Database, the largest publicly available dataset of annotated respiratory sounds. This dataset contains 920 recordings from 126 patients, with expert annotations marking the boundaries and classifications of individual respiratory cycles. Using this standardized benchmark allows the system's performance to be compared against published research and establishes a foundation for future improvements.

From a technical perspective, the project explores multiple approaches to audio classification, progressing from traditional machine learning baselines (Random Forest classifiers with hand-crafted MFCC features) to modern deep learning architectures (convolutional neural networks with supervised contrastive learning). This progression reflects the broader trend in the field and provides insight into how different approaches trade off between complexity, interpretability, and accuracy.

---

## 2. Keywords

Respiratory Sound Classification, Convolutional Neural Network, Supervised Contrastive Learning, Digital Auscultation, ICBHI Dataset, Lung Sound Analysis, Deep Learning

---

## 3. Literature Review

### 3.1 Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning (Gunel et al., 2021)

Gunel et al. proposed adapting supervised contrastive learning (SCL) to the fine-tuning stage of pre-trained language models for natural language understanding tasks. While their work targets NLP rather than audio, the core methodology is directly applicable to any classification problem with pre-trained feature extractors. The authors demonstrated that combining a supervised contrastive loss with the standard cross-entropy loss during fine-tuning yields more robust representations than cross-entropy alone. Their approach works by pulling together representations of samples sharing the same label in the embedding space while pushing apart representations of different-class samples. Across multiple GLUE benchmark tasks, the hybrid SCL+CE objective improved generalization and showed increased robustness to noise in training labels. This paper motivated the hybrid training method used in the AusculTek project, where supervised contrastive learning is combined with cross-entropy to train the CNN6 backbone on respiratory sound spectrograms. The insight that contrastive learning produces more discriminative embeddings is particularly valuable for the respiratory sound domain, where class boundaries can be subtle and training data is limited.

### 3.2 PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition (Kong et al., 2020)

Kong et al. introduced PANNs (Pre-trained Audio Neural Networks), a family of convolutional neural networks trained on the large-scale AudioSet dataset for general-purpose audio tagging. The architecture family includes CNN6, CNN10, and CNN14, with increasing depth and capacity. CNN6, the model adopted in this project, uses four convolutional blocks with 5x5 kernels, batch normalization, and average pooling, producing a 512-dimensional embedding. The models were pre-trained on AudioSet's 2 million audio clips spanning 527 sound event classes, learning general audio representations that transfer effectively to downstream tasks. The authors demonstrated that fine-tuning these pre-trained models consistently outperforms training from scratch on smaller domain-specific datasets. For the AusculTek project, this is critical because the ICBHI dataset contains only ~6,000 annotated respiratory cycles---far too few to train a deep CNN from scratch. By initializing with PANNs weights pre-trained on AudioSet, the model brings knowledge of general acoustic patterns (spectral shapes, temporal dynamics, harmonic structures) that transfer meaningfully to the respiratory sound domain, significantly reducing the amount of domain-specific data needed for effective fine-tuning.

### 3.3 Lung Sound Classification Using Supervised Contrastive Learning (Moummad & Farrugia, 2023)

Moummad and Farrugia directly addressed the problem of respiratory sound classification using the ICBHI 2017 dataset, combining PANNs-based feature extractors with supervised contrastive learning. Their work demonstrated that SCL-trained representations yield superior lung sound classification compared to standard cross-entropy training, achieving state-of-the-art results on the ICBHI benchmark. The authors experimented with multiple training strategies---pure supervised contrastive learning with linear evaluation, a multi-head variant incorporating patient metadata (recording location, device type), and a hybrid method jointly optimizing contrastive and cross-entropy losses. Their hybrid approach achieved an ICBHI score above 0.55, with particular improvements in sensitivity to abnormal classes (crackles, wheezes, both). A key contribution was showing that the contrastive objective helps mitigate the severe class imbalance in the ICBHI dataset, where normal sounds outnumber abnormal classes by roughly 4:1. This paper served as the primary reference for the AusculTek project's SCL training pipeline. The project's code directly implements the training strategies, loss functions, and data preprocessing described in this work, adapting them into a modular codebase with a web-based inference interface.

### 3.4 Related Project: ICBHI Challenge Baselines and Community Benchmarks

The ICBHI 2017 challenge established standardized evaluation protocols for respiratory sound classification that have been widely adopted by the research community. The challenge uses a specific train/test split and evaluates performance using the ICBHI score, defined as the average of sensitivity (SE) and specificity (SP): Score = (SE + SP) / 2. This metric addresses class imbalance by equally weighting the model's ability to detect abnormal sounds (sensitivity) and correctly identify normal sounds (specificity). Early challenge entries relied on hand-crafted features such as MFCCs, spectral centroid, and zero-crossing rate combined with traditional classifiers like SVMs and Random Forests, achieving ICBHI scores in the 0.40--0.50 range. More recent deep learning approaches have pushed scores above 0.55, with the best published results exceeding 0.60 using ensemble methods and extensive data augmentation. The AusculTek project implements both paradigms---a Random Forest baseline with MFCC features and a CNN6-based deep learning model---enabling direct comparison of traditional and modern approaches on the same dataset and evaluation protocol. The project's current ICBHI score of 0.5497 with the hybrid SCL method places it competitively within the range of published deep learning results, with room for improvement through extended training and hyperparameter optimization.

---

## 4. Overview of Your Project

### What AusculTek Is

AusculTek is an end-to-end respiratory sound classification system that takes a digital stethoscope recording (.wav file) as input and produces a detailed diagnostic report identifying the presence and location of adventitious lung sounds within the recording. The system is built as a web application powered by a deep learning backend, designed to be accessible to clinicians without any machine learning expertise.

The system operates through a four-stage pipeline:

1. **Audio Preprocessing:** The raw recording is resampled to a standard 16 kHz sample rate and, depending on the model backend, bandpass filtered to isolate the clinically relevant frequency range (50--2000 Hz) where respiratory sounds are concentrated.

2. **Segmentation:** The recording is divided into individual respiratory cycles. When ICBHI-format annotation files are available, the system uses expert-defined cycle boundaries. Otherwise, it applies a sliding window approach (8-second windows with 4-second overlap for the CNN model, or 3-second windows with 1.5-second overlap for the RF model) to automatically segment the recording.

3. **Classification:** Each segment is independently classified into one of four categories: Normal, Crackle, Wheeze, or Both (crackle and wheeze co-occurring). The CNN6 backend converts each segment into a mel spectrogram and passes it through the trained encoder and linear classifier. The Random Forest backend extracts a 130-dimensional feature vector (MFCCs, deltas, spectral descriptors) and classifies using the trained sklearn model.

4. **Aggregation and Visualization:** Per-segment predictions are aggregated into an overall recording-level assessment using confidence-weighted majority voting. Results are displayed through an interactive Streamlit dashboard featuring a confidence gauge, classification distribution pie chart, annotated waveform visualization, probability heatmap across segments, and a detailed segment-by-segment data table.

### Why AusculTek Makes a Difference

AusculTek addresses several concrete gaps in current clinical practice:

**Objectivity and Consistency.** Unlike human auscultation, the system produces the same output every time for the same input. This eliminates inter-listener variability and provides a reproducible baseline assessment that clinicians can use alongside their own judgment.

**Accessibility.** The web-based interface requires no specialized hardware or software beyond a browser. A clinician in a rural clinic can upload a recording and receive analysis within seconds, without needing access to a pulmonologist.

**Documentation.** Every analysis produces a permanent, shareable record including the raw classification data, confidence scores, and visualizations. This enables longitudinal tracking of patient respiratory health and facilitates teleconsultation by allowing recordings and analyses to be shared remotely.

**Education.** The per-segment probability display and annotated waveform provide transparency into the classification process, helping medical students and trainees develop their auscultation skills by comparing the system's assessment against their own.

**Dual-Model Architecture.** The system supports both a lightweight Random Forest model and a deep learning CNN6 model through the same interface, allowing deployment flexibility. The RF model runs efficiently on any hardware, while the CNN model provides higher accuracy when computational resources are available.

---

## 5. Additional Sections

### 5.1 System Architecture

This section will detail the modular architecture of the AusculTek system, including the separation between the inference pipeline, the web application layer, and the training infrastructure. It will include a system diagram showing data flow from audio input through preprocessing, model inference, and visualization output. The factory pattern used by `load_classifier()` to support multiple model backends will be described.

### 5.2 Dataset Description and Preprocessing

This section will provide a detailed description of the ICBHI 2017 Respiratory Sound Database, including recording conditions (number of patients, recording devices, auscultation locations), class distribution statistics (Normal: 2063, Crackle: 1215, Wheeze: 501, Both: 363 respiratory cycles), and the standardized 60/40 train/test split. The preprocessing pipeline will be documented, including resampling, bandpass filtering, circular padding, fade-in/fade-out, and mel spectrogram computation parameters.

### 5.3 Model Comparison and Evaluation

This section will present a comparative analysis of the different model approaches implemented in the project: the Random Forest baseline with MFCC features versus the CNN6 model trained with four different strategies (standard cross-entropy, supervised contrastive learning, multi-head SCL, and hybrid SCL+CE). Evaluation metrics will include the ICBHI score, sensitivity, specificity, and per-class precision/recall. Confusion matrices and training curves will be included once full training runs are completed.

### 5.4 Data Augmentation Strategy

This section will describe the SpecAugment-based data augmentation applied during CNN training, including frequency masking and time masking parameters. The role of augmentation in contrastive learning (generating the two views for the contrastive loss) and its impact on model robustness will be discussed. Class weighting strategies for handling the imbalanced ICBHI dataset will also be covered.

---

## 6. Implementation

### 6.1 Technology Stack

The AusculTek system is built on the following technologies and platforms:

**Programming Language:** Python 3.12

**Deep Learning Framework:** PyTorch 2.0+ with torchaudio for audio-specific transforms. PyTorch was selected for its dynamic computation graph, strong community support for audio/signal processing, and compatibility with the PANNs pretrained weights.

**Audio Processing:** librosa 0.11.0 for audio loading, resampling, and feature extraction; scipy for Butterworth bandpass filtering; torchaudio for mel spectrogram computation and augmentation transforms during training.

**Traditional ML:** scikit-learn 1.6.1 for the Random Forest baseline classifier and standard scaling.

**Web Application:** Streamlit 1.30+ for the interactive dashboard interface; Plotly 5.18+ for all visualizations (gauge charts, pie charts, heatmaps, waveform plots, bar charts).

**Training Infrastructure:** Google Colab with T4 GPU for CNN training, since the project has no access to a local GPU. A dedicated Colab notebook automates the full training pipeline including dataset setup, weight download, training, and model export.

**Data Visualization:** matplotlib and seaborn for training analysis and comparison plots generated during model development.

**Version Control:** Git with GitHub (github.com/bruhchup/DigitalAuscultationAnalysis) for source code management.

### 6.2 Model Architecture Details

The primary model uses the **CNN6** architecture from the PANNs family:

- **4 convolutional blocks** with 5x5 kernels (64, 128, 256, 512 channels)
- Batch normalization and average pooling at each stage
- Global pooling (max + mean over frequency and time dimensions) producing a **512-dimensional embedding**
- **Projector head** (512 -> 512 -> 128) for the contrastive learning objective
- **Linear classifier** (512 -> 4) for the final four-class prediction
- Initialized with PANNs weights pre-trained on AudioSet (Cnn6_mAP=0.343.pth)

The training uses the **hybrid SCL+CE loss**:

```
L = alpha * L_SupCon + (1 - alpha) * L_CE
```

Where `alpha = 0.5`, the supervised contrastive loss operates on projected embeddings with temperature `tau = 0.06`, and the cross-entropy loss uses inverse-frequency class weights to address dataset imbalance.

Training hyperparameters:
- Optimizer: Adam with learning rate 1e-4 and cosine annealing schedule
- Batch size: 128 (on Colab T4 GPU)
- Audio duration: 8 seconds at 16 kHz (128,000 samples)
- Mel spectrogram: 64 mel bands, 1024-point FFT, 512 hop length, 50--2000 Hz
- Data augmentation: SpecAugment (frequency and time masking)

### 6.3 Inference Pipeline

The inference pipeline is implemented in `audio_preprocessing/inference.py` with a factory pattern supporting two backends:

- **`AudioClassifier`** (sklearn): Loads a pickled Random Forest model and standard scaler. Extracts 130-dimensional MFCC feature vectors per segment (20 MFCCs + deltas + delta-deltas + spectral descriptors). Uses 3-second sliding windows.

- **`CNNClassifier`** (PyTorch): Loads the trained CNN6 encoder and linear classifier from saved state dictionaries. Computes mel spectrograms on-the-fly per segment. Uses 8-second sliding windows. Runs on GPU if available, falls back to CPU.

Both backends share the same public API (`classify(audio_path, annotation_path=None)`) and return identically structured result dictionaries, making them interchangeable from the web application's perspective.

### 6.4 Web Application

The Streamlit dashboard (`app.py`) provides:

- **Sidebar:** File upload (.wav), optional ICBHI annotation upload (.txt), session history
- **Summary metrics:** Overall classification, confidence, segment count, duration
- **Confidence gauge:** Radial gauge showing overall prediction confidence
- **Classification distribution:** Donut chart of per-segment label counts
- **Annotated waveform:** Time-domain plot with color-coded overlay regions per classified segment
- **Probability heatmap:** Class probabilities across all segments with cell annotations
- **Confidence timeline:** Bar chart of per-segment confidence colored by classification
- **Segment detail table:** Full tabular breakdown with start/end times, labels, and per-class probabilities
- **Raw JSON output:** Collapsible section with the complete classification result

### 6.5 Current Project Status

**Completed:**
- Full ICBHI dataset integration with metadata generation and train/test splitting
- Random Forest baseline model (trained and deployed)
- CNN6 model architecture with PANNs transfer learning
- Supervised contrastive learning training pipeline (four methods: SL, SCL, MSCL, Hybrid)
- Google Colab training notebook for GPU-accelerated training
- Initial 50-epoch hybrid SCL training run (ICBHI Score: 0.5497, SE: 0.333, SP: 0.766)
- Dual-backend inference pipeline with config-driven model selection
- Interactive Streamlit web dashboard with full visualization suite

**In Progress:**
- Extended 400-epoch training run on Google Colab for improved accuracy
- Sensitivity improvement (current SE of 0.333 indicates the model over-predicts Normal class)

---

## 7. Future Work

The following tasks represent the next steps for the AusculTek project:

**Extended Training.** The current model was trained for only 50 epochs due to Colab session time limits. A full 400-epoch training run with batch size 128 is expected to significantly improve performance, particularly sensitivity to abnormal classes. The Colab notebook is configured and ready for this run.

**Hyperparameter Tuning.** Systematic exploration of the contrastive loss temperature (tau), the loss weighting parameter (alpha), learning rate schedules, and class weight strategies. The current sensitivity of 0.333 suggests the class weights may need adjustment to more aggressively penalize misclassification of crackles and wheezes.

**Backbone Comparison.** The codebase supports CNN6, CNN10, and CNN14 backbones. Training and evaluating CNN10 and CNN14 will reveal whether the additional model capacity improves classification on this dataset or leads to overfitting given the limited training data.

**Real-Time Inference.** Adapting the system for streaming audio input from a connected digital stethoscope, enabling real-time classification during a clinical examination rather than requiring post-hoc file upload.

**Clinical Validation.** Partnering with healthcare professionals to evaluate the system's utility in a clinical setting, gathering feedback on the dashboard design, classification accuracy on real-world recordings (beyond the ICBHI benchmark), and integration into clinical workflows.

**Model Explainability.** Adding Grad-CAM or attention visualizations to show which spectrogram regions drive the model's predictions, increasing clinician trust and enabling model debugging.

---

## 8. References

1. Gunel, B., Du, J., Conneau, A., & Stoyanov, V. (2021). Supervised contrastive learning for pre-trained language model fine-tuning. *International Conference on Learning Representations (ICLR)*.

2. Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 28, 2880--2894.

3. Moummad, I., & Farrugia, N. (2023). Pretraining Respiratory Sound Representations using Metadata and Contrastive Learning. *IEEE International Workshop on Machine Learning for Signal Processing (MLSP)*.

4. Rocha, B. M., et al. (2019). An open access database for the evaluation of respiratory sound classification algorithms. *Physiological Measurement*, 40(3), 035001.

5. Khosla, P., et al. (2020). Supervised contrastive learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 18661--18673.

6. Gemmeke, J. F., et al. (2017). Audio Set: An ontology and human-labeled dataset for audio events. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 776--780.

7. McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18--25.

8. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

9. Park, D. S., et al. (2019). SpecAugment: A simple data augmentation method for automatic speech recognition. *Interspeech*, 2613--2617.

10. World Health Organization. (2023). Chronic respiratory diseases. Retrieved from https://www.who.int/health-topics/chronic-respiratory-diseases.
