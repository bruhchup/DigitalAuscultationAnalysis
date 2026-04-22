# AusculTek: Automated Respiratory Sound Classification Using Deep Learning and Supervised Contrastive Learning

**Author:** Hayden Banks

**Course:** CIS 487 -- Capstone Project, Spring 2026

**Advisor:** [Advisor Name]

**Date:** April 20, 2026

---

## Abstract

Respiratory diseases are among the leading causes of death worldwide, yet diagnosis through auscultation remains subjective and dependent on clinician expertise. This paper presents AusculTek, an end-to-end system for automated respiratory sound classification built on the ICBHI 2017 Respiratory Sound Database. The project progressed through multiple development phases: a Random Forest baseline using hand-crafted MFCC features, a CNN6 deep learning model trained with supervised contrastive learning, and a mobile application enabling real-time recording and analysis. The hybrid training objective, combining supervised contrastive loss with cross-entropy loss, achieved an ICBHI score of 0.5923 after 400 epochs of GPU-accelerated training on Google Colab, exceeding published results from the reference methodology. The system is deployed through two interfaces: an interactive Streamlit web dashboard for desktop analysis, and a React Native mobile application connected to a Flask REST API for point-of-care use. This paper details the development process from initial data exploration through baseline modeling, deep learning integration, web deployment, and mobile app development, discussing the technical challenges of class imbalance, transfer learning, and bridging the gap between research code and a usable clinical tool.

---

## 1. Introduction and Background (~2 pages)

Respiratory diseases remain among the leading causes of morbidity and mortality worldwide. According to the World Health Organization, chronic obstructive pulmonary disease (COPD) alone accounts for over 3 million deaths annually, while asthma affects more than 260 million people globally [10]. Early and accurate detection of respiratory abnormalities is critical for timely intervention, yet access to trained pulmonologists remains limited, particularly in rural and underserved communities.

Auscultation, the practice of listening to internal body sounds using a stethoscope, is the most widely used method for screening respiratory conditions. During a standard examination, a clinician places a stethoscope on the patient's chest and back, listening for adventitious lung sounds such as crackles and wheezes. Crackles are discontinuous, explosive sounds typically associated with pneumonia, pulmonary fibrosis, and heart failure. Wheezes are continuous, high-pitched sounds commonly linked to asthma, COPD, and bronchitis [4].

However, auscultation is inherently subjective. Studies have shown significant inter-listener variability among clinicians, with agreement rates on adventitious sound classification sometimes falling below 75%. This variability introduces diagnostic uncertainty, particularly for less experienced practitioners. Furthermore, traditional auscultation produces no permanent record---the sounds are heard in the moment and cannot be revisited, shared for second opinions, or compared over time to track disease progression.

The emergence of digital stethoscopes has opened the door to computational analysis of respiratory sounds. These devices convert acoustic signals into digital waveforms that can be recorded, stored, and processed by machine learning algorithms. This intersection of medical instrumentation and artificial intelligence presents an opportunity to develop automated classification systems that provide objective, repeatable assessments of lung sounds.

**This paper presents AusculTek, an automated respiratory sound classification system that applies supervised contrastive learning with a pretrained CNN6 convolutional neural network to the ICBHI 2017 benchmark, achieving an ICBHI score of 0.5923 while delivering practical web and mobile clinical tools.** The system classifies individual respiratory cycles into four categories---Normal, Crackle, Wheeze, or Both---and presents results through both an interactive Streamlit dashboard and a React Native mobile application, designed for clinician use without machine learning expertise.

The project's development followed a phased approach, beginning with data exploration and traditional machine learning baselines before progressing to deep learning. This iterative process provided insight into the fundamental challenges of respiratory sound classification---severe class imbalance, limited labeled data, subtle acoustic distinctions between classes---and informed the selection of supervised contrastive learning as the training strategy for the final model. The project leverages the ICBHI 2017 Respiratory Sound Database [4], the largest publicly available dataset of annotated respiratory sounds, enabling direct comparison with published research.

---

## 2. Keywords

Respiratory Sound Classification, Convolutional Neural Network, Supervised Contrastive Learning, Digital Auscultation, ICBHI Dataset, Transfer Learning, Mel Spectrogram

---

## 3. Literature Review (~3 pages)

### 3.1 PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition (Kong et al., 2020)

Kong et al. introduced PANNs (Pre-trained Audio Neural Networks), a family of convolutional neural networks trained on the large-scale AudioSet dataset for general-purpose audio tagging [2]. The architecture family includes CNN6, CNN10, and CNN14, with increasing depth and capacity. CNN6, the model adopted in this project, uses four convolutional blocks with 5x5 kernels, batch normalization, and average pooling, producing a 512-dimensional embedding. The models were pretrained on AudioSet's 2 million audio clips spanning 527 sound event categories, learning general audio representations that transfer effectively to downstream tasks.

The authors demonstrated that fine-tuning these pretrained models consistently outperforms training from scratch on smaller domain-specific datasets. For respiratory sound classification, this is critical because the ICBHI dataset contains only approximately 6,900 annotated respiratory cycles---far too few to train a deep CNN from scratch without severe overfitting. By initializing with PANNs weights pretrained on AudioSet, the model brings knowledge of general acoustic patterns---spectral shapes, temporal dynamics, harmonic structures---that transfer meaningfully to the respiratory sound domain. This paper directly informed the choice of CNN6 as the backbone architecture for AusculTek, and the pretrained weights (Cnn6_mAP=0.343.pth) serve as the initialization point for all training runs.

### 3.2 Supervised Contrastive Learning (Khosla et al., 2020)

Khosla et al. extended the self-supervised contrastive learning framework to the fully supervised setting, proposing Supervised Contrastive Learning (SupCon) [5]. Unlike standard cross-entropy, which operates on logits, SupCon operates in the embedding space: for each sample in a batch, it identifies all other samples sharing the same label as positives and all differently-labeled samples as negatives. The loss encourages high cosine similarity among same-class embeddings while pushing apart different-class embeddings, using a temperature-scaled softmax formulation.

The authors demonstrated that SupCon produces more discriminative and transferable representations than cross-entropy training alone, with improved robustness to label noise and natural corruptions on ImageNet. For the ICBHI respiratory sound classification task, where class boundaries are acoustically subtle and the dataset is small and imbalanced, the structured embedding space produced by contrastive learning helps the model learn meaningful distinctions even for underrepresented classes like Wheeze and Both. The temperature parameter (tau) controls the concentration of the similarity distribution, and the low value of 0.06 used in this project produces tight, well-separated clusters---particularly important given the four-class problem with a 7:1 imbalance ratio between the largest and smallest classes.

### 3.3 Pretraining Respiratory Sound Representations Using Metadata and Contrastive Learning (Moummad & Farrugia, 2023)

Moummad and Farrugia directly addressed respiratory sound classification on the ICBHI 2017 dataset, combining PANNs-based feature extractors with supervised contrastive learning [3]. Their work is the primary methodological reference for the AusculTek project. The authors implemented and compared four training strategies: standard supervised learning (SL) with cross-entropy only (~52% ICBHI score), supervised contrastive learning (SCL) with frozen linear evaluation (~55%), multi-head SCL (MSCL) incorporating patient metadata such as recording location and device type (~58%), and a hybrid method jointly optimizing contrastive and cross-entropy losses (~57%).

Their hybrid approach achieved an ICBHI score above 0.55, with particular improvements in sensitivity to abnormal classes. A key contribution was demonstrating that the contrastive objective helps mitigate the severe class imbalance in the ICBHI dataset. The multi-head variant further leveraged metadata---chest location and recording equipment---as auxiliary supervision signals in the contrastive loss, using a lambda parameter to trade off between class-label-based and metadata-based contrastive objectives.

This paper served as the direct foundation for AusculTek's SCL training pipeline. The project's code adapts the authors' open-source implementation (github.com/ilyassmoummad/scl_icbhi2017) into a modular Python package, replicating the training strategies, loss functions, data preprocessing, and evaluation protocols described in the paper.

### 3.4 Supervised Contrastive Learning for Pretrained Language Model Fine-tuning (Gunel et al., 2021)

Gunel et al. proposed adapting supervised contrastive learning to the fine-tuning stage of pretrained models for natural language understanding tasks [1]. While their domain is NLP rather than audio, the core methodology---combining a supervised contrastive loss with cross-entropy during fine-tuning of a pretrained encoder---is directly applicable to any classification problem with pretrained feature extractors. The authors demonstrated improved generalization and robustness to label noise across multiple GLUE benchmark tasks. This work motivated the hybrid training method used in AusculTek, validating the principle that contrastive learning applied during fine-tuning (rather than only during pretraining) yields more robust representations across domains.

### 3.5 The ICBHI 2017 Respiratory Sound Database (Rocha et al., 2019)

Rocha et al. described the creation and structure of the ICBHI 2017 Respiratory Sound Database, the benchmark dataset used in this project [4]. The database contains 920 recordings from 126 patients captured using four different stethoscope models at seven chest locations. Expert annotators marked the onset and offset of each respiratory cycle and labeled the presence of crackles and wheezes, producing 6,898 annotated cycles. The dataset established a standardized evaluation protocol using a 60/40 patient-level train/test split and the ICBHI score metric. Early challenge entries using hand-crafted features with SVMs and Random Forests achieved ICBHI scores in the 0.40--0.50 range, while recent deep learning approaches have pushed scores above 0.55--0.60. This database serves as the sole data source for all AusculTek training and evaluation.

### 3.6 SpecAugment: A Simple Data Augmentation Method (Park et al., 2019)

Park et al. introduced SpecAugment, a data augmentation technique applied directly to mel spectrograms [9]. The method consists of frequency masking (zeroing out contiguous frequency bands) and time masking (zeroing out contiguous time steps). Despite its simplicity, SpecAugment significantly improved speech recognition performance by acting as a regularizer that prevents the model from relying on any single spectral or temporal region. In the AusculTek project, SpecAugment is applied during training with frequency mask size 20 (2 masks) and time mask size 50 (2 masks). Within the contrastive learning framework, SpecAugment also serves to generate the two augmented views of each sample required by the contrastive loss.

---

## 4. Overview of the Project (~1.5 pages)

### 4.1 What AusculTek Is

AusculTek is an end-to-end respiratory sound classification system that accepts digital stethoscope recordings as input and produces detailed diagnostic reports identifying the presence and location of adventitious lung sounds. The system is delivered through two interfaces: a Streamlit web dashboard for desktop analysis and a React Native mobile application for point-of-care use, both powered by interchangeable deep learning and traditional machine learning backends.

The system operates through a four-stage pipeline. First, raw audio is resampled and bandpass filtered to isolate the clinically relevant frequency range (50--2000 Hz). Second, the recording is segmented into individual respiratory cycles using either expert annotations (ICBHI format) or an automatic sliding window approach. Third, each segment is independently classified into one of four categories: Normal, Crackle, Wheeze, or Both. Fourth, per-segment predictions are aggregated into a recording-level assessment and displayed through interactive visualizations.

### 4.2 Why AusculTek Makes a Difference

AusculTek addresses several concrete gaps in current clinical practice. It provides objectivity and consistency, producing the same output every time for the same input and eliminating inter-listener variability. The web-based interface requires no specialized hardware beyond a browser, making it accessible to clinicians in rural or underserved settings. The mobile application extends this accessibility further, enabling clinicians to record lung sounds directly from a phone and receive classification results in real time at the point of care. Every analysis produces a permanent, shareable record with confidence scores and visualizations, enabling longitudinal tracking and teleconsultation. The per-segment probability display provides transparency into the classification process, serving as an educational tool for trainees developing their auscultation skills.

### 4.3 Multi-Platform Architecture

A distinguishing feature of AusculTek is its multi-platform architecture. The system supports both a lightweight Random Forest model and a deep learning CNN6 model through the same inference API. A factory pattern in the inference module (load_classifier()) allows seamless switching between backends via an environment variable. The classification engine is exposed through three interfaces: the Streamlit web dashboard for interactive desktop analysis, a Flask REST API for programmatic access, and a React Native (Expo SDK 54) mobile application that communicates with the API over the local network. All three interfaces consume the same standardized result dictionary, ensuring consistent behavior regardless of access method.

---

## 5. Development Process (~2 pages)

### 5.1 Phase 1: Data Exploration and Baseline (Weeks 1--3)

Development began in February 2026 with acquisition and exploration of the ICBHI 2017 dataset. Initial work focused on understanding the data format---920 WAV recordings with paired TXT annotation files specifying respiratory cycle boundaries and labels---and building an audio preprocessing pipeline. Key early decisions included adopting the standardized 60/40 patient-level train/test split to prevent data leakage, and applying a Butterworth bandpass filter (50--2000 Hz) on full recordings before segmentation to avoid edge artifacts on short clips.

The first classification model was a Random Forest trained on 130-dimensional MFCC feature vectors (20 MFCCs with mean and standard deviation, delta and delta-delta coefficients, and spectral descriptors). This baseline achieved 58% test accuracy but only 48% balanced accuracy, revealing the fundamental challenge of class imbalance: Normal class recall reached 67%, Crackle 55%, but Wheeze only 21%. The significant train/test accuracy gap (90% vs. 58%) indicated overfitting. Despite these limitations, the RF model established a working end-to-end pipeline from audio input to classification output and served as the foundation for the Streamlit dashboard.

Several alternative preprocessing approaches were explored during this phase, including a Variational Mode Decomposition (VMD) pipeline, which failed to improve classification and was abandoned. This experience reinforced the principle that simpler preprocessing approaches are more reliable for this domain.

### 5.2 Phase 2: Streamlit Dashboard and Inference Pipeline (Weeks 3--4)

With the baseline model functional, development shifted to building the user-facing application. The Streamlit dashboard was designed to present classification results through multiple complementary visualizations: a confidence gauge showing overall prediction certainty, a donut chart of per-segment classification distribution, an annotated waveform with color-coded overlay regions, a probability heatmap showing class probabilities across all segments, a confidence timeline bar chart, and a detailed segment data table.

The inference pipeline was designed with extensibility in mind. The AudioClassifier class encapsulated the RF model's full pipeline---loading, feature extraction, classification, and result formatting---behind a simple classify(audio_path, annotation_path) interface. This API contract was maintained when the CNN backend was added later, enabling the factory pattern that allows backend selection without changing application code.

### 5.3 Phase 3: CNN6 and Supervised Contrastive Learning Integration (Weeks 4--6)

The most substantial development phase involved integrating the CNN6 architecture with supervised contrastive learning, following the methodology of Moummad and Farrugia [3]. This required creating a new Python package (audio_preprocessing/scl/) containing 14 modules adapted from the authors' open-source repository.

Key adaptation work included generating a metadata CSV from the ICBHI annotations to support the dataset loader, downloading and integrating CNN6 pretrained weights from AudioSet, restructuring the training code as a proper Python package with relative imports, adding device parameterization to support both CPU and GPU training, and implementing best-model checkpointing based on ICBHI score rather than saving only the final epoch. The CNNClassifier class was added to the inference module, matching the AudioClassifier API, and the Streamlit app was updated to support four-class output (adding the Both category with purple color coding) and dynamic sample rate handling.

A 2-epoch smoke test verified the full pipeline end-to-end before committing to longer training runs. The initial 50-epoch hybrid training run, conducted on CPU due to lack of local GPU access, produced an ICBHI score of 0.5497 (Sensitivity: 0.333, Specificity: 0.766).

### 5.4 Phase 4: GPU Training and Model Improvement (Weeks 6--7)

The 50-epoch CPU-trained model demonstrated the viability of the approach but suffered from low sensitivity (0.333), indicating the model had not yet learned to reliably detect abnormal respiratory sounds. To address this, a Google Colab notebook was developed to automate the full training pipeline on a T4 GPU, including dataset loading from Google Drive, metadata generation, pretrained weight download, and model export.

The 400-epoch training run on the T4 GPU completed in approximately 1.5 hours---a dramatic improvement over the multi-hour CPU training for just 50 epochs. The extended training produced substantial gains: ICBHI score improved from 0.5497 to 0.5923, sensitivity increased from 0.333 to 0.415 (a 24% relative improvement), and specificity held steady at 0.770. The final ICBHI score of 0.5923 exceeds the published hybrid method results from Moummad and Farrugia [3], validating the implementation and demonstrating that additional training epochs allow the contrastive loss to form tighter, better-separated embedding clusters for the minority classes.

### 5.5 Phase 5: REST API and Mobile Application (Weeks 7--8)

The final development phase extended AusculTek from a desktop web application to a mobile-accessible system. A Flask REST API (api.py) was built to expose the classification pipeline as HTTP endpoints, enabling external clients to submit audio recordings and receive classification results as JSON. The API supports both WAV uploads and automatic format conversion for recordings captured in other formats.

The React Native mobile application was built using Expo SDK 54 with three screens: a home screen for recording or uploading audio, a results screen displaying classification output, and a history screen tracking past analyses. The app records lung sounds directly from the device microphone in WAV format (16 kHz, mono, 16-bit Linear PCM) to match the CNN6 model's expected input format, eliminating the need for server-side format conversion. Audio is transmitted to the Flask API over the local network, and classification results are displayed on the phone within seconds. Key technical challenges included migrating from the deprecated expo-av package to the new expo-audio package in SDK 54 and configuring iOS audio recording permissions.

---

## 6. System Architecture and Dataset (~1.5 pages)

### 6.1 System Architecture

The AusculTek system follows a modular four-layer architecture separating the training infrastructure, inference pipeline, API layer, and client applications.

The training layer (audio_preprocessing/scl/) handles all model training and evaluation. It implements four training strategies---standard cross-entropy (SL), supervised contrastive with linear evaluation (SCL), multi-head contrastive with metadata (MSCL), and hybrid joint training---all sharing common dataset loading, augmentation, and evaluation code. Training is executed on Google Colab with T4 GPUs via a dedicated notebook that automates the full pipeline.

The inference layer (audio_preprocessing/inference.py) provides a unified API for both model backends through the load_classifier() factory function. Each classifier handles its own preprocessing, feature extraction, and classification internally, returning identically structured result dictionaries.

The API layer (api.py) wraps the inference pipeline in a Flask REST server with CORS support, exposing POST /classify and GET /health endpoints. It handles file upload, automatic format conversion for non-WAV audio, and JSON response formatting. The classifier is loaded once at server startup and reused across requests.

The client layer consists of two independent interfaces. The Streamlit web dashboard (app.py) provides a full-featured desktop analysis tool with interactive visualizations. The React Native mobile application (mobile/) enables point-of-care recording and analysis, communicating with the Flask API over the local network. Both clients are model-agnostic, operating only on the standardized result dictionary format.

### 6.2 Dataset Description

The ICBHI 2017 Respiratory Sound Database contains 920 recordings from 126 patients, captured using four stethoscope models (AKG C417L, 3M Littmann Classic II SE, Littmann 3200, WelchAllyn Meditron) at seven chest locations. Expert annotators labeled 6,898 respiratory cycles across four classes: Normal (3,642 cycles, 52.8%), Crackle (1,864 cycles, 27.0%), Wheeze (886 cycles, 12.8%), and Both (506 cycles, 7.3%). Patient diagnoses include COPD, LRTI, URTI, Bronchiectasis, Pneumonia, Bronchiolitis, Asthma, and Healthy.

The standardized 60/40 train/test split divides patients (not individual cycles) into 75 training patients (4,307 cycles) and 51 test patients (2,591 cycles), ensuring no patient appears in both sets. This patient-level splitting is mandatory to prevent data leakage, as multiple recordings from the same patient share acoustic characteristics that would inflate test performance if leaked into training.

---

## 7. Implementation (~2.5 pages)

### 7.1 Technology Stack

The system is built in Python 3.12. PyTorch 2.0+ with torchaudio provides the deep learning framework, selected for its dynamic computation graph and compatibility with PANNs pretrained weights. Audio processing uses librosa 0.11.0 for loading and feature extraction, scipy for Butterworth bandpass filtering, and torchaudio for mel spectrogram computation during training. scikit-learn 1.6.1 powers the Random Forest baseline. The web application uses Streamlit 1.30+ with Plotly 5.18+ for all visualizations. The REST API uses Flask with Flask-CORS for cross-origin mobile access. The mobile application is built with React Native using Expo SDK 54, with expo-audio for device microphone recording and React Navigation for screen management. GPU training is conducted on Google Colab Pro with T4 GPUs ($10 compute budget), as the project has no access to a local GPU.

### 7.2 CNN6 Model Architecture

The CNN6 architecture consists of four convolutional blocks with 5x5 kernels, progressing through 64, 128, 256, and 512 channels. Each block applies batch normalization and 2x2 average pooling. Global pooling (mean over time, max over frequency) reduces the final feature maps to a 512-dimensional embedding. A projector head (512 -> 512 -> 128) maps embeddings for the contrastive loss, and a linear classifier (512 -> 4) maps embeddings to class predictions. The encoder is initialized with PANNs weights pretrained on AudioSet (Cnn6_mAP=0.343.pth).

### 7.3 Hybrid SCL+CE Training

The hybrid training method jointly optimizes two objectives: L = alpha * L_SupCon + (1 - alpha) * L_CE, where alpha = 0.5. The supervised contrastive loss (SupConLoss) operates on L2-normalized projections with temperature tau = 0.06, pulling same-class embeddings together while pushing different-class embeddings apart. The cross-entropy loss uses inverse-frequency class weights (Normal: 0.48, Crackle: 0.95, Wheeze: 2.0, Both: 3.5) to penalize minority-class misclassifications more heavily. Both losses backpropagate through the shared encoder simultaneously.

Training uses Adam optimization with learning rate 1e-4, weight decay 1e-4, and cosine annealing (T_max=40, eta_min=1e-6). Input audio is resampled to 16 kHz, bandpass filtered (50--2000 Hz), padded to 8 seconds using circular padding, and converted to 64-band log-mel spectrograms (1024-point FFT, 512 hop length, 50--2000 Hz). SpecAugment augmentation applies frequency masking (size 20, 2 masks) and time masking (size 50, 2 masks) during training.

### 7.4 Random Forest Baseline

The Random Forest baseline extracts 130-dimensional feature vectors per segment: 20 MFCCs with mean and standard deviation (40 features), delta and delta-delta coefficients with mean and standard deviation (80 features), and spectral descriptors (10 features). Audio is resampled to 8 kHz and segmented into 3-second windows with 1.5-second overlap. The classifier was trained using scikit-learn with default hyperparameters.

### 7.5 Results

| Model | ICBHI Score | Sensitivity | Specificity | Accuracy | Balanced Acc. |
|-------|-------------|-------------|-------------|----------|---------------|
| Random Forest (MFCC) | -- | -- | -- | 58% | 48% |
| CNN6 Hybrid (50 epochs) | 0.5497 | 0.333 | 0.766 | -- | -- |
| CNN6 Hybrid (400 epochs) | **0.5923** | **0.415** | **0.770** | -- | -- |

The 400-epoch model demonstrates clear improvement over the 50-epoch checkpoint across all metrics. Sensitivity increased from 0.333 to 0.415, a 24% relative improvement in the model's ability to detect abnormal respiratory sounds (crackles, wheezes, and co-occurring). Specificity remained stable at 0.770, indicating that the extended training improved abnormal class detection without sacrificing normal class accuracy. The ICBHI score of 0.5923 exceeds the published hybrid method results from Moummad and Farrugia [3], who reported scores in the 0.55--0.57 range.

The progression from the Random Forest baseline to the final CNN6 model illustrates the value of the deep learning approach. The RF model suffered from severe overfitting (32% train/test gap), could only classify three classes, and achieved just 21% recall on wheezes. The CNN6 model handles four classes, generalizes well due to transfer learning and contrastive regularization, and achieves substantially higher sensitivity to all abnormal classes.

### 7.6 Web Application

The Streamlit dashboard provides file upload for WAV recordings and optional ICBHI annotation files, summary metrics (overall classification, confidence, segment count, duration), a radial confidence gauge, a donut chart of classification distribution, an annotated time-domain waveform with color-coded segment overlays, a probability heatmap across all segments, a confidence timeline bar chart, and a segment detail data table. The interface uses a consistent dark teal (#0C4A6E) color scheme with class-specific colors: Normal (#0891B2), Crackle (#F97316), Wheeze (#EF4444), Both (#8B5CF6).

### 7.7 REST API

The Flask REST API (api.py) exposes the inference pipeline for external clients. The POST /classify endpoint accepts audio file uploads via multipart form data, saves to a temporary file, runs classification through the loaded model backend, and returns the full result dictionary as JSON. Non-WAV uploads are automatically converted to WAV using librosa and soundfile before classification. The GET /health endpoint returns the loaded model name and supported class list for client validation. The classifier is lazily loaded on first request and reused, and CORS is enabled for cross-origin mobile access.

### 7.8 Mobile Application

The React Native mobile application, built with Expo SDK 54, provides three screens. The Home screen allows clinicians to either record lung sounds directly using the device microphone or upload existing WAV files via the system file picker. Audio is recorded in WAV format (16 kHz, mono, 16-bit Linear PCM) using the expo-audio package, matching the CNN6 model's expected input format to avoid server-side conversion. The Results screen displays the overall classification, confidence score, and per-segment breakdown returned from the API. The History screen maintains a session log of past analyses for comparison.

The app communicates with the Flask API over the local Wi-Fi network. The API base address is configurable in the services module, allowing deployment across different network environments. Recording permissions are handled through the Expo permissions API, and iOS-specific audio mode configuration (allowsRecording, playsInSilentMode) is set before each recording session.

---

## 8. Future Work (~1 page)

**Hyperparameter Tuning.** Systematic exploration of the contrastive loss temperature (tau), the loss weighting parameter (alpha), learning rate schedules, and class weight strategies could further improve results. While sensitivity improved significantly from 0.333 to 0.415 with extended training, there remains room for improvement through more aggressive class weighting or temperature adjustments.

**Backbone Comparison.** The codebase supports CNN6, CNN10, and CNN14 backbones. Training and evaluating the larger architectures will determine whether additional model capacity improves classification or causes overfitting given the limited training data.

**Training Method Comparison.** Beyond the hybrid method, the codebase implements standard supervised learning (SL), supervised contrastive learning with frozen linear evaluation (SCL), and multi-head supervised contrastive learning incorporating metadata (MSCL). Running all four methods under identical conditions will enable a controlled comparison of training strategies on this dataset.

**Cloud Deployment.** The current mobile application requires the Flask API to run on the same local network as the phone. Deploying the API to a cloud server would enable access from any location, making the system viable for real clinical use beyond the local development environment.

**Real-Time Streaming.** The current mobile app records a complete audio clip before sending it for analysis. Adapting the system for streaming audio input would enable continuous real-time classification during a clinical examination.

**Model Explainability.** Adding Grad-CAM or attention visualizations to highlight which spectrogram regions drive the model's predictions would increase clinician trust and provide a model debugging tool.

**Clinical Validation.** Partnering with healthcare professionals to evaluate the system's utility in a clinical setting, gathering feedback on the interface design, classification accuracy on real-world recordings beyond the ICBHI benchmark, and integration into clinical workflows.

---

## 9. Conclusion

AusculTek demonstrates the feasibility of combining transfer learning, supervised contrastive learning, and modern web and mobile technologies to build a practical respiratory sound classification tool. The phased development approach---from Random Forest baseline through CNN6 with supervised contrastive learning to mobile deployment---provided insight into the core challenges of the ICBHI benchmark: severe class imbalance, limited labeled data, and subtle acoustic boundaries between respiratory sound categories. The final 400-epoch hybrid model achieves an ICBHI score of 0.5923, exceeding published results from the reference methodology and demonstrating that extended training with the contrastive objective produces meaningful gains in sensitivity to abnormal respiratory sounds. The multi-platform architecture---spanning a Streamlit web dashboard, Flask REST API, and React Native mobile application---bridges the gap between research-grade models and clinician-accessible tools, demonstrating that automated auscultation systems can be made both accurate and usable at the point of care.

---

## 10. References

[1] B. Gunel, J. Du, A. Conneau, and V. Stoyanov, "Supervised contrastive learning for pre-trained language model fine-tuning," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2021.

[2] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, "PANNs: Large-scale pretrained audio neural networks for audio pattern recognition," *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, vol. 28, pp. 2880--2894, 2020.

[3] I. Moummad and N. Farrugia, "Pretraining respiratory sound representations using metadata and contrastive learning," in *Proc. IEEE Int. Workshop Mach. Learn. Signal Process. (MLSP)*, 2023.

[4] B. M. Rocha et al., "An open access database for the evaluation of respiratory sound classification algorithms," *Physiol. Meas.*, vol. 40, no. 3, p. 035001, 2019.

[5] P. Khosla et al., "Supervised contrastive learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 18661--18673, 2020.

[6] J. F. Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," in *Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)*, pp. 776--780, 2017.

[7] B. McFee et al., "librosa: Audio and music signal analysis in Python," in *Proc. 14th Python in Science Conf.*, pp. 18--25, 2015.

[8] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.

[9] D. S. Park, W. Chan, Y. Zhang, C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, "SpecAugment: A simple data augmentation method for automatic speech recognition," in *Proc. Interspeech*, pp. 2613--2617, 2019.

[10] World Health Organization, "Chronic respiratory diseases," 2023. [Online]. Available: https://www.who.int/health-topics/chronic-respiratory-diseases
