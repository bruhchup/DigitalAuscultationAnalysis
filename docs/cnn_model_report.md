# AusculTek CNN6 Model Report: Current State and Methods

**Author:** Hayden Banks
**Course:** CIS 487 — Capstone Project, Spring 2026
**Date:** April 17, 2026

---

## 1. Model Overview and Current Performance

The AusculTek project employs a CNN6 convolutional neural network for automated respiratory sound classification on the ICBHI 2017 Respiratory Sound Database. The model classifies individual respiratory cycles extracted from digital stethoscope recordings into four categories: Normal, Crackle, Wheeze, or Both (co-occurring crackle and wheeze). The current deployed model was trained for 50 epochs using a hybrid supervised contrastive learning and cross-entropy (SCL+CE) training objective.

After 50 epochs of hybrid training, the model achieved the following results on the ICBHI standardized test set (60/40 patient-level split):

| Metric         | Value  |
|----------------|--------|
| ICBHI Score    | 0.5497 |
| Sensitivity    | 0.333  |
| Specificity    | 0.766  |

The ICBHI Score, defined as (Sensitivity + Specificity) / 2, is the standard evaluation metric for this benchmark. The current score of 0.5497 falls within the competitive range for published deep learning approaches on this dataset, which typically report scores between 0.50 and 0.60. However, the disparity between specificity (0.766) and sensitivity (0.333) reveals that the model, at 50 epochs, has not yet fully learned to distinguish the minority abnormal classes (Crackle, Wheeze, Both) from the dominant Normal class, which comprises 52.8% of the dataset. This imbalance in detection ability is expected at this early stage of training; the contrastive learning objective requires more epochs to form well-separated class clusters in the embedding space that allow the linear classifier to draw effective decision boundaries for the underrepresented classes.

## 2. Architecture

The CNN6 architecture originates from the PANNs (Pre-trained Audio Neural Networks) family introduced by Kong et al. (2020). The model consists of four convolutional blocks, each using 5x5 kernels with batch normalization and 2x2 average pooling. The channel dimensions progress through 64, 128, 256, and 512 across the four blocks. After the convolutional stack, a global pooling operation (combining mean over time and max over frequency) reduces the feature maps to a fixed 512-dimensional embedding vector regardless of input length.

Two additional heads extend the base encoder for the hybrid training strategy. A projector head (two fully connected layers: 512 to 512 to 128) maps embeddings into a lower-dimensional space where the supervised contrastive loss is computed. A linear classifier (512 to 4) maps the encoder embeddings directly to class logits for the cross-entropy loss and final inference predictions. The encoder is initialized with PANNs weights pre-trained on AudioSet (Cnn6_mAP=0.343.pth), a large-scale dataset of over 2 million audio clips spanning 527 sound event categories. This transfer learning initialization is essential because the ICBHI dataset contains only approximately 6,900 annotated respiratory cycles — far too few to train a deep CNN from scratch without severe overfitting.

## 3. Methods

### 3.1 Primary Reference

The training pipeline is based on the work of Moummad and Farrugia (2023), "Pretraining Respiratory Sound Representations using Metadata and Contrastive Learning," presented at the IEEE International Workshop on Machine Learning for Signal Processing (MLSP). This paper demonstrated that supervised contrastive learning applied to PANNs-based encoders yields superior respiratory sound classification compared to standard cross-entropy training on the ICBHI 2017 benchmark. The authors explored multiple training strategies and found that a hybrid objective — jointly optimizing contrastive and cross-entropy losses — achieved their best results, with ICBHI scores in the 0.55–0.58 range. The AusculTek project directly implements the training strategies, loss formulations, and preprocessing pipeline described in this work, adapting the original codebase (github.com/ilyassmoummad/scl_icbhi2017) into a modular system with a web-based inference interface.

### 3.2 Supervised Contrastive Learning

The core methodological contribution adopted from Moummad and Farrugia (2023), itself grounded in Khosla et al. (2020), is supervised contrastive learning (SCL). Unlike standard cross-entropy training, which optimizes class logits directly, SCL operates on the embedding space by pulling together representations of samples that share the same class label while pushing apart representations of different-class samples. For each sample in a batch, the contrastive loss identifies all other samples with the same label as positives and all samples with different labels as negatives. Pairwise cosine similarities are computed on L2-normalized projections, scaled by a temperature parameter (tau = 0.06), and the loss encourages high similarity among positives relative to negatives.

The hybrid training method, selected for the 50-epoch model, combines the supervised contrastive loss with standard cross-entropy in a single joint objective:

**L = alpha * L_SupCon + (1 - alpha) * L_CE**

where alpha = 0.5, giving equal weight to both objectives. The contrastive loss is computed on the projector head outputs, while the cross-entropy loss is computed on the linear classifier outputs. Both losses are backpropagated through the shared encoder simultaneously, meaning the encoder learns representations that are both contrastively well-structured and directly discriminative for the classification task.

### 3.3 Data Preprocessing and Augmentation

Input audio undergoes several preprocessing steps before entering the model. Raw recordings are resampled to 16 kHz and bandpass filtered (50–2000 Hz Butterworth filter) to isolate the clinically relevant frequency range for respiratory sounds. Individual respiratory cycles are extracted using ICBHI annotation boundaries and padded or truncated to a fixed 8-second duration (128,000 samples) using circular padding, which repeats the audio cyclically rather than filling with silence. Fade-in and fade-out effects (1,000 samples) are applied at segment edges to avoid spectral artifacts from abrupt transitions.

Each audio segment is converted to a 64-band log-mel spectrogram with a 1024-point FFT, 512-sample hop length, and frequency bounds of 50–2000 Hz. The spectrograms are normalized to the [0, 1] range and then standardized using precomputed dataset statistics (mean = 0.3690, std = 0.0255).

During training, SpecAugment (Park et al., 2019) is applied as data augmentation, consisting of frequency masking (mask size 20, 2 masks) and time masking (mask size 50, 2 masks). In the contrastive learning framework, each sample is augmented twice to produce two distinct views, and the contrastive loss encourages the model to produce similar embeddings for both views of the same sample while distinguishing them from views of other samples.

### 3.4 Class Imbalance Handling

The ICBHI dataset exhibits significant class imbalance: Normal (3,642 samples, 52.8%), Crackle (1,864, 27.0%), Wheeze (886, 12.8%), and Both (506, 7.3%). To mitigate this, the cross-entropy component of the hybrid loss uses inverse-frequency class weights, assigning higher penalty to misclassifications of minority classes (e.g., weight ~3.5 for Both versus ~0.48 for Normal). The contrastive loss component additionally helps address imbalance by learning a structured embedding space where minority-class samples cluster tightly, improving their separability even when few examples are available.

### 3.5 Training Configuration

The 50-epoch model was trained with the Adam optimizer (learning rate 1e-4, weight decay 1e-4) using a cosine annealing learning rate schedule (T_max = 40, eta_min = 1e-6). Batch size was 128 on a Google Colab T4 GPU. An extended 400-epoch training run is planned to improve sensitivity to abnormal classes, as the contrastive objective benefits substantially from longer training to form well-separated clusters.

## 4. References

1. Moummad, I., & Farrugia, N. (2023). Pretraining respiratory sound representations using metadata and contrastive learning. *IEEE International Workshop on Machine Learning for Signal Processing (MLSP)*.

2. Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 28, 2880–2894.

3. Khosla, P., Tewari, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised contrastive learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 18661–18673.

4. Park, D. S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A simple data augmentation method for automatic speech recognition. *Interspeech*, 2613–2617.

5. Gunel, B., Du, J., Conneau, A., & Stoyanov, V. (2021). Supervised contrastive learning for pre-trained language model fine-tuning. *International Conference on Learning Representations (ICLR)*.

6. Rocha, B. M., et al. (2019). An open access database for the evaluation of respiratory sound classification algorithms. *Physiological Measurement*, 40(3), 035001.
