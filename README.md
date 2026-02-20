# Kannada Character Recognition using CNN

## Overview
This project implements a deep learning pipeline for recognizing Kannada characters using Convolutional Neural Networks (CNN). The pipeline processes input images, extracts features, and classifies them into predefined Kannada character classes. The dataset includes characters in clean and degraded forms, such as burnt or edge-detected variations.

---

## Features
- **Preprocessing**: Edge detection (Canny, Sobel), normalization, and augmentation.
- **Feature Extraction**: Automatic feature detection using CNN layers.
- **Classification**: Character classification using fully connected layers and a softmax output layer.
- **Dataset**: Supports Kannada characters in various degraded conditions.

---

## Pipeline Architecture

### 1. Input Acquisition
- Kannada character images (128x128 resolution).

### 2. Preprocessing
- Edge detection to highlight character features.
- Normalization of pixel values.
- Optional data augmentation for model generalization.

### 3. Feature Extraction
- CNN layers for detecting local patterns (edges, curves, shapes).
- Pooling layers for dimensionality reduction.

### 4. Classification
- Fully connected layers for feature vector mapping.
- Softmax output for class probabilities.

### 5. Postprocessing
- Validation and refinement of predicted output.

---

## Dataset
- 47 Kannada character classes.
- 128x128 grayscale images.
- Variations include:
  - Clean characters.
  - Damaged/burnt characters.
  - Edge-detected characters.

---

## Requirements

### Software
- Python >= 3.8
- TensorFlow >= 2.0
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn


## Results
- **Accuracy**: Achieved 82% accuracy on the test set.

---

