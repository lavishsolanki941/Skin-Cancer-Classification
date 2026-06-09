# Skin Cancer Classification Using Machine Learning

> A comparative study of classical machine learning models on the HAM10000 dataset for automated skin lesion classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-green?style=flat&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Team](#team)
- [Future Scope](#future-scope)

---

## Overview

Skin cancer is one of the fastest-growing cancers globally. Early and accurate diagnosis is critical for improving patient survival rates. Traditional diagnosis relies on manual interpretation of dermatoscopic images — a process that is subjective, time-consuming, and prone to human error.

This project presents a **classical machine learning approach (non-deep-learning)** for classifying dermatoscopic skin lesion images using the HAM10000 benchmark dataset. The goal was to build a **transparent, interpretable, and CPU-efficient** pipeline that achieves meaningful classification accuracy without requiring GPUs or deep learning infrastructure.

**Key Highlights:**
- End-to-end ML pipeline from raw images to classification
- Handcrafted feature engineering (no CNN feature extraction)
- Comparative benchmarking of 3 classical ML models
- Best model: SVM (RBF Kernel) — highest accuracy on test set
- Runs entirely on standard CPU hardware (laptop-friendly)

---

## Dataset

**HAM10000 — Human Against Machine with 10,000 training images**

| Property | Details |
|---|---|
| Total Images | 10,015 dermatoscopic images |
| Resolution | 600 × 450 pixels (JPEG) |
| Diagnostic Classes | 7 |
| Metadata | HAM10000_metadata.csv |
| Source | [Kaggle / ISIC Archive](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) |

**Diagnostic Classes:**

| Label | Condition |
|---|---|
| mel | Melanoma |
| nv | Melanocytic nevi |
| bcc | Basal cell carcinoma |
| akiec | Actinic keratoses |
| bkl | Benign keratosis |
| df | Dermatofibroma |
| vasc | Vascular lesion |

---

## Methodology

- Preprocessed dermatoscopic images using OpenCV.
- Extracted handcrafted image features from resized grayscale images.
- Split the dataset into training and testing sets.
- Trained and evaluated Logistic Regression, Random Forest, and SVM models.
- Compared model performance using Accuracy, Precision, Recall, and F1-score.

### Pipeline Overview

```
Raw Images → Preprocessing → Feature Extraction → Scaling → Train/Test Split → Model Training → Evaluation
```

### 1. Preprocessing
- Images loaded using **OpenCV**
- Resized to **32×32 pixels** for computational efficiency
- Converted to **grayscale**
- Pixel values normalized by dividing by 255.0

### 2. Feature Extraction (Handcrafted)
Each image is represented as a 1,027-dimensional feature vector:

| Feature | Description | Dimensions |
|---|---|---|
| Mean Intensity | Average pixel brightness; distinguishes dark melanomas from benign lesions | 1 |
| Standard Deviation | Texture and contrast; malignant lesions have higher contrast | 1 |
| Max Intensity | Brightest pixel region | 1 |
| Flattened Pixels | Full spatial structure of 32×32 grayscale image | 1,024 |

### 3. Feature Scaling
Applied **StandardScaler** (zero mean, unit variance) — required for Logistic Regression and SVM to converge correctly.

### 4. Train-Test Split
- **80% training / 20% testing**
- `random_state=42` for reproducibility

---

## Models Used

### Logistic Regression
- Baseline linear model
- Solver: `lbfgs`
- Fast and interpretable
- Limitation: Struggles with non-linear class boundaries

### Random Forest
- Ensemble of **300 decision trees**
- `min_samples_split=3`, `min_samples_leaf=2`
- Handles non-linearity robustly
- Strong performance on tabular image features

### SVM (RBF Kernel) ⭐ Best Model
- High-dimensional mapper via RBF kernel
- Handles 1,027-dimensional feature space effectively
- Strongest generalization across all 7 classes
- Best accuracy on the test set

---

## Results

Comparative performance on ~800-image test set:

| Model | Performance |
|---|---|
| SVM (RBF Kernel) |  Best Performance |
| Random Forest (300 trees) | Strong Performance |
| Logistic Regression | Baseline Performance |

**Key Finding:** SVM significantly outperformed linear models, confirming that skin lesion features exist in a complex, non-linear space that simple linear boundaries cannot capture effectively.

Evaluation metrics used: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Project Structure

```
skin-cancer-classification/
│
├── data/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
│
├── notebooks/
│   └── skin_cancer_classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── model_training.py
│
├── results/
│   └── classification_report.txt
│
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### Prerequisites
```bash
pip install numpy pandas opencv-python scikit-learn matplotlib
```

### Requirements
```
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/lavishsolanki941/skin-cancer-classification.git
cd skin-cancer-classification
```

2. Download the HAM10000 dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) and place it in the `data/` folder.

3. Run the notebook:
```bash
jupyter notebook notebooks/skin_cancer_classification.ipynb
```

---

## Team

| Name | Student ID |
|---|---|
| Lavish Solanki 
| Laksh Johri 
| Udit Malik 

Bennett University, Greater Noida — B.Tech CSE (AI/ML), 2024–2028

---

## Future Scope

- **Deep Learning:** Implement CNN-based models (ResNet, EfficientNet) for higher accuracy using full-resolution images
- **Advanced Features:** Extract HOG (Histogram of Oriented Gradients) and GLCM (texture analysis) features
- **Class Imbalance:** Address using SMOTE or weighted loss functions
- **Web Deployment:** Build a real-time Flask/FastAPI web app for dermatologist use

---

## References

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). HAM10000 dataset. *Scientific Data*.
2. [Scikit-learn Documentation](https://scikit-learn.org)
3. [OpenCV Documentation](https://docs.opencv.org)
4. [Matplotlib Documentation](https://matplotlib.org)

---

*This project was built as part of the B.Tech Computer Science (AI/ML) curriculum at Bennett University.*
