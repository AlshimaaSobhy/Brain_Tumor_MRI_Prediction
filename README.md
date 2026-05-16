# Brain Tumor Classification using Deep Learning

## Overview
This project presents a deep learning-based system for **brain tumor classification** using MRI images. The model classifies brain MRI scans into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The project compares the performance of:
- Custom CNN (Baseline Model)
- VGG16 Transfer Learning
- ResNet50 Transfer Learning

The main objective is to evaluate different deep learning architectures for medical image classification and identify the most effective model for brain tumor detection.

---

# Dataset

The dataset contains MRI brain scan images divided into:

- Training Set
- Testing Set

## Classes
- `glioma`
- `meningioma`
- `pituitary`
- `notumor`

## Dataset Structure

```bash
Brain_Tumor_Dataset/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Google Colab

---

# Data Preprocessing

The following preprocessing and augmentation techniques were applied:

- Image resizing to `128 × 128`
- Pixel normalization (`rescale=1./255`)
- Rotation augmentation
- Width shifting
- Height shifting
- Shearing
- Zoom augmentation
- Horizontal flipping
- Validation split (`20%`)

---

# Models Implemented

## 1. CNN Baseline Model

Custom Convolutional Neural Network consisting of:

- Conv2D layers
- Batch Normalization
- MaxPooling layers
- Dropout layer
- Dense layers

---

## 2. VGG16 Transfer Learning

Pretrained `VGG16` model with:

- Frozen convolutional base
- Custom dense classifier
- Dropout regularization

---

## 3. ResNet50 Transfer Learning

Pretrained `ResNet50` model with:

- Frozen pretrained layers
- Global Average Pooling
- Fully connected classifier

---

# Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | Categorical Crossentropy |
| Batch Size | 32 |
| Image Size | 128×128 |

## Epochs
- CNN → 20 Epochs
- VGG16 → 15 Epochs
- ResNet50 → 15 Epochs

## Callbacks Used
- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau

---

# Results

| Model | Test Accuracy |
|---|---|
| CNN Baseline | 79.79% |
| VGG16 Transfer Learning | 81.85% |
| ResNet50 Transfer Learning | 51.64% |

---

# Best Performing Model

The **VGG16 Transfer Learning** model achieved the highest performance with:

- **81.85% Test Accuracy**
- Better generalization capability
- Strong classification performance across all classes

---

# Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve & AUC

---

# Sample Outputs

## Confusion Matrix
The project generates confusion matrices to visualize model predictions and classification performance.

## ROC Curves
ROC curves are plotted for all tumor classes to compare AUC scores.

## Training Curves
Training and validation accuracy/loss graphs are generated for performance analysis.

---

# Key Findings

- Transfer learning improved model performance significantly.
- VGG16 outperformed both the custom CNN and ResNet50.
- ResNet50 underperformed due to insufficient fine-tuning and dataset complexity.
- Data augmentation reduced overfitting and improved generalization.

---

# Future Improvements

Possible enhancements include:

- Fine-tuning pretrained layers
- Using larger image resolutions
- Implementing EfficientNet or DenseNet
- Applying advanced augmentation techniques
- Hyperparameter optimization
- Deploying the model as a web application

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Notebook

Open the notebook using:

- Jupyter Notebook
- Google Colab

Then execute all cells sequentially.

---

# Project Workflow

```text
Dataset Collection
        ↓
Data Preprocessing & Augmentation
        ↓
Model Building
        ↓
Training
        ↓
Evaluation
        ↓
Performance Comparison
