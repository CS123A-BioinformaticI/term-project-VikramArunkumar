# 🧠 Brain Tumor Detection using Deep Learning

A machine learning-based solution for automated detection and classification of brain tumors from MRI images using a Convolutional Neural Network (CNN).

## 📋 Overview

This project implements a Sequential CNN model to classify brain MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

The model aims to assist in early diagnosis and support medical professionals in identifying brain tumor types from MRI images.

## 🎯 Dataset

- **Source**: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Image Size**: 299 × 299 pixels
- **Split**: Separate training and testing sets

## 🏗️ Model Architecture

The model uses a Sequential architecture with the following characteristics:

- **Input Shape**: (299, 299, 3)
- **Architecture**: Fully connected Convolutional Neural Network
- **Regularization**: 
  - Dropout Layer 1: 0.3
  - Dropout Layer 2: 0.25
- **Optimizer**: Adamax
- **Loss Function**: Categorical Cross-Entropy (CCE)
- **Output**: 4 classes with Softmax activation

### Key Features
✅ Dropout regularization to prevent overfitting  
✅ Adamax optimizer for adaptive learning rates  
✅ Categorical cross-entropy for multi-class classification

## 🛠️ Dependencies 

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas, Pillow
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Python 3.x (3.8 or higher)

## 📝 Instructions
1. Clone the repository
2. Install the required dependencies
3. Organize the dataset into the following structure:
```
brain-tumor-detection/
│
├── Training/              # Training images organized by class
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
├── Testing/               # Testing images organized by class
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
├── BrainTumorDetection.ipynb           # Main Jupyter notebook with model implementation
│
└── README.md             # Project documentation
```
4. Update dataset path in BrainTumorDetection.ipynb to match local directory structure

## 📊 Our Results
- **Training Accuracy**: 99.98%
- **Testing Accuracy**: 99.85%
- **Precision**: 99.54%
- **Recall**: 99.54%
- **F1-Score**: 99.54%






