# 🛒 Semi-Supervised Grocery Image Classification

A machine learning mini-project that classifies grocery images into 20 categories using semi-supervised learning algorithms.

## 🎯 Project Overview

This project demonstrates the power of **semi-supervised learning** for image classification. The system uses only a small fraction of labeled data combined with a larger pool of unlabeled data to train classifiers that can identify grocery items from images.

### Features
- **20 Grocery Categories**: bacon, banana, bread, broccoli, butter, carrots, cheese, chicken, cucumber, eggs, fish, lettuce, milk, onions, peppers, potatoes, sausages, spinach, tomato, yogurt
- **3 SSL Algorithms**: Label Propagation, Label Spreading, Self-Training
- **Interactive Web App**: Built with Streamlit for easy image classification
- **Pretrained Feature Extraction**: Uses ResNet18 for robust image features

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train with 10% labeled data (default)
python src/train.py

# Or specify custom labeled ratio
python src/train.py --labeled-ratio 0.2

# Quick test with limited samples
python src/train.py --max-samples 50
```

### 3. Run the Web Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
ML-MINI-PROJECT/
├── DS2GROCERIES/           # Dataset
│   ├── train/              # Training images
│   ├── test/               # Test images
│   └── val/                # Validation images
├── src/
│   ├── __init__.py
│   ├── data_utils.py       # Data loading utilities
│   ├── feature_extraction.py  # ResNet18 feature extractor
│   ├── semi_supervised.py  # SSL algorithms
│   └── train.py            # Training pipeline
├── models/                 # Saved models
├── features/               # Cached feature vectors
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.MD
```

## 🔬 Semi-Supervised Learning Algorithms

### 1. Label Propagation
A graph-based method that spreads labels through a k-nearest neighbors similarity graph. Labels propagate from labeled to unlabeled samples based on feature similarity.

### 2. Label Spreading
Similar to Label Propagation but uses a normalized graph Laplacian, making it more robust to noise and outliers.

### 3. Self-Training
An iterative approach that:
1. Trains on labeled data
2. Predicts on unlabeled data
3. Adds high-confidence predictions to the training set
4. Repeats until convergence

## 🏗️ Technical Architecture

1. **Feature Extraction**: Pretrained ResNet18 (ImageNet weights) extracts 512-dimensional feature vectors from images
2. **Semi-Supervised Classification**: SSL algorithms work on the extracted features to classify images

```
Input Image → ResNet18 → 512-D Features → SSL Algorithm → Grocery Category
```

## 📊 Usage Examples

### Training with Custom Parameters

```python
from src.train import train_models

# Train with 20% labeled data
results = train_models(
    data_dir="DS2GROCERIES",
    labeled_ratio=0.2,
    algorithms=['label_propagation', 'label_spreading', 'self_training']
)
```

### Making Predictions

```python
from src.feature_extraction import FeatureExtractor
from src.semi_supervised import SemiSupervisedClassifier
from src.data_utils import CLASS_NAMES

# Load models
extractor = FeatureExtractor()
model = SemiSupervisedClassifier.load("models/label_spreading_model.joblib")

# Extract features from image
features = extractor.extract_from_path("path/to/image.jpg")

# Predict
prediction = model.predict(features.reshape(1, -1))
print(f"Predicted: {CLASS_NAMES[prediction[0]]}")
```

## 🌐 Web Application

The Streamlit app provides:

- **🏠 Home**: Project overview and quick start guide
- **🔍 Predict**: Upload images for real-time classification
- **📊 Dataset Explorer**: Browse dataset categories and statistics
- **📈 Model Performance**: Compare algorithm accuracies
- **ℹ️ About**: Technical documentation

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3+
- Streamlit 1.28+
- See `requirements.txt` for full list

## 📄 License

This project is for educational purposes (ML Mini-Project).
