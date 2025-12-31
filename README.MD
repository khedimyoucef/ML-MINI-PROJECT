# 🛒 Semi-Supervised Grocery Image Classification

A machine learning mini-project that classifies grocery images into 20 categories using semi-supervised learning algorithms.

## 🎯 Project Overview

This project demonstrates the power of **semi-supervised learning** for image classification. The system uses only a small fraction of labeled data combined with a larger pool of unlabeled data to train classifiers that can identify grocery items from images.

**Team Members**

- Khedim Youcef
- Bensetallah Soufiane
- Zitouni Ahmed
- Houari Mohamed
- **group** 1
- **sepciality** Artificial Intelligence and DataScience
- **Course**: Machine Learning

## To Test the app directly

# deployed app url

- https://ml-mini-project-4tlzmet4x5wzard2dsp9fw.streamlit.app/

# Note that :

if the deployed app is not working then i may need some time to push the entire project to git using git LFS
as the size can be multiple gigabytes for the extracted features , trained models and the full datasets

## Used Datasets

# The Main Dataset Used

https://www.kaggle.com/datasets/liamboyd1/multi-class-food-image-dataset

# The secondary Recipe Dataset Used

https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images

### Features

- **20 Grocery Categories**: bacon, banana, bread, broccoli, butter, carrots, cheese, chicken, cucumber, eggs, fish, lettuce, milk, onions, peppers, potatoes, sausages, spinach, tomato, yogurt
- **3 SSL Algorithms**: Label Propagation, Label Spreading, Self-Training
- **Interactive Web App**: Built with Streamlit for easy image classification
- **Pretrained Feature Extraction**: Uses ResNet50 for robust image features

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
├── DS2GROCERIES/           # Main grocery dataset
│   ├── train/              # Training images (20 categories)
│   ├── test/               # Test images
│   └── val/                # Validation images
├── DS3RECIPES/             # Recipe dataset
│   ├── Food Images/        # Recipe images
│   └── Food Ingredients and Recipe Dataset with Image Name Mapping.csv
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_utils.py       # Data loading and preprocessing utilities
│   ├── dataset_cleaner.py  # Dataset cleaning and validation tool
│   ├── feature_extraction.py  # ResNet50 feature extractor
│   ├── recipe_utils.py     # Recipe search utilities
│   ├── semi_supervised.py  # SSL algorithms implementation
│   └── train.py            # Training pipeline
├── models/                 # Saved trained models (.pkl files)
├── features/               # Cached feature vectors (.npz files)
├── cleaning_results/       # Dataset cleaning reports and file lists
├── OnlineTestImages/       # Test images for online prediction
├── SCREENSHOTS/            # Application screenshots
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.MD               # This file
├── problem-statement.pdf   # Project requirements
├── FinalReport.pdf         # Project report
└── .gitignore
```

## 🧹 Data Cleaning

The dataset was scraped from the web (Kaggle) and may contain corrupted, duplicate, or mislabeled images. We provide an automated cleaning tool to identify and remove problematic images.

### Running the Dataset Cleaner

```bash
# Quick scan (corrupted + duplicates only, ~5-10 min)
python src/dataset_cleaner.py --data-dir DS2GROCERIES --skip-outliers

# Full scan with AI-based outlier detection (~30-60 min)
python src/dataset_cleaner.py --data-dir DS2GROCERIES

# Use fine-tuned model for better outlier detection
python src/dataset_cleaner.py --data-dir DS2GROCERIES --model-path models/feature_extractor.pth

# Find more outliers (10% contamination rate)
python src/dataset_cleaner.py --data-dir DS2GROCERIES --contamination 0.1
```

### What the Cleaner Detects

| Type | Description | Auto-Delete Safe? |
|------|-------------|-------------------|
| **Corrupted Images** | Images that fail to load or have invalid formats | ✅ Yes |
| **Duplicate Images** | Near-identical images (using perceptual hashing) | ✅ Yes |
| **Outliers** | Images that don't fit their labeled class (AI-detected) | ⚠️ Review first |

### Output Files

After running the cleaner, results are saved in `cleaning_results/`:

| File | Description |
|------|-------------|
| `dataset_cleaning_report.html` | Visual HTML report for reviewing flagged images |
| `safe_to_delete.txt` | Corrupted + duplicate images (safe to delete) |
| `corrupted_images.txt` | List of corrupted images |
| `duplicate_images.txt` | List of duplicate images to remove |
| `outlier_images.txt` | Potential mislabeled images (review before deleting) |
| `summary.json` | Summary statistics |

### Deleting Flagged Images (PowerShell)

**Delete corrupted + duplicate images** (safe):
```powershell
Get-Content cleaning_results\safe_to_delete.txt | ForEach-Object { Remove-Item $_ -Force -ErrorAction SilentlyContinue }
```

**Delete only duplicate images**:
```powershell
Get-Content cleaning_results\duplicate_images.txt | ForEach-Object { Remove-Item $_ -Force -ErrorAction SilentlyContinue }
```

**Delete ALL outliers** (use with caution - review first!):
```powershell
Get-Content cleaning_results\outlier_images.txt | Select-Object -Skip 1 | ForEach-Object { $_.Split("`t")[0] } | ForEach-Object { Remove-Item $_ -Force -ErrorAction SilentlyContinue }
```

**Delete only high-score outliers** (score > 2.0, more aggressive):
```powershell
Get-Content cleaning_results\outlier_images.txt | Select-Object -Skip 1 | ForEach-Object { $parts = $_.Split("`t"); if ([float]$parts[2] -gt 2.0) { Remove-Item $parts[0] -Force -ErrorAction SilentlyContinue } }
```

### Recommended Cleaning Workflow

1. **Run the full scan**: `python src/dataset_cleaner.py --data-dir DS2GROCERIES`
2. **Review the HTML report**: Open `cleaning_results/dataset_cleaning_report.html` in your browser
3. **Delete safe files first**: Run the `safe_to_delete.txt` deletion command
4. **Review outliers manually**: Look at the outlier images in the report before deleting
5. **Re-train your model**: `python src/train.py`

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

1. **Feature Extraction**: Pretrained ResNet50 (ImageNet weights) extracts 2048-dimensional feature vectors from images
2. **Semi-Supervised Classification**: SSL algorithms work on the extracted features to classify images

```
Input Image → ResNet50 → 2048-D Features → SSL Algorithm → Grocery Category
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
- **📈 Model Performance**: Compare algorithm performance metrics
- **ℹ️ About**: Technical documentation

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3+
- Streamlit 1.28+
- See `requirements.txt` for full list

## 🖥️ Hardware Support

### AMD GPUs (Windows)

This project supports AMD GPUs on Windows via **DirectML**.

1. Ensure you have a DirectX 12 capable GPU.
2. The code will automatically detect if CUDA (NVIDIA) is unavailable and attempt to use DirectML.
3. You should see "Using DirectML device (AMD GPU)" in the console when training starts.

To manually verify or install:
```bash
pip install torch-directml
```

## 📄 License

This project is for educational purposes (ML Mini-Project).
