# 🔄 MLOps with Weights & Biases

This document provides comprehensive documentation for the MLOps integration in the Semi-Supervised Grocery Classification project using **Weights & Biases (W&B)**.

## 📋 Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Experiment Tracking](#experiment-tracking)
- [Data Versioning](#data-versioning)
- [Model Registry](#model-registry)
- [Hyperparameter Sweeps](#hyperparameter-sweeps)
- [Production Monitoring](#production-monitoring)
- [Quick Reference](#quick-reference)

---

## Overview

This project implements the complete MLOps lifecycle using W&B:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Version   │───▶│   Experiment    │───▶│     Model       │
│    Tracking     │    │    Tracking     │    │    Registry     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         │                      ▼                      │
         │             ┌─────────────────┐             │
         │             │  Hyperparameter │             │
         │             │     Sweeps      │             │
         │             └─────────────────┘             │
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Production Monitoring                         │
│     (Inference Tracking, Drift Detection, Performance Alerts)   │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Module | Description |
|-----------|--------|-------------|
| **Experiment Tracking** | `train_with_wandb.py` | Log hyperparameters, metrics, artifacts |
| **Data Versioning** | `data_versioning.py` | Track dataset versions with W&B Artifacts |
| **Model Registry** | `model_registry.py` | Version and compare trained models |
| **Sweeps** | `sweep_config.yaml` | Automated hyperparameter optimization |
| **Monitoring** | `inference_monitor.py` | Production inference tracking |

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `wandb>=0.16.0` along with other project dependencies.

### 2. Create W&B Account

1. Go to [wandb.ai](https://wandb.ai/) and create a free account
2. Get your API key from [Settings](https://wandb.ai/settings)

### 3. Login to W&B

```bash
wandb login
```

Or set the environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

### 4. Configure Project (Optional)

Edit `src/wandb_config.py` to customize:

```python
WANDB_PROJECT = "grocery-classification-mlops"  # Your project name
WANDB_ENTITY = None  # Your W&B username or team
```

---

## Experiment Tracking

### Basic Training with W&B

```bash
# Run training with W&B experiment tracking
python src/train_with_wandb.py

# With custom parameters
python src/train_with_wandb.py \
    --labeled-ratio 0.2 \
    --algorithms label_propagation label_spreading self_training \
    --fine-tune-epochs 10 \
    --run-name "my-experiment"
```

### What Gets Logged

| Category | Metrics | Description |
|----------|---------|-------------|
| **Dataset** | `dataset/train_samples`, `dataset/test_samples` | Dataset statistics |
| **Data Split** | `data/labeled_samples`, `data/unlabeled_samples` | SSL split info |
| **Training** | `timing/fine_tune_seconds`, `training/fine_tune_epochs` | Training timing |
| **Per-Algorithm** | `{algo}/test_accuracy`, `{algo}/f1_score` | Model performance |
| **Confusion Matrix** | `{algo}_confusion_matrix` | Visual confusion matrix |
| **Summary** | `summary/best_algorithm`, `summary/best_test_accuracy` | Best results |

### Python API Usage

```python
from src.train_with_wandb import train_with_wandb

# Run training with tracking
results = train_with_wandb(
    labeled_ratio=0.2,
    algorithms=['label_propagation', 'label_spreading'],
    run_name="custom-experiment"
)

print(f"Best accuracy: {results['summary']['best_test_accuracy']}")
```

---

## Data Versioning

### Create Dataset Artifact

```bash
# Create v1 of dataset artifact
python src/data_versioning.py \
    --artifact-name grocery-dataset \
    --version v1 \
    --description "Initial dataset with 20 classes"

# Quick quality check (no artifact creation)
python src/data_versioning.py --quality-check
```

### What Gets Tracked

- **Metadata**: Total samples, class distribution, splits
- **Sample Images**: Representative images from each class
- **Statistics**: Class imbalance ratio, per-class counts
- **Version History**: All dataset versions with lineage

### Python API

```python
from src.data_versioning import create_dataset_artifact, load_dataset_artifact

# Create artifact
artifact = create_dataset_artifact(
    data_dir="DS2GROCERIES",
    artifact_name="grocery-dataset",
    version="v1"
)

# Load artifact metadata
info = load_dataset_artifact("grocery-dataset", "v1")
print(f"Total samples: {info['metadata']['total_samples']}")
```

---

## Model Registry

### Register Models

```bash
# Register a single model
python src/model_registry.py register models/label_spreading_model.pkl \
    --name label-spreading-classifier \
    --accuracy 0.85 \
    --f1 0.84

# Register all trained models
python src/model_registry.py register-all --models-dir models

# List registered models
python src/model_registry.py list

# Compare models
python src/model_registry.py compare \
    label-propagation-classifier \
    label-spreading-classifier \
    self-training-classifier

# Promote to production
python src/model_registry.py promote label-spreading-classifier \
    --to-alias production
```

### Model Lifecycle

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Training │───▶│ Register │───▶│ Compare  │───▶│ Promote  │
│          │    │ (latest) │    │ (staging)│    │(production)│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Python API

```python
from src.model_registry import register_model, load_model_from_registry

# Register model
register_model(
    model_path="models/best_model.pkl",
    model_name="grocery-classifier",
    metrics={"test_accuracy": 0.87, "f1_score": 0.86},
    aliases=["latest", "best"]
)

# Load model for inference
model = load_model_from_registry("grocery-classifier", "production")
```

---

## Hyperparameter Sweeps

### Run a Sweep

```bash
# Step 1: Create the sweep
wandb sweep sweep_config.yaml

# Step 2: Run agents (the number specifies how many runs)
wandb agent YOUR_SWEEP_ID --count 10
```

### Sweep Configuration

The `sweep_config.yaml` defines:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `labeled_ratio` | [0.1, 0.15, 0.2, 0.25, 0.3] | Fraction of labeled data |
| `algorithm` | [label_propagation, label_spreading, self_training] | SSL algorithm |
| `n_neighbors` | 5-20 | Graph neighbors |
| `fine_tune_epochs` | [5, 10, 15] | Epochs for fine-tuning |
| `fine_tune_lr` | 1e-5 to 1e-3 | Learning rate (log scale) |

### Optimization Method

- **Method**: Bayesian optimization
- **Goal**: Maximize `test_accuracy`
- **Early termination**: Hyperband (stops poor runs early)

### Custom Sweep

```python
from src.wandb_config import create_sweep, SWEEP_CONFIG

# Create and run sweep
sweep_id = create_sweep(SWEEP_CONFIG)
print(f"Run: wandb agent {sweep_id}")
```

---

## Production Monitoring

### Simulate Production Monitoring

```bash
# Run monitoring on test set
python src/inference_monitor.py \
    --model-path models/label_spreading_model.pkl \
    --test-dir DS2GROCERIES/test \
    --n-samples 100

# Get dashboard creation help
python src/inference_monitor.py --dashboard-help
```

### What Gets Monitored

| Metric | Description |
|--------|-------------|
| `monitor/accuracy` | Running accuracy on predictions |
| `monitor/avg_confidence` | Average prediction confidence |
| `monitor/avg_inference_time_ms` | Inference latency |
| `monitor/throughput_samples_per_sec` | Processing speed |
| `monitor/drift_score` | Data drift indicator |
| `prediction_distribution` | Distribution of predictions |
| `confidence_histogram` | Confidence score distribution |

### Python API

```python
from src.inference_monitor import InferenceMonitor

# Create monitor
monitor = InferenceMonitor(
    model_path="models/label_spreading_model.pkl",
    extractor_path="models/feature_extractor.pth"
)

# Make monitored predictions
for image_path, true_label in test_data:
    pred, conf, probs = monitor.predict(image_path, true_label)
    print(f"Predicted: {pred} with {conf:.2%} confidence")

# Check for drift
monitor.detect_drift()

# Finish session
monitor.finish()
```

---

## Quick Reference

### CLI Commands

```bash
# Training with W&B tracking
python src/train_with_wandb.py --labeled-ratio 0.2 --run-name my-run

# Data versioning
python src/data_versioning.py --version v1

# Model registration
python src/model_registry.py register-all

# Hyperparameter sweep
wandb sweep sweep_config.yaml && wandb agent <ID> --count 10

# Inference monitoring
python src/inference_monitor.py --n-samples 100
```

### W&B Dashboard URLs

After running experiments, view results at:

- **Project**: `https://wandb.ai/<username>/grocery-classification-mlops`
- **Runs**: `https://wandb.ai/<username>/grocery-classification-mlops/runs`
- **Artifacts**: `https://wandb.ai/<username>/grocery-classification-mlops/artifacts`
- **Sweeps**: `https://wandb.ai/<username>/grocery-classification-mlops/sweeps`

### File Structure

```
src/
├── wandb_config.py        # W&B configuration & utilities
├── train_with_wandb.py    # Training with experiment tracking
├── data_versioning.py     # Dataset artifact management
├── model_registry.py      # Model versioning & registry
└── inference_monitor.py   # Production monitoring

sweep_config.yaml          # Hyperparameter sweep config
```

---

## Troubleshooting

### Common Issues

**"wandb: ERROR api_key"**
```bash
wandb login  # Re-authenticate
```

**Slow or failed artifact uploads**
```python
# Use smaller sample sizes for testing
python src/data_versioning.py --samples-per-class 1
```

**Sweep not running**
```bash
# Check sweep status
wandb sweep --help

# Ensure correct project
export WANDB_PROJECT=grocery-classification-mlops
```

### Best Practices

1. **Use meaningful run names**: `--run-name "exp-labeled20-neighbors10"`
2. **Tag experiments**: Tags help filter and organize runs
3. **Clean up**: Delete failed or test runs from W&B dashboard
4. **Use sweeps for optimization**: More efficient than manual tuning

---

## Additional Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Artifacts Guide](https://docs.wandb.ai/guides/artifacts)
- [W&B Sweeps Guide](https://docs.wandb.ai/guides/sweeps)
- [W&B Model Registry](https://docs.wandb.ai/guides/model_registry)
