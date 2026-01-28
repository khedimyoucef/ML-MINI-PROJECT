"""
Weights & Biases Configuration Module

This module provides centralized configuration and utilities for W&B integration
in the Semi-Supervised Grocery Classification MLOps pipeline.
"""

import os
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, List


# ============================================================================
# Project Configuration
# ============================================================================

# W&B Project Settings
WANDB_PROJECT = "grocery-classification-mlops"
WANDB_ENTITY = None  # Set to your W&B username/team, or None for default

# Default experiment configuration
DEFAULT_CONFIG = {
    # Data parameters
    "data_dir": "DS2GROCERIES",
    "labeled_ratio": 0.2,
    "max_samples_per_class": None,
    
    # Training parameters
    "fine_tune_epochs": 10,
    "fine_tune_lr": 1e-4,
    "batch_size": 16,
    
    # Semi-supervised algorithm parameters
    "algorithms": ["label_propagation", "label_spreading", "self_training"],
    "n_neighbors": 10,
    "max_iter": 2000,
    "kernel": "knn",
    
    # Feature extraction
    "feature_dim": 2048,
    "backbone": "resnet50",
    
    # Class information
    "num_classes": 20,
    "class_names": [
        "bacon", "banana", "bread", "broccoli", "butter",
        "carrots", "cheese", "chicken", "cucumber", "eggs",
        "fish", "lettuce", "milk", "onions", "peppers",
        "potatoes", "sausages", "spinach", "tomato", "yogurt"
    ],
    
    # MLOps settings
    "upload_artifacts": False,  # Default to False (save data), require flag to upload
}


# ============================================================================
# W&B Initialization Utilities
# ============================================================================

def init_wandb(
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    job_type: str = "training",
    resume: Optional[str] = None
) -> wandb.run:
    """
    Initialize a W&B run with project defaults.
    
    Args:
        run_name: Optional name for the run
        config: Configuration dict to log (merged with defaults)
        tags: Optional tags for the run
        notes: Optional notes/description
        job_type: Type of job (training, evaluation, sweep, etc.)
        resume: Resume mode (allow, must, never, auto)
        
    Returns:
        W&B run object
    """
    # Merge provided config with defaults
    full_config = DEFAULT_CONFIG.copy()
    if config:
        full_config.update(config)
    
    # Default tags
    default_tags = ["semi-supervised", "grocery-classification", "mlops"]
    if tags:
        default_tags.extend(tags)
    
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=full_config,
        tags=default_tags,
        notes=notes,
        job_type=job_type,
        resume=resume,
        reinit=True
    )
    
    return run


def finish_wandb():
    """Properly finish the current W&B run."""
    if wandb.run is not None:
        wandb.finish()


# ============================================================================
# Logging Utilities
# ============================================================================

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix"
):
    """
    Log a confusion matrix to W&B.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Title for the chart
    """
    if wandb.run is None:
        return
        
    if class_names is None:
        class_names = DEFAULT_CONFIG["class_names"]
    
    # Create W&B confusion matrix
    wandb.log({
        title: wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names
        )
    })


def log_classification_report(
    report: Dict[str, Any],
    prefix: str = ""
):
    """
    Log classification report metrics to W&B.
    
    Args:
        report: Classification report dict from sklearn
        prefix: Prefix for metric names
    """
    if wandb.run is None:
        return
    
    metrics = {}
    
    # Log per-class metrics
    for class_name in DEFAULT_CONFIG["class_names"]:
        if class_name in report:
            for metric, value in report[class_name].items():
                if isinstance(value, (int, float)):
                    metrics[f"{prefix}{class_name}/{metric}"] = value
    
    # Log aggregate metrics
    for key in ["accuracy", "macro avg", "weighted avg"]:
        if key in report:
            if isinstance(report[key], dict):
                for metric, value in report[key].items():
                    metrics[f"{prefix}{key}/{metric}"] = value
            else:
                metrics[f"{prefix}{key}"] = report[key]
    
    wandb.log(metrics)


def log_model_artifact(
    model_path: str,
    artifact_name: str,
    artifact_type: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
    aliases: Optional[List[str]] = None
):
    """
    Log a model file as a W&B artifact.
    
    Args:
        model_path: Path to the model file
        artifact_name: Name for the artifact
        artifact_type: Type of artifact (model, dataset, etc.)
        metadata: Optional metadata dict
        aliases: Optional list of aliases (e.g., ["latest", "best"])
    """
    if wandb.run is None:
        return
    
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {}
    )
    
    # Check if uploads are enabled
    if wandb.config.get("upload_artifacts", True):
        artifact.add_file(model_path)
    else:
        # Just log metadata to the file, but don't upload the blob
        print(f"  Note: Skipping upload of {model_path} (upload_artifacts=False)")
    
    wandb.log_artifact(artifact, aliases=aliases or ["latest"])


def log_dataset_artifact(
    artifact_name: str,
    metadata: Dict[str, Any],
    sample_images: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None
):
    """
    Log dataset metadata as a W&B artifact (without raw images for size).
    
    Args:
        artifact_name: Name for the artifact
        metadata: Dataset metadata (statistics, paths, etc.)
        sample_images: Optional list of sample image paths to include
        aliases: Optional list of aliases
    """
    if wandb.run is None:
        return
    
    artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        metadata=metadata
    )
    
    # Add sample images if provided and uploads are enabled
    if sample_images and wandb.config.get("upload_artifacts", True):
        for img_path in sample_images[:50]:  # Limit to 50 samples
            if os.path.exists(img_path):
                artifact.add_file(img_path, name=f"samples/{Path(img_path).name}")
    elif sample_images:
        print(f"  Note: Skipping upload of sample images (upload_artifacts=False)")
    
    wandb.log_artifact(artifact, aliases=aliases or ["latest"])


def log_images(
    images: List[Any],
    captions: Optional[List[str]] = None,
    key: str = "images"
):
    """
    Log images to W&B.
    
    Args:
        images: List of images (PIL, numpy, or paths)
        captions: Optional captions for each image
        key: Key name for the images
    """
    if wandb.run is None:
        return
    
    wandb_images = []
    for i, img in enumerate(images):
        caption = captions[i] if captions and i < len(captions) else None
        wandb_images.append(wandb.Image(img, caption=caption))
    
    wandb.log({key: wandb_images})


def create_summary_table(
    data: List[Dict[str, Any]],
    columns: List[str],
    table_name: str = "summary"
):
    """
    Create and log a W&B Table.
    
    Args:
        data: List of row dictionaries
        columns: Column names
        table_name: Name for the table
    """
    if wandb.run is None:
        return
    
    table = wandb.Table(columns=columns)
    for row in data:
        table.add_data(*[row.get(col, "") for col in columns])
    
    wandb.log({table_name: table})


# ============================================================================
# Sweep Configuration
# ============================================================================

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "test_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "labeled_ratio": {
            "values": [0.1, 0.15, 0.2, 0.25, 0.3]
        },
        "algorithm": {
            "values": ["label_propagation", "label_spreading", "self_training"]
        },
        "n_neighbors": {
            "min": 5,
            "max": 20
        },
        "fine_tune_epochs": {
            "values": [5, 10, 15]
        },
        "fine_tune_lr": {
            "min": 1e-5,
            "max": 1e-3
        }
    }
}


def create_sweep(config: Optional[Dict] = None) -> str:
    """
    Create a W&B sweep and return the sweep ID.
    
    Args:
        config: Optional sweep configuration (uses default if None)
        
    Returns:
        Sweep ID
    """
    sweep_config = config or SWEEP_CONFIG
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY)
    return sweep_id


# ============================================================================
# Monitoring Utilities
# ============================================================================

def log_inference_prediction(
    image_path: str,
    predicted_class: str,
    true_class: Optional[str],
    confidence: float,
    probabilities: Optional[Dict[str, float]] = None,
    inference_time_ms: Optional[float] = None
):
    """
    Log a single inference prediction for monitoring.
    
    Args:
        image_path: Path to the image
        predicted_class: Predicted class name
        true_class: True class name (if known)
        confidence: Prediction confidence
        probabilities: Optional dict of class probabilities
        inference_time_ms: Optional inference time in milliseconds
    """
    if wandb.run is None:
        return
    
    log_data = {
        "inference/predicted_class": predicted_class,
        "inference/confidence": confidence,
    }
    
    if true_class:
        log_data["inference/true_class"] = true_class
        log_data["inference/correct"] = int(predicted_class == true_class)
    
    if inference_time_ms:
        log_data["inference/time_ms"] = inference_time_ms
    
    # Log the image with prediction
    if os.path.exists(image_path):
        caption = f"Pred: {predicted_class} ({confidence:.1%})"
        if true_class:
            caption += f" | True: {true_class}"
        log_data["inference/image"] = wandb.Image(image_path, caption=caption)
    
    wandb.log(log_data)


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("W&B Configuration Module")
    print("=" * 50)
    print(f"Project: {WANDB_PROJECT}")
    print(f"Entity: {WANDB_ENTITY or 'default'}")
    print(f"\nDefault Config:")
    for key, value in DEFAULT_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Configuration loaded successfully!")
    print("\nTo start a W&B run, use:")
    print("  from src.wandb_config import init_wandb")
    print("  run = init_wandb(run_name='my-experiment')")
