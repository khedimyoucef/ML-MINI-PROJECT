"""
Training Pipeline with Weights & Biases Integration

This module provides the complete training pipeline for the semi-supervised 
grocery classification project with full MLOps experiment tracking via W&B.

Features:
- Experiment tracking with hyperparameters and metrics
- Confusion matrix and classification report logging
- Model artifact versioning
- Training time tracking
- Sweep integration for hyperparameter optimization
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import pickle
from torch.utils.data import Subset, DataLoader
import torch
import wandb
from sklearn.metrics import classification_report

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    GroceryDataset, 
    create_semi_supervised_split,
    CLASS_NAMES,
    get_dataset_stats,
    get_training_transform,
    get_image_transform
)
from src.feature_extraction import FeatureExtractor
from src.semi_supervised import (
    SemiSupervisedClassifier,
    train_and_evaluate,
)
from src.wandb_config import (
    init_wandb,
    finish_wandb,
    log_metrics,
    log_confusion_matrix,
    log_classification_report,
    log_model_artifact,
    create_summary_table,
    DEFAULT_CONFIG,
    WANDB_PROJECT
)


def train_with_wandb(
    data_dir: str = None,
    output_dir: str = "models",
    features_dir: str = "features",
    labeled_ratio: float = 0.2,
    max_samples_per_class: int = None,
    algorithms: list = None,
    device: str = None,
    fine_tune_epochs: int = 10,
    fine_tune_lr: float = 1e-4,
    n_neighbors: int = 10,
    max_iter: int = 2000,
    run_name: str = None,
    sweep_mode: bool = False
):
    """
    Complete training pipeline with W&B experiment tracking.
    
    Args:
        data_dir: Path to dataset
        output_dir: Directory to save models
        features_dir: Directory for cached features
        labeled_ratio: Fraction of labeled data
        max_samples_per_class: Limit samples per class (for testing)
        algorithms: List of algorithms to train
        device: Device for feature extraction (cuda/cpu/dml)
        fine_tune_epochs: Epochs for fine-tuning
        fine_tune_lr: Learning rate for fine-tuning
        n_neighbors: Number of neighbors for graph-based algorithms
        max_iter: Maximum iterations for algorithms
        run_name: Optional name for the W&B run
        sweep_mode: If True, use wandb.config for hyperparameters
    """
    
    # Set defaults
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent / "DS2GROCERIES")
    
    if algorithms is None:
        algorithms = ['label_propagation', 'label_spreading', 'self_training']
    
    # Prepare configuration
    config = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "features_dir": features_dir,
        "labeled_ratio": labeled_ratio,
        "max_samples_per_class": max_samples_per_class,
        "algorithms": algorithms,
        "device": device,
        "fine_tune_epochs": fine_tune_epochs,
        "fine_tune_lr": fine_tune_lr,
        "n_neighbors": n_neighbors,
        "max_iter": max_iter,
    }
    
    # Initialize W&B
    if not sweep_mode:
        run = init_wandb(
            run_name=run_name or f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config,
            tags=["training", "semi-supervised"],
            notes=f"Training with {labeled_ratio*100:.0f}% labeled data",
            job_type="training"
        )
    else:
        # In sweep mode, wandb.init is already called
        # Update config from sweep
        labeled_ratio = wandb.config.get("labeled_ratio", labeled_ratio)
        if "algorithm" in wandb.config:
            algorithms = [wandb.config["algorithm"]]
        n_neighbors = wandb.config.get("n_neighbors", n_neighbors)
        fine_tune_epochs = wandb.config.get("fine_tune_epochs", fine_tune_epochs)
        fine_tune_lr = wandb.config.get("fine_tune_lr", fine_tune_lr)
    
    try:
        print("=" * 60)
        print("Semi-Supervised Grocery Classification Training (W&B)")
        print("=" * 60)
        print(f"W&B Project: {WANDB_PROJECT}")
        print(f"Run: {wandb.run.name if wandb.run else 'N/A'}")
        
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(features_dir).mkdir(parents=True, exist_ok=True)
        
        # Get and log dataset statistics
        print("\nDataset Statistics:")
        stats = get_dataset_stats(data_dir)
        for split, split_stats in stats['splits'].items():
            print(f"  {split}: {split_stats['total']} images")
        
        # Log dataset stats to W&B
        log_metrics({
            "dataset/train_samples": stats['splits'].get('train', {}).get('total', 0),
            "dataset/test_samples": stats['splits'].get('test', {}).get('total', 0),
            "dataset/val_samples": stats['splits'].get('val', {}).get('total', 0),
            "dataset/num_classes": len(CLASS_NAMES),
        })
        
        # ============================================================
        # 1. FEATURE EXTRACTION & FINE-TUNING
        # ============================================================
        print("\n" + "=" * 60)
        print("Phase 1: Feature Extraction & Fine-Tuning")
        print("=" * 60)
        
        extractor = FeatureExtractor(device=device)
        
        # Prepare training data
        print("\n  Preparing training data...")
        train_dataset_full = GroceryDataset(
            data_dir, 
            split='train',
            transform=get_image_transform(),
            max_samples_per_class=max_samples_per_class
        )
        
        train_dataset_aug = GroceryDataset(
            data_dir, 
            split='train',
            transform=get_training_transform(),
            max_samples_per_class=max_samples_per_class
        )
        
        # Create semi-supervised split
        print(f"\nCreating semi-supervised split ({labeled_ratio*100:.0f}% labeled)...")
        y_train_full = train_dataset_full.get_labels()
        
        y_train_ssl, labeled_mask = create_semi_supervised_split(
            y_train_full, 
            labeled_ratio=labeled_ratio
        )
        n_labeled = labeled_mask.sum()
        n_unlabeled = (~labeled_mask).sum()
        print(f"  Labeled samples: {n_labeled}")
        print(f"  Unlabeled samples: {n_unlabeled}")
        
        # Log split info
        log_metrics({
            "data/labeled_samples": int(n_labeled),
            "data/unlabeled_samples": int(n_unlabeled),
            "data/labeled_ratio": labeled_ratio,
        })
        
        # Fine-tune on labeled data
        print("\nFine-tuning ResNet50 on labeled subset...")
        fine_tune_start = time.time()
        
        labeled_indices = np.where(labeled_mask)[0]
        labeled_subset = Subset(train_dataset_aug, labeled_indices)
        
        labeled_loader = DataLoader(
            labeled_subset,
            batch_size=16,
            shuffle=True,
            num_workers=0
        )
        
        # Fine-tune with progress logging
        extractor.fine_tune(
            labeled_loader,
            epochs=fine_tune_epochs,
            learning_rate=fine_tune_lr
        )
        
        fine_tune_time = time.time() - fine_tune_start
        print(f"  Fine-tuning completed in {fine_tune_time:.1f}s")
        
        log_metrics({
            "timing/fine_tune_seconds": fine_tune_time,
            "training/fine_tune_epochs": fine_tune_epochs,
            "training/fine_tune_lr": fine_tune_lr,
        })
        
        # Save feature extractor
        extractor_path = os.path.join(output_dir, "feature_extractor.pth")
        extractor.save(extractor_path)
        print(f"  Saved fine-tuned feature extractor to {extractor_path}")
        
        # Log feature extractor as artifact
        log_model_artifact(
            extractor_path,
            artifact_name="feature-extractor",
            artifact_type="model",
            metadata={
                "backbone": "resnet50",
                "fine_tune_epochs": fine_tune_epochs,
                "labeled_samples": int(n_labeled),
            },
            aliases=["latest", f"labeled-{labeled_ratio}"]
        )
        
        # ============================================================
        # 2. EXTRACT FEATURES
        # ============================================================
        print("\n" + "=" * 60)
        print("Phase 2: Feature Extraction")
        print("=" * 60)
        
        feature_start = time.time()
        
        # Train features
        train_cache = os.path.join(features_dir, f'train_features_finetuned_{labeled_ratio}.npz')
        X_train, y_train = extractor.extract_and_cache(
            train_dataset_full, 
            train_cache,
            force_recompute=True
        )
        print(f"  Training features shape: {X_train.shape}")
        
        # Test data
        print("\n  Processing test data...")
        test_dataset = GroceryDataset(
            data_dir, 
            split='test',
            max_samples_per_class=max_samples_per_class
        )
        test_cache = os.path.join(features_dir, 'test_features_resnet50.npz')
        X_test, y_test = extractor.extract_and_cache(
            test_dataset, 
            test_cache,
            force_recompute=True
        )
        print(f"  Test features shape: {X_test.shape}")
        
        feature_time = time.time() - feature_start
        
        log_metrics({
            "timing/feature_extraction_seconds": feature_time,
            "features/train_shape": list(X_train.shape),
            "features/test_shape": list(X_test.shape),
            "features/dimension": X_train.shape[1],
        })
        
        # ============================================================
        # 3. TRAIN SEMI-SUPERVISED MODELS
        # ============================================================
        print("\n" + "=" * 60)
        print("Phase 3: Training Semi-Supervised Models")
        print("=" * 60)
        
        results = {}
        all_results_data = []
        
        for algo in algorithms:
            print(f"\n  Training {algo}...")
            algo_start = time.time()
            
            try:
                clf, result = train_and_evaluate(
                    X_train, y_train_ssl,
                    X_test, y_test,
                    algorithm=algo,
                    class_names=CLASS_NAMES,
                    y_train_true=y_train,
                    kernel='knn',
                    n_neighbors=n_neighbors,
                    max_iter=max_iter
                )
                
                algo_time = time.time() - algo_start
                
                # Save model
                model_path = os.path.join(output_dir, f'{algo}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(clf, f)
                print(f"  Saved model to {model_path}")
                
                # Store results
                results[algo] = {
                    'train_accuracy': float(result['train_accuracy']),
                    'train_accuracy_full': float(result['train_accuracy_full']) if result['train_accuracy_full'] is not None else None,
                    'test_accuracy': float(result['test_accuracy']),
                    'f1_score': float(result['f1_score']),
                    'recall_score': float(result['recall_score']),
                    'precision_score': float(result['precision_score']),
                    'n_labeled': int(result['n_labeled']),
                    'n_unlabeled': int(result['n_unlabeled']),
                    'training_time': algo_time
                }
                
                # Log metrics to W&B
                log_metrics({
                    f"{algo}/train_accuracy": result['train_accuracy'],
                    f"{algo}/train_accuracy_full": result['train_accuracy_full'] or 0,
                    f"{algo}/test_accuracy": result['test_accuracy'],
                    f"{algo}/f1_score": result['f1_score'],
                    f"{algo}/recall_score": result['recall_score'],
                    f"{algo}/precision_score": result['precision_score'],
                    f"{algo}/training_time": algo_time,
                })
                
                # Log confusion matrix
                y_pred = clf.predict(X_test)
                log_confusion_matrix(
                    y_test.tolist(),
                    y_pred.tolist(),
                    class_names=CLASS_NAMES,
                    title=f"{algo}_confusion_matrix"
                )
                
                # Get and log classification report
                report = classification_report(
                    y_test, y_pred,
                    target_names=CLASS_NAMES,
                    output_dict=True
                )
                log_classification_report(report, prefix=f"{algo}/")
                
                # Log model artifact
                log_model_artifact(
                    model_path,
                    artifact_name=f"{algo}-model",
                    artifact_type="model",
                    metadata={
                        "algorithm": algo,
                        "test_accuracy": result['test_accuracy'],
                        "f1_score": result['f1_score'],
                        "labeled_ratio": labeled_ratio,
                        "n_neighbors": n_neighbors,
                    },
                    aliases=["latest"]
                )
                
                # Add to summary table data
                all_results_data.append({
                    "algorithm": algo,
                    "test_accuracy": f"{result['test_accuracy']:.4f}",
                    "f1_score": f"{result['f1_score']:.4f}",
                    "precision": f"{result['precision_score']:.4f}",
                    "recall": f"{result['recall_score']:.4f}",
                    "training_time": f"{algo_time:.1f}s"
                })
                
            except Exception as e:
                print(f"  Error training {algo}: {e}")
                results[algo] = {'error': str(e)}
                log_metrics({f"{algo}/error": str(e)})
        
        # Create summary table
        if all_results_data:
            create_summary_table(
                all_results_data,
                columns=["algorithm", "test_accuracy", "f1_score", "precision", "recall", "training_time"],
                table_name="algorithm_comparison"
            )
        
        # ============================================================
        # 4. SAVE RESULTS & SUMMARY
        # ============================================================
        print("\n" + "=" * 60)
        print("Phase 4: Saving Results")
        print("=" * 60)
        
        # Add metadata
        results['metadata'] = {
            'labeled_ratio': labeled_ratio,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_dim': X_train.shape[1],
            'n_classes': len(CLASS_NAMES),
            'class_names': CLASS_NAMES,
            'n_neighbors': n_neighbors,
            'fine_tune_epochs': fine_tune_epochs,
            'fine_tune_lr': fine_tune_lr,
        }
        
        results_path = os.path.join(output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved training results to {results_path}")
        
        # Find best model
        best_algo = None
        best_accuracy = 0
        for algo in algorithms:
            if 'error' not in results.get(algo, {}):
                if results[algo]['test_accuracy'] > best_accuracy:
                    best_accuracy = results[algo]['test_accuracy']
                    best_algo = algo
        
        # Log summary metrics
        if best_algo:
            log_metrics({
                "summary/best_algorithm": best_algo,
                "summary/best_test_accuracy": best_accuracy,
                "test_accuracy": best_accuracy,  # For sweep optimization
            })
            
            # Update best model alias
            print(f"\nBest model: {best_algo} with {best_accuracy:.4f} test accuracy")
        
        # Print summary table
        print("\n" + "=" * 80)
        print("Training Summary:")
        print("-" * 80)
        print(f"{'Algorithm':<20} {'Test Acc':>10} {'F1 Score':>10} {'Recall':>10} {'Precision':>10} {'Time':>10}")
        print("-" * 80)
        
        for algo in algorithms:
            if 'error' not in results.get(algo, {}):
                r = results[algo]
                print(f"{algo:<20} {r['test_accuracy']:>9.2%} {r['f1_score']:>9.2f} {r['recall_score']:>9.2f} {r['precision_score']:>9.2f} {r['training_time']:>9.1f}s")
            else:
                print(f"{algo:<20} {'ERROR':>10}")
        
        print("=" * 80)
        print(f"\n✓ W&B Run completed: {wandb.run.url if wandb.run else 'N/A'}")
        
        return results
        
    finally:
        # Always finish W&B run
        if not sweep_mode:
            finish_wandb()


def run_sweep_agent():
    """Run as a sweep agent (called by wandb agent)."""
    wandb.init()
    train_with_wandb(sweep_mode=True)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Train semi-supervised models with W&B experiment tracking"
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default=str(Path(__file__).parent.parent / 'DS2GROCERIES'),
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default='features',
        help='Directory for cached features'
    )
    parser.add_argument(
        '--labeled-ratio',
        type=float,
        default=0.2,
        help='Fraction of labeled training data (0.0 to 1.0)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Max samples per class (for testing)'
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['label_propagation', 'label_spreading', 'self_training'],
        help='Algorithms to train'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for feature extraction (cuda/cpu)'
    )
    parser.add_argument(
        '--fine-tune-epochs',
        type=int,
        default=10,
        help='Epochs for fine-tuning'
    )
    parser.add_argument(
        '--fine-tune-lr',
        type=float,
        default=1e-4,
        help='Learning rate for fine-tuning'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=10,
        help='Number of neighbors for graph-based algorithms'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for the W&B run'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run as sweep agent'
    )
    
    args = parser.parse_args()
    
    if args.sweep:
        run_sweep_agent()
    else:
        train_with_wandb(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            features_dir=args.features_dir,
            labeled_ratio=args.labeled_ratio,
            max_samples_per_class=args.max_samples,
            algorithms=args.algorithms,
            device=args.device,
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_lr=args.fine_tune_lr,
            n_neighbors=args.n_neighbors,
            run_name=args.run_name
        )


if __name__ == "__main__":
    main()
