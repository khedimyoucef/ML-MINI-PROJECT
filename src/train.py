"""
Training Pipeline

This module provides the complete training pipeline for the
semi-supervised grocery classification project.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    GroceryDataset, 
    create_semi_supervised_split,
    CLASS_NAMES,
    get_dataset_stats
)
from src.feature_extraction import FeatureExtractor, extract_dataset_features
from src.semi_supervised import (
    SemiSupervisedClassifier,
    train_and_evaluate,
    compare_algorithms
)


def train_models(
    data_dir: str = "DS2GROCERIES",
    output_dir: str = "models",
    features_dir: str = "features",
    labeled_ratio: float = 0.1,
    max_samples_per_class: int = None,
    algorithms: list = None,
    device: str = None
):
    """
    Complete training pipeline.
    
    Args:
        data_dir: Path to dataset
        output_dir: Directory to save models
        features_dir: Directory for cached features
        labeled_ratio: Fraction of labeled data
        max_samples_per_class: Limit samples per class (for testing)
        algorithms: List of algorithms to train
        device: Device for feature extraction
    """
    print("=" * 60)
    print("Semi-Supervised Grocery Classification Training")
    print("=" * 60)
    
    if algorithms is None:
        algorithms = ['label_propagation', 'label_spreading', 'self_training']
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(features_dir).mkdir(parents=True, exist_ok=True)
    
    # Print dataset stats
    print("\n📊 Dataset Statistics:")
    stats = get_dataset_stats(data_dir)
    for split, split_stats in stats['splits'].items():
        print(f"  {split}: {split_stats['total']} images")
    
    # Load or extract features
    print("\n🔧 Feature Extraction:")
    extractor = FeatureExtractor(device=device)
    
    # Training data
    print("\n  Processing training data...")
    train_dataset = GroceryDataset(
        data_dir, 
        split='train',
        max_samples_per_class=max_samples_per_class
    )
    train_cache = os.path.join(features_dir, 'train_features.npz')
    X_train, y_train = extractor.extract_and_cache(train_dataset, train_cache)
    print(f"  Training features shape: {X_train.shape}")
    
    # Test data
    print("\n  Processing test data...")
    test_dataset = GroceryDataset(
        data_dir, 
        split='test',
        max_samples_per_class=max_samples_per_class
    )
    test_cache = os.path.join(features_dir, 'test_features.npz')
    X_test, y_test = extractor.extract_and_cache(test_dataset, test_cache)
    print(f"  Test features shape: {X_test.shape}")
    
    # Create semi-supervised split
    print(f"\n🏷️ Creating semi-supervised split ({labeled_ratio*100:.0f}% labeled)...")
    y_train_ssl, labeled_mask = create_semi_supervised_split(
        y_train, 
        labeled_ratio=labeled_ratio
    )
    n_labeled = labeled_mask.sum()
    n_unlabeled = (~labeled_mask).sum()
    print(f"  Labeled samples: {n_labeled}")
    print(f"  Unlabeled samples: {n_unlabeled}")
    
    # Train and evaluate each algorithm
    print("\n🤖 Training Semi-Supervised Models:")
    results = {}
    
    for algo in algorithms:
        print(f"\n  Training {algo}...")
        try:
            clf, result = train_and_evaluate(
                X_train, y_train_ssl,
                X_test, y_test,
                algorithm=algo,
                class_names=CLASS_NAMES
            )
            
            # Save the model
            model_path = os.path.join(output_dir, f"{algo}_model.joblib")
            clf.save(model_path)
            print(f"  ✓ Saved model to {model_path}")
            
            results[algo] = {
                'train_accuracy': float(result['train_accuracy']),
                'test_accuracy': float(result['test_accuracy']),
                'n_labeled': int(result['n_labeled']),
                'n_unlabeled': int(result['n_unlabeled'])
            }
            
        except Exception as e:
            print(f"  ✗ Error training {algo}: {e}")
            results[algo] = {'error': str(e)}
    
    # Save training results
    results_path = os.path.join(output_dir, 'training_results.json')
    
    # Add metadata
    results['metadata'] = {
        'labeled_ratio': labeled_ratio,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'feature_dim': X_train.shape[1],
        'n_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📝 Saved training results to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 Training Summary:")
    print("-" * 60)
    print(f"{'Algorithm':<20} {'Train Acc':>12} {'Test Acc':>12}")
    print("-" * 60)
    
    for algo in algorithms:
        if 'error' not in results.get(algo, {}):
            train_acc = results[algo]['train_accuracy']
            test_acc = results[algo]['test_accuracy']
            print(f"{algo:<20} {train_acc:>11.2%} {test_acc:>11.2%}")
        else:
            print(f"{algo:<20} {'ERROR':>12} {'ERROR':>12}")
    
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train semi-supervised models for grocery classification"
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='DS2GROCERIES',
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
        default=0.1,
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
    
    args = parser.parse_args()
    
    train_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        features_dir=args.features_dir,
        labeled_ratio=args.labeled_ratio,
        max_samples_per_class=args.max_samples,
        algorithms=args.algorithms,
        device=args.device
    )


if __name__ == "__main__":
    main()
