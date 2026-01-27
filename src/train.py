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
import pickle
from torch.utils.data import Subset, DataLoader
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    GroceryDataset, 
    create_semi_supervised_split,
    CLASS_NAMES,
    create_semi_supervised_split,
    CLASS_NAMES,
    get_dataset_stats,
    get_training_transform,
    get_image_transform
)
from src.feature_extraction import FeatureExtractor, extract_dataset_features
from src.semi_supervised import (
    SemiSupervisedClassifier,
    train_and_evaluate,
    compare_algorithms
)


def train_models(
    data_dir: str = str(Path(__file__).parent.parent / "DS2GROCERIES"),
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
        device: Device for feature extraction (cuda/cpu/dml)
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
    print("\nDataset Statistics:")
    stats = get_dataset_stats(data_dir)
    for split, split_stats in stats['splits'].items():
        print(f"  {split}: {split_stats['total']} images")
    
    # Load or extract features
    print("\nFeature Extraction & Fine-Tuning:")
    extractor = FeatureExtractor(device=device)
    
    # 1. Prepare Training Data
    print("\n  Preparing training data...")
    # Standard dataset for feature extraction (no augmentation)
    # We use this to extract features for the UNLABELED data and the TEST data.
    # We don't want to augment these because we want the model to see the "true" images.
    train_dataset_full = GroceryDataset(
        data_dir, 
        split='train',
        transform=get_image_transform(),
        max_samples_per_class=max_samples_per_class
    )
    
    # Augmented dataset for fine-tuning
    # We use this ONLY for the small set of LABELED data during the fine-tuning phase.
    # Augmentation helps the model learn robust features from limited data.
    train_dataset_aug = GroceryDataset(
        data_dir, 
        split='train',
        transform=get_training_transform(),
        max_samples_per_class=max_samples_per_class
    )
    
    # 2. Create Semi-Supervised Split
    # This is the crucial part of our experiment!
    # We artificially hide labels for most of our data to simulate a real-world scenario
    # where labeling data is expensive and time-consuming.
    # We do this BEFORE feature extraction now, because we need to know
    # which samples are labeled to fine-tune on them.
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
    
    # 3. Fine-Tune on Labeled Data
    # To get the best possible features, we fine-tune the ResNet50 backbone
    # using ONLY the small set of labeled data we have.
    # This adapts the general-purpose ImageNet features to our specific grocery domain.
    print("\nFine-tuning ResNet50 on labeled subset...")
    labeled_indices = np.where(labeled_mask)[0]
    labeled_subset = Subset(train_dataset_aug, labeled_indices)
    
    labeled_loader = DataLoader(
        labeled_subset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    # Fine-tune the extractor
    # This updates the weights of the ResNet50 model to better recognize our grocery items.
    # Even though we only have a few labeled samples, this step significantly improves
    # the quality of the features we extract later.
    extractor.fine_tune(
        labeled_loader,
        epochs=10,  # Reduced to prevent overfitting on smaller dataset
        learning_rate=1e-4  # Standard learning rate
    )
    
    # Save fine-tuned feature extractor
    extractor_path = os.path.join(output_dir, "feature_extractor.pth")
    extractor.save(extractor_path)
    print(f"  Saved fine-tuned feature extractor to {extractor_path}")
    
    # 4. Extract Features (using fine-tuned model)
    # Now that we have a domain-adapted feature extractor, we convert all our images
    # (both labeled and unlabeled) into compact feature vectors.
    # These vectors will be the input for our semi-supervised algorithms.
    print("\n  Extracting features with fine-tuned model...")
    
    # Train features
    train_cache = os.path.join(features_dir, f'train_features_finetuned_{labeled_ratio}.npz')
    # We force recompute because the model has changed (fine-tuned)
    # and we don't want to load old non-finetuned features.
    # This step converts all 100% of our training images (labeled + unlabeled)
    # into 2048-dimensional vectors.
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
    # Force recompute to use the fine-tuned model
    X_test, y_test = extractor.extract_and_cache(
        test_dataset, 
        test_cache,
        force_recompute=True
    )
    print(f"  Test features shape: {X_test.shape}")
    
    # 3. Train Semi-Supervised Models
    print("\nTraining Semi-Supervised Models:")
    # This dictionary will store the performance metrics for each algorithm we try.
    results = {}
    
    # Iterate through each algorithm name in our list (e.g., ['label_propagation', ...])
    for algo in algorithms:
        print(f"\n  Training {algo}...")
        try:
            # train_and_evaluate is our custom function that handles the entire training process
            # for a single algorithm. It returns the trained model and a dictionary of results.
            clf, result = train_and_evaluate(
                X_train, y_train_ssl,  # Training data (features and semi-supervised labels)
                X_test, y_test,    # Test data (features and true labels)
                algorithm=algo,    # Name of the algorithm to use
                class_names=CLASS_NAMES, # List of class names for reporting
                y_train_true=y_train, # Pass ground truth for full training accuracy
                # Algorithm-specific parameters:
                kernel='knn',      # Use k-Nearest Neighbors for graph construction
                n_neighbors=10,    # Increased from 7 for better label propagation
                max_iter=2000      # Increased for better convergence
            )
            
            # Save the trained model to disk using Python's pickle module
            # This allows us to load and use the model later without retraining.
            model_path = os.path.join(output_dir, f'{algo}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(clf, f)
            print(f"  Saved model to {model_path}")
            
            # Store the results in our main dictionary
            # We convert numpy types to standard Python types (float, int) 
            # so that they can be saved to JSON later.
            results[algo] = {
                'train_accuracy': float(result['train_accuracy']),
                'train_accuracy_full': float(result['train_accuracy_full']) if result['train_accuracy_full'] is not None else None,
                'test_accuracy': float(result['test_accuracy']),
                'f1_score': float(result['f1_score']),
                'recall_score': float(result['recall_score']),
                'precision_score': float(result['precision_score']),
                'n_labeled': int(result['n_labeled']),
                'n_unlabeled': int(result['n_unlabeled'])
            }
            
        except Exception as e:
            print(f"  Error training {algo}: {e}")
            results[algo] = {'error': str(e)}
    
    # Save training results
    results_path = os.path.join(output_dir, 'training_results.json')
    
    # 4. Save Results
    # We add some metadata to the results so we remember how this experiment was run.
    results['metadata'] = {
        'labeled_ratio': labeled_ratio,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'feature_dim': X_train.shape[1],
        'n_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES
    }
    
    # Save the results dictionary to a JSON file.
    # JSON is a standard text-based format for storing data structures.
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved training results to {results_path}")
    
    # Print a nice summary table to the console
    print("\n" + "=" * 80)
    print("Training Summary:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Labeled Acc':>12} {'Full Train Acc':>15} {'Test Acc':>10} {'F1 Score':>10} {'Recall':>10} {'Precision':>10}")
    print("-" * 100)
    
    for algo in algorithms:
        if 'error' not in results.get(algo, {}):
            train_acc = results[algo]['train_accuracy']
            train_acc_full = results[algo].get('train_accuracy_full', 0)
            test_acc = results[algo]['test_accuracy']
            f1 = results[algo].get('f1_score', 0)
            recall = results[algo].get('recall_score', 0)
            precision = results[algo].get('precision_score', 0)
            
            full_acc_str = f"{train_acc_full:>14.2%}" if train_acc_full is not None else "N/A"
            
            print(f"{algo:<20} {train_acc:>11.2%} {full_acc_str} {test_acc:>9.2%} {f1:>9.2f} {recall:>9.2f} {precision:>9.2f}")
        else:
            print(f"{algo:<20} {'ERROR':>12} {'ERROR':>15} {'ERROR':>10} {'ERROR':>10} {'ERROR':>10} {'ERROR':>10}")
    
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train semi-supervised models for grocery classification"
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
        default=0.2,  # Increased from 0.1 for better accuracy
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
