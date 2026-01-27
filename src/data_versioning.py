"""
Data Versioning Module with Weights & Biases

This module provides dataset versioning and tracking using W&B Artifacts.
It tracks dataset statistics, class distributions, and sample images
while keeping the artifact size manageable (no full dataset upload).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import random
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    GroceryDataset,
    CLASS_NAMES,
    get_dataset_stats,
    get_image_transform
)
from src.wandb_config import (
    init_wandb,
    finish_wandb,
    log_images,
    WANDB_PROJECT,
    WANDB_ENTITY,
    DEFAULT_CONFIG
)


def create_dataset_artifact(
    data_dir: str,
    artifact_name: str = "grocery-dataset",
    version: str = "v1",
    description: str = None,
    include_samples: bool = True,
    samples_per_class: int = 3,
    run_name: str = None
) -> wandb.Artifact:
    """
    Create a W&B artifact for the dataset with metadata and sample images.
    
    This logs dataset statistics and a small subset of sample images,
    not the entire dataset (to keep artifact size manageable).
    
    Args:
        data_dir: Path to the dataset root
        artifact_name: Name for the artifact
        version: Version tag
        description: Optional description
        include_samples: Whether to include sample images
        samples_per_class: Number of samples per class to include
        run_name: Optional run name
        
    Returns:
        W&B Artifact object
    """
    # Initialize W&B run
    run = init_wandb(
        run_name=run_name or f"data-versioning-{version}",
        tags=["data-versioning", "dataset"],
        notes=description or f"Dataset artifact {artifact_name} {version}",
        job_type="data-versioning"
    )
    
    try:
        print("=" * 60)
        print("Dataset Versioning with W&B")
        print("=" * 60)
        print(f"Data directory: {data_dir}")
        print(f"Artifact name: {artifact_name}")
        print(f"Version: {version}")
        
        # Get dataset statistics
        print("\nCollecting dataset statistics...")
        stats = get_dataset_stats(data_dir)
        
        # Prepare metadata
        metadata = {
            "name": artifact_name,
            "version": version,
            "source_path": str(data_dir),
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "splits": {},
        }
        
        # Process each split
        for split in ['train', 'test', 'val']:
            split_path = Path(data_dir) / split
            if split_path.exists():
                split_stats = stats['splits'].get(split, {})
                metadata["splits"][split] = {
                    "total_samples": split_stats.get('total', 0),
                    "samples_per_class": split_stats.get('per_class', {}),
                }
                print(f"  {split}: {split_stats.get('total', 0)} samples")
        
        # Calculate total samples
        metadata["total_samples"] = sum(
            s.get("total_samples", 0) 
            for s in metadata["splits"].values()
        )
        
        # Log dataset statistics as metrics
        wandb.log({
            "dataset/total_samples": metadata["total_samples"],
            "dataset/num_classes": metadata["num_classes"],
            "dataset/train_samples": metadata["splits"].get("train", {}).get("total_samples", 0),
            "dataset/test_samples": metadata["splits"].get("test", {}).get("total_samples", 0),
            "dataset/val_samples": metadata["splits"].get("val", {}).get("total_samples", 0),
        })
        
        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=description or f"Grocery classification dataset with {len(CLASS_NAMES)} classes",
            metadata=metadata
        )
        
        # Add metadata JSON
        metadata_path = "/tmp/dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        artifact.add_file(metadata_path, name="metadata.json")
        
        # Collect and add sample images
        if include_samples:
            print(f"\nCollecting {samples_per_class} sample images per class...")
            sample_images_data = []
            
            for split in ['train', 'test']:
                split_path = Path(data_dir) / split
                if not split_path.exists():
                    continue
                    
                for class_name in CLASS_NAMES:
                    class_path = split_path / class_name
                    if not class_path.exists():
                        continue
                    
                    # Get sample images
                    images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))
                    if images:
                        samples = random.sample(images, min(samples_per_class, len(images)))
                        for img_path in samples:
                            artifact.add_file(
                                str(img_path), 
                                name=f"samples/{split}/{class_name}/{img_path.name}"
                            )
                            sample_images_data.append({
                                "path": str(img_path),
                                "class": class_name,
                                "split": split
                            })
            
            print(f"  Added {len(sample_images_data)} sample images")
            
            # Log sample images to W&B
            sample_table = wandb.Table(columns=["image", "class", "split"])
            for sample in sample_images_data[:20]:  # Limit table size
                try:
                    sample_table.add_data(
                        wandb.Image(sample["path"]),
                        sample["class"],
                        sample["split"]
                    )
                except Exception as e:
                    print(f"  Warning: Could not add image {sample['path']}: {e}")
            
            wandb.log({"dataset_samples": sample_table})
        
        # Create class distribution visualization
        print("\nCreating class distribution visualization...")
        train_stats = metadata["splits"].get("train", {}).get("samples_per_class", {})
        if train_stats:
            class_data = [[name, count] for name, count in train_stats.items()]
            class_table = wandb.Table(data=class_data, columns=["class", "count"])
            wandb.log({
                "class_distribution": wandb.plot.bar(
                    class_table, "class", "count",
                    title="Training Set Class Distribution"
                )
            })
        
        # Log the artifact
        print("\nLogging artifact to W&B...")
        wandb.log_artifact(artifact, aliases=["latest", version])
        
        print(f"\n✓ Dataset artifact created successfully!")
        print(f"  Artifact: {artifact_name}:{version}")
        print(f"  Total samples: {metadata['total_samples']}")
        print(f"  W&B URL: {wandb.run.url}")
        
        return artifact
        
    finally:
        finish_wandb()


def load_dataset_artifact(
    artifact_name: str = "grocery-dataset",
    version: str = "latest",
    download_dir: str = None
) -> Dict[str, Any]:
    """
    Load a dataset artifact from W&B.
    
    Args:
        artifact_name: Name of the artifact
        version: Version to load (default: latest)
        download_dir: Optional directory to download files to
        
    Returns:
        Dictionary with artifact metadata
    """
    # Initialize W&B in offline mode if not already initialized
    if wandb.run is None:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="data-loading",
            mode="online"
        )
    
    try:
        # Get artifact
        artifact_path = f"{WANDB_ENTITY or ''}/{WANDB_PROJECT}/{artifact_name}:{version}"
        if not WANDB_ENTITY:
            artifact_path = f"{artifact_name}:{version}"
            
        artifact = wandb.use_artifact(artifact_path)
        
        # Download if requested
        if download_dir:
            artifact_dir = artifact.download(download_dir)
            print(f"Downloaded artifact to: {artifact_dir}")
        
        # Return metadata
        return {
            "name": artifact.name,
            "version": artifact.version,
            "metadata": artifact.metadata,
            "created_at": str(artifact.created_at) if artifact.created_at else None,
        }
        
    finally:
        if wandb.run:
            wandb.finish()


def compare_dataset_versions(
    artifact_name: str = "grocery-dataset",
    version1: str = "v1",
    version2: str = "v2"
) -> Dict[str, Any]:
    """
    Compare two versions of a dataset artifact.
    
    Args:
        artifact_name: Name of the artifact
        version1: First version to compare
        version2: Second version to compare
        
    Returns:
        Dictionary with comparison results
    """
    v1_info = load_dataset_artifact(artifact_name, version1)
    v2_info = load_dataset_artifact(artifact_name, version2)
    
    comparison = {
        "version1": version1,
        "version2": version2,
        "total_samples_diff": (
            v2_info["metadata"].get("total_samples", 0) - 
            v1_info["metadata"].get("total_samples", 0)
        ),
        "splits_comparison": {}
    }
    
    for split in ['train', 'test', 'val']:
        v1_split = v1_info["metadata"].get("splits", {}).get(split, {})
        v2_split = v2_info["metadata"].get("splits", {}).get(split, {})
        
        comparison["splits_comparison"][split] = {
            f"{version1}_samples": v1_split.get("total_samples", 0),
            f"{version2}_samples": v2_split.get("total_samples", 0),
            "diff": v2_split.get("total_samples", 0) - v1_split.get("total_samples", 0)
        }
    
    return comparison


def log_data_quality_metrics(
    data_dir: str,
    run_name: str = None
):
    """
    Log data quality metrics to W&B.
    
    Args:
        data_dir: Path to dataset root
        run_name: Optional run name
    """
    run = init_wandb(
        run_name=run_name or "data-quality-check",
        tags=["data-quality"],
        job_type="data-quality"
    )
    
    try:
        print("Checking data quality...")
        stats = get_dataset_stats(data_dir)
        
        # Calculate quality metrics
        train_stats = stats['splits'].get('train', {}).get('per_class', {})
        if train_stats:
            counts = list(train_stats.values())
            mean_samples = sum(counts) / len(counts)
            min_samples = min(counts)
            max_samples = max(counts)
            
            # Class imbalance ratio
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            
            wandb.log({
                "data_quality/mean_samples_per_class": mean_samples,
                "data_quality/min_samples_per_class": min_samples,
                "data_quality/max_samples_per_class": max_samples,
                "data_quality/class_imbalance_ratio": imbalance_ratio,
                "data_quality/num_classes": len(CLASS_NAMES),
            })
            
            print(f"  Mean samples/class: {mean_samples:.1f}")
            print(f"  Min samples/class: {min_samples}")
            print(f"  Max samples/class: {max_samples}")
            print(f"  Class imbalance ratio: {imbalance_ratio:.2f}")
        
        print(f"\n✓ Data quality metrics logged to W&B")
        print(f"  URL: {wandb.run.url}")
        
    finally:
        finish_wandb()


def main():
    parser = argparse.ArgumentParser(
        description="Dataset versioning with W&B Artifacts"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(Path(__file__).parent.parent / 'DS2GROCERIES'),
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--artifact-name',
        type=str,
        default='grocery-dataset',
        help='Name for the dataset artifact'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1',
        help='Version tag for the artifact'
    )
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help='Description for the artifact'
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=3,
        help='Number of sample images per class to include'
    )
    parser.add_argument(
        '--no-samples',
        action='store_true',
        help='Skip including sample images'
    )
    parser.add_argument(
        '--quality-check',
        action='store_true',
        help='Only run data quality check (no artifact creation)'
    )
    
    args = parser.parse_args()
    
    if args.quality_check:
        log_data_quality_metrics(args.data_dir)
    else:
        create_dataset_artifact(
            data_dir=args.data_dir,
            artifact_name=args.artifact_name,
            version=args.version,
            description=args.description,
            include_samples=not args.no_samples,
            samples_per_class=args.samples_per_class
        )


if __name__ == "__main__":
    main()
