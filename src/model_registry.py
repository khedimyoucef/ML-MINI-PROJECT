"""
Model Registry Module with Weights & Biases

This module provides model versioning, registration, and lineage tracking
using W&B Artifacts and Model Registry features.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import pickle
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wandb_config import (
    init_wandb,
    finish_wandb,
    WANDB_PROJECT,
    WANDB_ENTITY,
    DEFAULT_CONFIG
)
from src.data_utils import CLASS_NAMES


def register_model(
    model_path: str,
    model_name: str,
    model_type: str = "semi-supervised-classifier",
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    aliases: Optional[List[str]] = None,
    description: str = None,
    run_name: str = None
) -> wandb.Artifact:
    """
    Register a trained model to W&B Model Registry.
    
    Args:
        model_path: Path to the model file (.pkl or .pth)
        model_name: Name for the model in the registry
        model_type: Type of model
        metrics: Performance metrics
        config: Training configuration
        aliases: List of aliases (e.g., ["latest", "best", "production"])
        description: Model description
        run_name: Optional W&B run name
        
    Returns:
        W&B Artifact
    """
    # Initialize W&B
    run = init_wandb(
        run_name=run_name or f"register-{model_name}",
        config=config,
        tags=["model-registry"],
        notes=description,
        job_type="model-registration"
    )
    
    try:
        print("=" * 60)
        print("Model Registration with W&B")
        print("=" * 60)
        print(f"Model path: {model_path}")
        print(f"Model name: {model_name}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Prepare metadata
        metadata = {
            "model_type": model_type,
            "file_path": model_path,
            "file_size_mb": os.path.getsize(model_path) / (1024 * 1024),
            "class_names": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
        }
        
        if metrics:
            metadata["metrics"] = metrics
            print(f"\nMetrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        if config:
            metadata["config"] = config
        
        # Create artifact
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=description or f"Semi-supervised {model_name} classifier",
            metadata=metadata
        )
        
        # Add model file
        artifact.add_file(model_path)
        
        # Add model info JSON
        info_path = "/tmp/model_info.json"
        with open(info_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        artifact.add_file(info_path, name="model_info.json")
        
        # Log metrics
        if metrics:
            wandb.log({"model/" + k: v for k, v in metrics.items()})
        
        # Log artifact with aliases
        effective_aliases = aliases or ["latest"]
        wandb.log_artifact(artifact, aliases=effective_aliases)
        
        print(f"\n✓ Model registered successfully!")
        print(f"  Artifact: {model_name}")
        print(f"  Aliases: {effective_aliases}")
        print(f"  W&B URL: {wandb.run.url}")
        
        return artifact
        
    finally:
        finish_wandb()


def load_model_from_registry(
    model_name: str,
    version: str = "latest",
    download_dir: str = None
) -> Any:
    """
    Load a model from W&B Model Registry.
    
    Args:
        model_name: Name of the model in the registry
        version: Version or alias to load
        download_dir: Directory to download model to
        
    Returns:
        Loaded model object
    """
    # Initialize W&B
    if wandb.run is None:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="model-loading"
        )
    
    try:
        print(f"Loading model {model_name}:{version} from registry...")
        
        # Get artifact
        artifact_path = f"{model_name}:{version}"
        if WANDB_ENTITY:
            artifact_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_path}"
        
        artifact = wandb.use_artifact(artifact_path)
        
        # Download
        download_path = download_dir or "/tmp/wandb_models"
        artifact_dir = artifact.download(download_path)
        
        # Find and load model file
        for file_name in os.listdir(artifact_dir):
            if file_name.endswith('.pkl'):
                model_path = os.path.join(artifact_dir, file_name)
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✓ Loaded model from {model_path}")
                return model
            elif file_name.endswith('.pth'):
                import torch
                model_path = os.path.join(artifact_dir, file_name)
                model = torch.load(model_path)
                print(f"✓ Loaded model from {model_path}")
                return model
        
        raise FileNotFoundError("No model file found in artifact")
        
    finally:
        if wandb.run:
            wandb.finish()


def list_registered_models() -> List[Dict[str, Any]]:
    """
    List all registered models in the project.
    
    Returns:
        List of model information dictionaries
    """
    api = wandb.Api()
    
    # Get all model artifacts
    project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    artifacts = api.artifacts(type_name="model", name=project_path)
    
    models = []
    for artifact in artifacts:
        models.append({
            "name": artifact.name,
            "version": artifact.version,
            "aliases": artifact.aliases,
            "created_at": str(artifact.created_at),
            "metadata": artifact.metadata
        })
    
    return models


def compare_models(
    model_names: List[str],
    metric: str = "test_accuracy"
) -> Dict[str, Any]:
    """
    Compare multiple registered models by a metric.
    
    Args:
        model_names: List of model names to compare
        metric: Metric to compare by
        
    Returns:
        Comparison results
    """
    api = wandb.Api()
    
    comparison = {
        "metric": metric,
        "models": {}
    }
    
    for name in model_names:
        try:
            artifact_path = f"{WANDB_PROJECT}/{name}:latest"
            if WANDB_ENTITY:
                artifact_path = f"{WANDB_ENTITY}/{artifact_path}"
            
            artifact = api.artifact(artifact_path)
            metrics = artifact.metadata.get("metrics", {})
            
            comparison["models"][name] = {
                "version": artifact.version,
                metric: metrics.get(metric, None),
                "all_metrics": metrics
            }
        except Exception as e:
            comparison["models"][name] = {"error": str(e)}
    
    # Find best model
    best_model = None
    best_value = 0
    for name, data in comparison["models"].items():
        if "error" not in data and data.get(metric):
            if data[metric] > best_value:
                best_value = data[metric]
                best_model = name
    
    comparison["best_model"] = best_model
    comparison["best_value"] = best_value
    
    return comparison


def promote_model(
    model_name: str,
    from_alias: str = "latest",
    to_alias: str = "production"
):
    """
    Promote a model to a new alias (e.g., production).
    
    Args:
        model_name: Name of the model
        from_alias: Current alias/version
        to_alias: New alias to add
    """
    api = wandb.Api()
    
    artifact_path = f"{WANDB_PROJECT}/{model_name}:{from_alias}"
    if WANDB_ENTITY:
        artifact_path = f"{WANDB_ENTITY}/{artifact_path}"
    
    artifact = api.artifact(artifact_path)
    artifact.aliases.append(to_alias)
    artifact.save()
    
    print(f"✓ Model {model_name} promoted to '{to_alias}'")
    print(f"  Current aliases: {artifact.aliases}")


def register_all_models(
    models_dir: str = "models",
    training_results_path: str = None
):
    """
    Register all trained models from a directory.
    
    Args:
        models_dir: Directory containing model files
        training_results_path: Optional path to training results JSON
    """
    models_dir = Path(models_dir)
    
    # Load training results if available
    results = {}
    results_path = training_results_path or models_dir / "training_results.json"
    if Path(results_path).exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    
    # Register each model
    for model_file in models_dir.glob("*_model.pkl"):
        algo_name = model_file.stem.replace("_model", "")
        
        # Get metrics from results
        metrics = {}
        if algo_name in results and "error" not in results[algo_name]:
            metrics = {
                "test_accuracy": results[algo_name].get("test_accuracy"),
                "f1_score": results[algo_name].get("f1_score"),
                "precision_score": results[algo_name].get("precision_score"),
                "recall_score": results[algo_name].get("recall_score"),
            }
        
        # Get config
        config = results.get("metadata", {})
        
        register_model(
            model_path=str(model_file),
            model_name=f"{algo_name}-classifier",
            metrics=metrics,
            config=config,
            aliases=["latest"],
            description=f"Semi-supervised {algo_name} classifier for grocery images"
        )
        
        print()  # Blank line between registrations


def main():
    parser = argparse.ArgumentParser(
        description="Model Registry operations with W&B"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a model")
    register_parser.add_argument("model_path", help="Path to model file")
    register_parser.add_argument("--name", required=True, help="Model name")
    register_parser.add_argument("--accuracy", type=float, help="Test accuracy")
    register_parser.add_argument("--f1", type=float, help="F1 score")
    register_parser.add_argument("--alias", nargs="+", default=["latest"], help="Aliases")
    register_parser.add_argument("--description", help="Description")
    
    # Register all command
    register_all_parser = subparsers.add_parser("register-all", help="Register all models")
    register_all_parser.add_argument("--models-dir", default="models", help="Models directory")
    register_all_parser.add_argument("--results", help="Training results JSON path")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List registered models")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("models", nargs="+", help="Model names to compare")
    compare_parser.add_argument("--metric", default="test_accuracy", help="Metric to compare")
    
    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote model to alias")
    promote_parser.add_argument("model_name", help="Model name")
    promote_parser.add_argument("--from-alias", default="latest", help="Current alias")
    promote_parser.add_argument("--to-alias", default="production", help="New alias")
    
    args = parser.parse_args()
    
    if args.command == "register":
        metrics = {}
        if args.accuracy:
            metrics["test_accuracy"] = args.accuracy
        if args.f1:
            metrics["f1_score"] = args.f1
        
        register_model(
            model_path=args.model_path,
            model_name=args.name,
            metrics=metrics if metrics else None,
            aliases=args.alias,
            description=args.description
        )
        
    elif args.command == "register-all":
        register_all_models(
            models_dir=args.models_dir,
            training_results_path=args.results
        )
        
    elif args.command == "list":
        models = list_registered_models()
        print("Registered Models:")
        print("-" * 60)
        for model in models:
            print(f"  {model['name']} (v{model['version']})")
            print(f"    Aliases: {model['aliases']}")
            print(f"    Created: {model['created_at']}")
            if "metrics" in model.get("metadata", {}):
                print(f"    Metrics: {model['metadata']['metrics']}")
            print()
            
    elif args.command == "compare":
        comparison = compare_models(args.models, args.metric)
        print(f"Model Comparison by {args.metric}:")
        print("-" * 60)
        for name, data in comparison["models"].items():
            if "error" not in data:
                print(f"  {name}: {data.get(args.metric, 'N/A')}")
            else:
                print(f"  {name}: ERROR - {data['error']}")
        print(f"\nBest: {comparison['best_model']} ({comparison['best_value']:.4f})")
        
    elif args.command == "promote":
        promote_model(
            model_name=args.model_name,
            from_alias=args.from_alias,
            to_alias=args.to_alias
        )
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
