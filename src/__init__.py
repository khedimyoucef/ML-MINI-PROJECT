# Semi-Supervised Grocery Classification
# Source code package

"""
Semi-Supervised Grocery Classification with MLOps

Modules:
- data_utils: Data loading and preprocessing utilities
- feature_extraction: ResNet50 feature extraction
- semi_supervised: SSL algorithms (Label Propagation, Label Spreading, Self-Training)
- train: Original training pipeline
- train_with_wandb: Training with W&B experiment tracking
- wandb_config: W&B configuration and utilities
- data_versioning: Dataset versioning with W&B Artifacts
- model_registry: Model versioning and registry
- inference_monitor: Production inference monitoring
- dataset_cleaner: Dataset cleaning and validation
- recipe_utils: Recipe search utilities
- nutrition_data: Nutritional information utilities
- hf_utils: Hugging Face integration utilities
"""

__version__ = "1.0.0"
