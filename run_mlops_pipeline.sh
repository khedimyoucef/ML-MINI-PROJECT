#!/bin/bash

# MLOps Pipeline Automation Script
# Runs the full lifecycle: Data Versioning -> Multi-Algo Training -> Registry -> Promotion -> Monitoring

echo "========================================================"
echo "🚀 Starting Full MLOps Pipeline Automation"
echo "========================================================"

# 1. Data Versioning
# 1. Data Versioning (Dynamic Versioning)
echo -e "\n📦 Step 1: Data Versioning"
VERSION="v_$(date +%Y%m%d_%H%M%S)"
echo "Creating dataset artifact version $VERSION..."
python src/data_versioning.py --version "$VERSION" --description "Automated pipeline run $(date)"
# Note: No uploads by default (saves data)

# 2. Training Extra Algorithms (To ensure we have results for all)
echo -e "\n🧪 Step 2: Training Multiple Algorithms"

# A. Label Propagation (Baseline)
echo "Training Label Propagation..."
python src/train_with_wandb.py \
    --algorithms label_propagation \
    --labeled-ratio 0.2 \
    --run-name "manual-label-prop" \
    --fine-tune-epochs 5

# B. Self Training (Complex)
echo "Training Self Training..."
python src/train_with_wandb.py \
    --algorithms self_training \
    --labeled-ratio 0.2 \
    --run-name "manual-self-training" \
    --fine-tune-epochs 5

# 3. Model Registry
echo -e "\n®️ Step 3: Model Registry"
# Registering all models found in the models/ directory
# This picks up the ones we just trained + the sweep ones
python src/model_registry.py register-all --models-dir models
# Note: By default, this does NOT upload the files, just metadata.

# 4. Model Promotion (Dynamic Selection)
echo -e "\n🏆 Step 4: Model Promotion"
# Detect best model from training results
BEST_MODEL=$(python -c "import json; r=json.load(open('models/training_results.json')); print(max([k for k in r if 'error' not in r[k]], key=lambda x: r[x].get('test_accuracy', 0)))")
echo "Detected best model: $BEST_MODEL"

if [ -f "models/${BEST_MODEL}_model.pkl" ]; then
    echo "Promoting $BEST_MODEL to production..."
    python src/model_registry.py promote "${BEST_MODEL}-classifier" --to-alias production
else
    echo "Best model file found, using fallback..."
    python src/model_registry.py promote label-spreading-classifier --to-alias production
fi

# 5. Production Monitoring
echo -e "\n📈 Step 5: Production Monitoring Simulation"
MODEL="models/${BEST_MODEL}_model.pkl"

if [ ! -f "$MODEL" ]; then
    MODEL="models/label_spreading_model.pkl"
fi

echo "Simulating inference using $MODEL..."
python src/inference_monitor.py \
    --model-path $MODEL \
    --extractor-path models/feature_extractor.pth \
    --n-samples 50

echo "========================================================"
echo "✅ Pipeline Completed!"
echo "Go to your W&B Dashboard to take screenshots."
echo "========================================================"
