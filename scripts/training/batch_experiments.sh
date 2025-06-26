#!/bin/bash

# Batch experiments runner for DiffusionDrive
# Usage: ./batch_experiments.sh [options]
#   --batch-sizes "32,64,128,256"    Comma-separated list of batch sizes
#   --epochs N                       Max epochs for all experiments
#   --base-name NAME                 Base experiment name

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/common.sh"

# Default values
BATCH_SIZES="32,64,128,256"
MAX_EPOCHS=300
BASE_NAME="batch_sweep"
GPU_DEVICES="0,1,2,3,4,5,6,7"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --batch-sizes)
        BATCH_SIZES="$2"
        shift 2
        ;;
    --epochs)
        MAX_EPOCHS="$2"
        shift 2
        ;;
    --base-name)
        BASE_NAME="$2"
        shift 2
        ;;
    --gpus)
        GPU_DEVICES="$2"
        shift 2
        ;;
    --help)
        echo "Usage: $0 [options]"
        echo "  --batch-sizes SIZES  Comma-separated batch sizes (default: 32,64,128,256)"
        echo "  --epochs N           Max epochs (default: 300)"
        echo "  --base-name NAME     Base experiment name (default: batch_sweep)"
        echo "  --gpus DEVICES       GPU devices (default: 0,1,2,3,4,5,6,7)"
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Setup
setup_logging "batch_experiments_${BASE_NAME}"
setup_environment

# Convert comma-separated list to array
IFS=',' read -ra BATCH_SIZE_ARRAY <<<"$BATCH_SIZES"

echo "Batch Experiment Configuration:" | tee -a "$LOG_FILE"
echo "  Base Name: $BASE_NAME" | tee -a "$LOG_FILE"
echo "  Batch Sizes: ${BATCH_SIZE_ARRAY[*]}" | tee -a "$LOG_FILE"
echo "  Max Epochs: $MAX_EPOCHS" | tee -a "$LOG_FILE"

# Run experiments for each batch size
for BATCH_SIZE in "${BATCH_SIZE_ARRAY[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Starting experiment with batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    EXPERIMENT_NAME="${BASE_NAME}_bs${BATCH_SIZE}_ep${MAX_EPOCHS}"

    # Run training script
    "$SCRIPT_DIR/train.sh" \
        --name "$EXPERIMENT_NAME" \
        --epochs "$MAX_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --workers 16 \
        --gpus "$GPU_DEVICES" \
        --config "default_training_w_callbacks"

    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "Experiment $EXPERIMENT_NAME completed successfully" | tee -a "$LOG_FILE"
    else
        echo "ERROR: Experiment $EXPERIMENT_NAME failed" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "All batch experiments completed" | tee -a "$LOG_FILE"

