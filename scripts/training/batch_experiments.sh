#!/bin/bash

# Batch experiments runner for DiffusionDrive
# Usage: ./batch_experiments.sh [options]
#   --batch-sizes "32,64,128,256"    Comma-separated list of batch sizes
#   --epochs N                       Max epochs for all experiments
#   --base-name NAME                 Base experiment name
#   --lr RATE                        Single learning rate for all experiments (optional)
#   --lr-list "1e-4,2e-4,4e-4,8e-4"  Comma-separated learning rates for each batch size

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/common.sh"

# Default values
BATCH_SIZES="32,64,128,256"
MAX_EPOCHS=300
BASE_NAME="batch_sweep"
GPU_DEVICES="0,1,2,3,4,5,6,7"
LEARNING_RATE=""
LEARNING_RATE_LIST=""

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
    --lr)
        LEARNING_RATE="$2"
        shift 2
        ;;
    --lr-list)
        LEARNING_RATE_LIST="$2"
        shift 2
        ;;
    --help)
        echo "Usage: $0 [options]"
        echo "  --batch-sizes SIZES  Comma-separated batch sizes (default: 32,64,128,256)"
        echo "  --epochs N           Max epochs (default: 300)"
        echo "  --base-name NAME     Base experiment name (default: batch_sweep)"
        echo "  --gpus DEVICES       GPU devices (default: 0,1,2,3,4,5,6,7)"
        echo "  --lr RATE            Single learning rate for all experiments (optional)"
        echo "  --lr-list RATES      Comma-separated learning rates for each batch size"
        echo ""
        echo "Note: --lr and --lr-list are mutually exclusive. If --lr-list is provided,"
        echo "      it must have the same number of values as batch sizes."
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

# Validate --lr and --lr-list are not both provided
if [ ! -z "$LEARNING_RATE" ] && [ ! -z "$LEARNING_RATE_LIST" ]; then
    echo "Error: --lr and --lr-list cannot be used together" | tee -a "$LOG_FILE"
    exit 1
fi

# Convert comma-separated lists to arrays
IFS=',' read -ra BATCH_SIZE_ARRAY <<<"$BATCH_SIZES"
if [ ! -z "$LEARNING_RATE_LIST" ]; then
    IFS=',' read -ra LR_ARRAY <<<"$LEARNING_RATE_LIST"
    
    # Validate number of learning rates matches number of batch sizes
    if [ ${#LR_ARRAY[@]} -ne ${#BATCH_SIZE_ARRAY[@]} ]; then
        echo "Error: Number of learning rates (${#LR_ARRAY[@]}) must match number of batch sizes (${#BATCH_SIZE_ARRAY[@]})" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

echo "Batch Experiment Configuration:" | tee -a "$LOG_FILE"
echo "  Base Name: $BASE_NAME" | tee -a "$LOG_FILE"
echo "  Batch Sizes: ${BATCH_SIZE_ARRAY[*]}" | tee -a "$LOG_FILE"
echo "  Max Epochs: $MAX_EPOCHS" | tee -a "$LOG_FILE"
if [ ! -z "$LEARNING_RATE" ]; then
    echo "  Learning Rate: $LEARNING_RATE (for all experiments)" | tee -a "$LOG_FILE"
elif [ ! -z "$LEARNING_RATE_LIST" ]; then
    echo "  Learning Rates: ${LR_ARRAY[*]}" | tee -a "$LOG_FILE"
fi

# Run experiments for each batch size
for i in "${!BATCH_SIZE_ARRAY[@]}"; do
    BATCH_SIZE="${BATCH_SIZE_ARRAY[$i]}"
    
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Starting experiment with batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
    
    # Determine learning rate for this experiment
    if [ ! -z "$LEARNING_RATE_LIST" ]; then
        CURRENT_LR="${LR_ARRAY[$i]}"
        echo "Learning rate: $CURRENT_LR" | tee -a "$LOG_FILE"
    elif [ ! -z "$LEARNING_RATE" ]; then
        CURRENT_LR="$LEARNING_RATE"
        echo "Learning rate: $CURRENT_LR" | tee -a "$LOG_FILE"
    else
        CURRENT_LR=""
    fi
    echo "========================================" | tee -a "$LOG_FILE"

    EXPERIMENT_NAME="${BASE_NAME}_bs${BATCH_SIZE}_ep${MAX_EPOCHS}"

    # Build training command
    TRAIN_CMD=("$SCRIPT_DIR/train.sh"
        "--name" "$EXPERIMENT_NAME"
        "--epochs" "$MAX_EPOCHS"
        "--batch-size" "$BATCH_SIZE"
        "--workers" "16"
        "--gpus" "$GPU_DEVICES"
        "--config" "default_training_w_callbacks")
    
    # Add learning rate if specified
    if [ ! -z "$CURRENT_LR" ]; then
        TRAIN_CMD+=("--lr" "$CURRENT_LR")
    fi
    
    # Run training script
    "${TRAIN_CMD[@]}"

    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "Experiment $EXPERIMENT_NAME completed successfully" | tee -a "$LOG_FILE"
    else
        echo "ERROR: Experiment $EXPERIMENT_NAME failed" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "All batch experiments completed" | tee -a "$LOG_FILE"

