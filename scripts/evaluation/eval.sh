#!/bin/bash

# Evaluation script for DiffusionDrive
# Usage: ./eval.sh [options]
#   --checkpoint PATH    Path to checkpoint file (required)
#   --name NAME         Experiment name (optional, auto-generated if not provided)
#   --agent AGENT       Agent type (default: diffusiondrive_agent)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/common.sh"

# Default values
CHECKPOINT_PATH=""
EXPERIMENT_NAME=""
AGENT="diffusiondrive_agent"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --checkpoint PATH    Path to checkpoint file (required)"
            echo "  --name NAME         Experiment name (optional)"
            echo "  --agent AGENT       Agent type (default: diffusiondrive_agent)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

# Auto-generate experiment name if not provided
if [ -z "$EXPERIMENT_NAME" ]; then
    # Extract checkpoint name
    CKPT_NAME=$(basename "$CHECKPOINT_PATH" .ckpt)
    
    # Try to extract training experiment name from path
    if [[ $CHECKPOINT_PATH =~ /(training_[^/]+)/ ]]; then
        TRAINING_NAME="${BASH_REMATCH[1]}"
        EVAL_NAME="${TRAINING_NAME/#training_/eval_}"
        EXPERIMENT_NAME="${EVAL_NAME}_${CKPT_NAME}"
    else
        EXPERIMENT_NAME="eval_${AGENT}_${CKPT_NAME}"
    fi
fi

# Setup
setup_logging "evaluation_${EXPERIMENT_NAME}"
setup_environment

# Log configuration
echo "Evaluation Configuration:" | tee -a "$LOG_FILE"
echo "  Experiment: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "  Agent: $AGENT" | tee -a "$LOG_FILE"
echo "  Checkpoint: $CHECKPOINT_PATH" | tee -a "$LOG_FILE"

# Check if experiment already exists
if [ -d "$NAVSIM_EXP_ROOT/exp/$EXPERIMENT_NAME" ]; then
    echo "WARNING: Experiment directory already exists: $NAVSIM_EXP_ROOT/exp/$EXPERIMENT_NAME" | tee -a "$LOG_FILE"
    echo "Skipping evaluation to avoid overwriting results" | tee -a "$LOG_FILE"
    exit 0
fi

# Build evaluation arguments
build_evaluation_args "$CHECKPOINT_PATH" "$EXPERIMENT_NAME" "$AGENT"

# Start evaluation
log_start "Evaluation"

# Run evaluation
python3 -u "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    "${EVAL_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

# Log completion
log_finish "Evaluation"