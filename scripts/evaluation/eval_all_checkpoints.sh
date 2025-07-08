#!/bin/bash

# Evaluate all checkpoints in a directory
# Usage: ./eval_all_checkpoints.sh [options]
#   --dir PATH          Directory containing checkpoints (default: navsim_workspace/)
#   --pattern PATTERN   File pattern to match (default: *.ckpt)
#   --agent AGENT       Agent type (default: diffusiondrive_agent)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/common.sh"

# Default values
CHECKPOINT_DIR="navsim_workspace/"
FILE_PATTERN="*.ckpt"
AGENT="diffusiondrive_agent"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --pattern)
            FILE_PATTERN="$2"
            shift 2
            ;;
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --dir PATH          Directory containing checkpoints (default: navsim_workspace/)"
            echo "  --pattern PATTERN   File pattern to match (default: *.ckpt)"
            echo "  --agent AGENT       Agent type (default: diffusiondrive_agent)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup
setup_logging "eval_all_checkpoints"
setup_environment

echo "Batch Evaluation Configuration:" | tee -a "$LOG_FILE"
echo "  Directory: $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
echo "  Pattern: $FILE_PATTERN" | tee -a "$LOG_FILE"
echo "  Agent: $AGENT" | tee -a "$LOG_FILE"

# Find all checkpoints
CHECKPOINTS=()
while IFS= read -r -d '' file; do
    CHECKPOINTS+=("$file")
done < <(find "$CHECKPOINT_DIR" -type f -name "$FILE_PATTERN" -print0 | sort -z)

echo "Found ${#CHECKPOINTS[@]} checkpoint(s)" | tee -a "$LOG_FILE"

# Evaluate each checkpoint
for CKPT in "${CHECKPOINTS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Evaluating checkpoint: $CKPT" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # Run evaluation script
    "$SCRIPT_DIR/eval.sh" \
        --checkpoint "$CKPT" \
        --agent "$AGENT"
    
    # Check if evaluation was successful
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully" | tee -a "$LOG_FILE"
    else
        echo "ERROR: Evaluation failed for $CKPT" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "All evaluations completed" | tee -a "$LOG_FILE"