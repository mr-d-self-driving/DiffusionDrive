#!/bin/bash

# Common functions for DiffusionDrive scripts

# Setup logging for any script
setup_logging() {
    local script_name="${1:-script}"
    LOG_DIR="${LOG_DIR:-logs}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/${script_name}_$(date +%Y%m%d_%H%M%S).log"
    export LOG_FILE
    echo "Logging to: $LOG_FILE"
}

# Log start of a task
log_start() {
    local task_name="${1:-Task}"
    export START_TIME=$(date +%s)
    START_DATETIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "=== Starting: $task_name ===" | tee -a "$LOG_FILE"
    echo "Start Time: $START_DATETIME" | tee -a "$LOG_FILE"
}

# Log completion of a task with duration
log_finish() {
    local task_name="${1:-Task}"
    local finish_time=$(date +%s)
    local finish_datetime=$(date +"%Y-%m-%d %H:%M:%S")
    local duration=$((finish_time - START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo "=== Finished: $task_name ===" | tee -a "$LOG_FILE"
    echo "Finish Time: $finish_datetime" | tee -a "$LOG_FILE"
    echo "Duration: ${hours}h ${minutes}m ${seconds}s" | tee -a "$LOG_FILE"
}

# Setup environment variables
setup_environment() {
    export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
    
    # Check required environment variables
    if [ -z "$NAVSIM_DEVKIT_ROOT" ]; then
        echo "ERROR: NAVSIM_DEVKIT_ROOT not set" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    if [ -z "$NAVSIM_EXP_ROOT" ]; then
        echo "ERROR: NAVSIM_EXP_ROOT not set" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    echo "Environment:" | tee -a "$LOG_FILE"
    echo "  NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT" | tee -a "$LOG_FILE"
    echo "  NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT" | tee -a "$LOG_FILE"
}

# Setup CUDA devices
setup_cuda() {
    local devices="${1:-0,1,2,3,4,5,6,7}"
    export CUDA_VISIBLE_DEVICES="$devices"
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
}

# Common training arguments builder
build_training_args() {
    local agent="${1:-diffusiondrive_agent}"
    local experiment_name="$2"
    local max_epochs="${3:-100}"
    local batch_size="${4:-32}"
    local num_workers="${5:-8}"
    
    TRAINING_ARGS=(
        "agent=$agent"
        "experiment_name=$experiment_name"
        "train_test_split=navtrain"
        "split=trainval"
        "trainer.params.max_epochs=$max_epochs"
        "dataloader.params.batch_size=$batch_size"
        "dataloader.params.num_workers=$num_workers"
        "cache_path=${NAVSIM_EXP_ROOT}/training_cache/"
        "use_cache_without_dataset=True"
        "force_cache_computation=False"
    )
    
    export TRAINING_ARGS
}

# Common evaluation arguments builder
build_evaluation_args() {
    local checkpoint_path="$1"
    local experiment_name="$2"
    local agent="${3:-diffusiondrive_agent}"
    
    EVAL_ARGS=(
        "train_test_split=navtest"
        "agent=$agent"
        "worker=ray_distributed"
        "agent.checkpoint_path='$checkpoint_path'"
        "experiment_name='$experiment_name'"
    )
    
    export EVAL_ARGS
}