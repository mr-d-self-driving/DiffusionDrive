#!/bin/bash

find navsim_workspace/ -type f -iname "*.ckpt" -print0 | while IFS= read -r -d '' CKPT; do

    ckptname=$(basename "$CKPT" .ckpt)
    [[ $CKPT =~ /(training_[^/]+) ]]
    experiment_name="${BASH_REMATCH[1]}"
    eval_name="${experiment_name/#training_/eval_}"
    EXP_NAME="${eval_name}_${ckptname}"

    # If "navsim_workspace/exp/$EXP_NAME" exists, skip this checkpoint
    if [[ -d "navsim_workspace/exp/$EXP_NAME" ]]; then
        echo "Skipping existing experiment: $EXP_NAME"
        continue
    fi

    # EXP_NAME="diffusiondrive_agent_pretrained"
    start_time=$(date +%s)
    start_datetime=$(date +"%Y-%m-%d %H:%M:%S")
    LOG_FILE="logs/evaluation_log_$(date +"%Y%m%d_%H%M%S").log" # Create a unique log file name

    echo "Start Time: $start_datetime" | tee -a "$LOG_FILE"
    echo "Checkpoint: $CKPT" | tee -a "$LOG_FILE"
    # Run the evaluation script with the found checkpoint
    python3 -u "$NAVSIM_DEVKIT_ROOT"/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=diffusiondrive_agent \
        worker=ray_distributed \
        agent.checkpoint_path=\'"$CKPT"\' \
        experiment_name=\'"$EXP_NAME"\' 2>&1 | tee -a "$LOG_FILE"

    # Record the finish time
    finish_time=$(date +%s)
    finish_datetime=$(date +"%Y-%m-%d %H:%M:%S")

    echo "Finish Time: $finish_datetime" | tee -a "$LOG_FILE"
    # Calculate the duration
    duration=$((finish_time - start_time))

    # Convert duration to a more readable format (optional)
    minutes=$((duration / 60))
    seconds=$((duration % 60))

    echo "Script finished." | tee -a "$LOG_FILE"
    echo "Total time taken: $duration seconds ($minutes minutes and $seconds seconds)." | tee -a "$LOG_FILE"
done
