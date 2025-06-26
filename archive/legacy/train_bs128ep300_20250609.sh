MAX_EPOCHS=300
BATCH_SIZE=128
NUM_WORKERS=12
EXP_NAME="default_bs128ep300"

start_time=$(date +%s)
start_datetime=$(date +"%Y-%m-%d %H:%M:%S")
LOG_FILE="logs/training_log_$(date +"%Y%m%d_%H%M%S").log" # Create a unique log file name

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "Use GPU $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "Start Time: $start_datetime" | tee -a "$LOG_FILE"

python3 $NAVSIM_DEVKIT_ROOT/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name="training_diffusiondrive_agent_${EXP_NAME}" \
    train_test_split=navtrain \
    split=trainval \
    trainer.params.max_epochs=$MAX_EPOCHS \
    dataloader.params.batch_size=$BATCH_SIZE \
    dataloader.params.num_workers=$NUM_WORKERS \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
    use_cache_without_dataset=True \
    force_cache_computation=False | tee -a "$LOG_FILE"

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
