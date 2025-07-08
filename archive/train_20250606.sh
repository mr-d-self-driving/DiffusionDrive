# This is for testing purpose only.
# Please move to root folder before running this script
start_time=$(date +%s)
start_datetime=$(date +"%Y-%m-%d %H:%M:%S")

echo "Start Time: $start_datetime"

HYDRA_FULL_ERROR=1 python3 $NAVSIM_DEVKIT_ROOT/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name=training_diffusiondrive_agent \
    train_test_split=navtrain \
    split=trainval \
    trainer.params.max_epochs=100 \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
    use_cache_without_dataset=True \
    force_cache_computation=False

# Record the finish time
finish_time=$(date +%s)
finish_datetime=$(date +"%Y-%m-%d %H:%M:%S")

echo "Finish Time: $finish_datetime"

# Calculate the duration
duration=$((finish_time - start_time))

# Convert duration to a more readable format (optional)
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "Script finished."
echo "Total time taken: $duration seconds ($minutes minutes and $seconds seconds)."
