#!/bin/bash
# cache dataset for training
python3 navsim/planning/script/run_dataset_caching.py agent=diffusiondrive_agent experiment_name=training_diffusiondrive_agent train_test_split=navtrain

# cache dataset for evaluation
python3 navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path="$NAVSIM_EXP_ROOT"/metric_cache
