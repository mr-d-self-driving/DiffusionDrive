#!/bin/bash

# Test script for common.sh utilities

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../test_framework.sh"
source "$SCRIPT_DIR/../../../scripts/utils/common.sh"

echo "Testing common.sh utilities..."
echo "============================="

# Test setup_logging function
test_start "setup_logging creates log directory and file"
TEST_DIR=$(create_test_dir)
LOG_DIR="$TEST_DIR/logs"
setup_logging "test_script"
assert_dir_exists "$LOG_DIR"
assert_not_empty "$LOG_FILE"
assert_contains "$LOG_FILE" "test_script"
test_end
cleanup_test_dir "$TEST_DIR"

# Test log_start and log_finish functions
test_start "log_start and log_finish work correctly"
TEST_DIR=$(create_test_dir)
LOG_DIR="$TEST_DIR/logs"
setup_logging "test_timing"
log_start "Test Task"
sleep 1  # Simulate some work
log_finish "Test Task"
assert_file_exists "$LOG_FILE"
# Check that log file contains expected content
LOG_CONTENT=$(cat "$LOG_FILE")
assert_contains "$LOG_CONTENT" "Starting: Test Task"
assert_contains "$LOG_CONTENT" "Finished: Test Task"
assert_contains "$LOG_CONTENT" "Duration:"
test_end
cleanup_test_dir "$TEST_DIR"

# Test setup_environment with missing variables
test_start "setup_environment fails with missing NAVSIM_DEVKIT_ROOT"
TEST_DIR=$(create_test_dir)
LOG_DIR="$TEST_DIR/logs"
setup_logging "test_env"
unset NAVSIM_DEVKIT_ROOT
unset NAVSIM_EXP_ROOT
assert_failure "setup_environment" "Should fail without NAVSIM_DEVKIT_ROOT"
test_end
cleanup_test_dir "$TEST_DIR"

# Test setup_environment with valid variables
test_start "setup_environment succeeds with valid variables"
TEST_DIR=$(create_test_dir)
LOG_DIR="$TEST_DIR/logs"
setup_logging "test_env"
export NAVSIM_DEVKIT_ROOT="/test/devkit"
export NAVSIM_EXP_ROOT="/test/exp"
assert_success "setup_environment" "Should succeed with valid environment"
LOG_CONTENT=$(cat "$LOG_FILE")
assert_contains "$LOG_CONTENT" "NAVSIM_DEVKIT_ROOT: /test/devkit"
assert_contains "$LOG_CONTENT" "NAVSIM_EXP_ROOT: /test/exp"
test_end
cleanup_test_dir "$TEST_DIR"

# Test setup_cuda function
test_start "setup_cuda sets CUDA_VISIBLE_DEVICES"
TEST_DIR=$(create_test_dir)
LOG_DIR="$TEST_DIR/logs"
setup_logging "test_cuda"
setup_cuda "0,1"
assert_equals "0,1" "$CUDA_VISIBLE_DEVICES"
LOG_CONTENT=$(cat "$LOG_FILE")
assert_contains "$LOG_CONTENT" "Using GPUs: 0,1"
test_end
cleanup_test_dir "$TEST_DIR"

# Test build_training_args function
test_start "build_training_args creates correct arguments array"
setup_mock_env
build_training_args "test_agent" "test_exp" 50 64 16
assert_not_empty "${TRAINING_ARGS[@]}"
# Check specific arguments
ARGS_STRING="${TRAINING_ARGS[*]}"
assert_contains "$ARGS_STRING" "agent=test_agent"
assert_contains "$ARGS_STRING" "experiment_name=test_exp"
assert_contains "$ARGS_STRING" "trainer.params.max_epochs=50"
assert_contains "$ARGS_STRING" "dataloader.params.batch_size=64"
assert_contains "$ARGS_STRING" "dataloader.params.num_workers=16"
test_end

# Test build_evaluation_args function
test_start "build_evaluation_args creates correct arguments array"
setup_mock_env
build_evaluation_args "/path/to/checkpoint.ckpt" "eval_exp" "test_agent"
assert_not_empty "${EVAL_ARGS[@]}"
# Check specific arguments
ARGS_STRING="${EVAL_ARGS[*]}"
assert_contains "$ARGS_STRING" "agent=test_agent"
assert_contains "$ARGS_STRING" "experiment_name='eval_exp'"
assert_contains "$ARGS_STRING" "agent.checkpoint_path='/path/to/checkpoint.ckpt'"
assert_contains "$ARGS_STRING" "worker=ray_distributed"
test_end

# Print test summary
test_summary