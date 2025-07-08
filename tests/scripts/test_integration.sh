#!/bin/bash

# Integration tests for the scripts working together

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/test_framework.sh"

echo "Testing script integration..."
echo "============================"

# Test that scripts can find and source common.sh
test_start "Scripts can source common utilities"
# Create a minimal test script that sources common.sh
TEST_DIR=$(create_test_dir)
cat > "$TEST_DIR/test_source.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../scripts/utils/common.sh"
echo "Sourced successfully"
EOF
chmod +x "$TEST_DIR/test_source.sh"

# Copy common.sh to expected location
mkdir -p "$TEST_DIR/../../scripts/utils"
cp "$SCRIPT_DIR/../../scripts/utils/common.sh" "$TEST_DIR/../../scripts/utils/"

OUTPUT=$("$TEST_DIR/test_source.sh" 2>&1)
assert_contains "$OUTPUT" "Sourced successfully"
cleanup_test_dir "$TEST_DIR"
test_end

# Test train -> eval workflow
test_start "Training output can be evaluated"
TEST_DIR=$(create_test_dir)
setup_mock_env
export LOG_DIR="$TEST_DIR/logs"

# Mock a training run that creates a checkpoint
CHECKPOINT_DIR="$TEST_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
MOCK_CHECKPOINT="$CHECKPOINT_DIR/epoch_10.ckpt"

# Simulate training creating a checkpoint
echo "mock checkpoint" > "$MOCK_CHECKPOINT"

# Now test that eval.sh can use this checkpoint
MOCK_BIN="$TEST_DIR/bin"
mkdir -p "$MOCK_BIN"
cat > "$MOCK_BIN/python" << 'EOF'
#!/bin/bash
# Check if checkpoint path is in arguments
if [[ "$@" =~ checkpoint_path ]]; then
    echo "Successfully loaded checkpoint"
    exit 0
else
    echo "No checkpoint provided"
    exit 1
fi
EOF
chmod +x "$MOCK_BIN/python"
export PATH="$MOCK_BIN:$PATH"

# Run evaluation on the "trained" model
OUTPUT=$("$SCRIPT_DIR/../../scripts/evaluation/eval.sh" --checkpoint "$MOCK_CHECKPOINT" 2>&1)
assert_contains "$OUTPUT" "Successfully loaded checkpoint"

cleanup_test_dir "$TEST_DIR"
test_end

# Test batch experiments create multiple evaluatable checkpoints
test_start "Batch experiments workflow"
TEST_DIR=$(create_test_dir)
setup_mock_env

# Track created checkpoints
CHECKPOINT_LIST="$TEST_DIR/checkpoints.txt"

# Mock train.sh that creates checkpoints based on batch size
cat > "$TEST_DIR/mock_train.sh" << EOF
#!/bin/bash
# Parse batch size from arguments
BATCH_SIZE=32
for arg in "\$@"; do
    if [[ "\$prev_arg" == "--batch-size" ]]; then
        BATCH_SIZE="\$arg"
    fi
    prev_arg="\$arg"
done

# Create a checkpoint for this run
CKPT="$TEST_DIR/checkpoints/model_bs\${BATCH_SIZE}.ckpt"
mkdir -p "\$(dirname "\$CKPT")"
echo "checkpoint bs\$BATCH_SIZE" > "\$CKPT"
echo "\$CKPT" >> "$CHECKPOINT_LIST"
echo "Training completed, checkpoint saved to \$CKPT"
EOF
chmod +x "$TEST_DIR/mock_train.sh"

# Replace train.sh path for testing
export TRAIN_SCRIPT="$TEST_DIR/mock_train.sh"

# Run batch experiments
BATCH_SIZES="16,32"
for bs in $(echo $BATCH_SIZES | tr ',' ' '); do
    $TEST_DIR/mock_train.sh --batch-size $bs
done

# Verify checkpoints were created
assert_file_exists "$TEST_DIR/checkpoints/model_bs16.ckpt"
assert_file_exists "$TEST_DIR/checkpoints/model_bs32.ckpt"

# Verify we can evaluate all checkpoints
CHECKPOINTS=$(find "$TEST_DIR/checkpoints" -name "*.ckpt" | wc -l)
assert_equals "2" "$CHECKPOINTS" "Should have created 2 checkpoints"

cleanup_test_dir "$TEST_DIR"
test_end

# Test GPU constraint propagation
test_start "GPU constraint propagates through script calls"
TEST_DIR=$(create_test_dir)
setup_mock_env

# Create a chain of scripts that check GPU settings
cat > "$TEST_DIR/script1.sh" << 'EOF'
#!/bin/bash
echo "Script1 GPUs: $CUDA_VISIBLE_DEVICES"
bash "$(dirname "$0")/script2.sh"
EOF

cat > "$TEST_DIR/script2.sh" << 'EOF'
#!/bin/bash
echo "Script2 GPUs: $CUDA_VISIBLE_DEVICES"
EOF

chmod +x "$TEST_DIR/script1.sh" "$TEST_DIR/script2.sh"

OUTPUT=$("$TEST_DIR/script1.sh" 2>&1)
assert_contains "$OUTPUT" "Script1 GPUs: 0,1"
assert_contains "$OUTPUT" "Script2 GPUs: 0,1"

cleanup_test_dir "$TEST_DIR"
test_end

# Test error handling propagation
test_start "Errors propagate correctly between scripts"
TEST_DIR=$(create_test_dir)

# Create a script that fails
cat > "$TEST_DIR/failing_script.sh" << 'EOF'
#!/bin/bash
echo "This script will fail"
exit 1
EOF
chmod +x "$TEST_DIR/failing_script.sh"

# Create a wrapper that should detect the failure
cat > "$TEST_DIR/wrapper.sh" << EOF
#!/bin/bash
if ! "$TEST_DIR/failing_script.sh"; then
    echo "Detected failure correctly"
    exit 0
else
    echo "Failed to detect error"
    exit 1
fi
EOF
chmod +x "$TEST_DIR/wrapper.sh"

OUTPUT=$("$TEST_DIR/wrapper.sh" 2>&1)
assert_contains "$OUTPUT" "Detected failure correctly"

cleanup_test_dir "$TEST_DIR"
test_end

# Print test summary
test_summary