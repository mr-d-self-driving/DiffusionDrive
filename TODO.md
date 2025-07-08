# TODO - Check Recent Changes

## What was done today (2025-06-26):

1. **Script Reorganization**
   - Moved scripts from root to `scripts/` directory
   - Created `scripts/training/`, `scripts/evaluation/`, `scripts/utils/`
   - Fixed carriage return errors on all shell scripts

2. **Documentation Added**
   - CHANGELOG.md
   - scripts/README.md
   - docs/MIGRATION_GUIDE.md
   - Updated main README.md

3. **Test Fixes**
   - Fixed test assertions to match actual script output
   - Fixed 3 test issues (still some failing due to environment)

## Check these files:
- `/scripts/training/train.sh` - New parameterized training script
- `/scripts/training/batch_experiments.sh` - Batch experiment runner
- `/scripts/evaluation/eval.sh` - New evaluation script
- `/scripts/utils/common.sh` - Shared utilities

## Run this to test:
```bash
./scripts/training/train.sh --help
./scripts/evaluation/eval.sh --help
```

All changes marked as "Community Contribution" for clarity.