# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-06-26 - Community Contribution

**Note: The following changes were contributed by a community member and are not from the original DiffusionDrive authors.**

### Added
- Organized script structure following software engineering best practices
- Reusable shell script components in `scripts/utils/common.sh`
- Parameterized training script with CLI arguments
- Batch experiment runner for automated hyperparameter sweeps
- Comprehensive evaluation scripts with automatic checkpoint discovery
- Script documentation in `scripts/README.md`

### Changed
- **BREAKING**: Reorganized all scripts into structured directories:
  - Training scripts moved to `scripts/training/`
  - Evaluation scripts moved to `scripts/evaluation/`
  - Shared utilities in `scripts/utils/`
- Replaced date-based script naming with parameterized scripts
- Unified logging and error handling across all scripts

### Deprecated
- Date-based script naming convention (e.g., `train_bs32_20250617.sh`)
- Hardcoded parameters in individual script files

### Removed
- Root directory script clutter (8 training scripts)
- Duplicate code across multiple scripts

### Migration Guide
Old scripts have been moved to `archive/legacy/`. To migrate:

**Before:**
```bash
./train_bs32_20250617.sh
```

**After:**
```bash
./scripts/training/train.sh --name my_experiment --batch-size 32 --epochs 1000
```

## [0.1.0] - 2025-06-09

### Added
- Initial release of DiffusionDrive
- TransFuser backbone with diffusion-based trajectory planning
- NAVSIM integration for autonomous driving simulation
- Basic training and evaluation scripts