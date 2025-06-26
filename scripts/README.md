# DiffusionDrive Scripts

This directory contains organized scripts for training and evaluating DiffusionDrive models.

## Directory Structure

```
scripts/
├── training/           # Training scripts
├── evaluation/         # Evaluation scripts
├── utils/             # Shared utilities
└── data/              # Data preparation scripts
```

## Usage Examples

### Training

1. **Single training run:**
```bash
./scripts/training/train.sh --name my_experiment --epochs 100 --batch-size 32
```

2. **Batch size sweep:**
```bash
./scripts/training/batch_experiments.sh --batch-sizes "32,64,128" --epochs 300
```

### Evaluation

1. **Evaluate single checkpoint:**
```bash
./scripts/evaluation/eval.sh --checkpoint path/to/checkpoint.ckpt
```

2. **Evaluate all checkpoints:**
```bash
./scripts/evaluation/eval_all_checkpoints.sh --dir navsim_workspace/
```

## Script Parameters

### train.sh
- `--name`: Experiment name (required)
- `--epochs`: Maximum epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--workers`: Number of workers (default: 8)
- `--gpus`: GPU devices (default: 0,1,2,3,4,5,6,7)
- `--config`: Config name (default: default_training)
- `--agent`: Agent type (default: diffusiondrive_agent)

### eval.sh
- `--checkpoint`: Path to checkpoint file (required)
- `--name`: Experiment name (auto-generated if not provided)
- `--agent`: Agent type (default: diffusiondrive_agent)

## Environment Requirements

Before running any scripts, ensure these environment variables are set:
- `NAVSIM_DEVKIT_ROOT`: Root directory of the repository
- `NAVSIM_EXP_ROOT`: Directory for experiment outputs

## Logging

All scripts automatically create timestamped log files in the `logs/` directory.