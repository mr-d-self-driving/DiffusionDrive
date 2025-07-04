# DiffusionDrive Training with Bench2Drive Dataset

This guide provides step-by-step instructions for training DiffusionDrive using the Bench2Drive CARLA dataset.

## Prerequisites

1. **Install Dependencies**:

```bash
# Basic requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning wandb hydra-core omegaconf
pip install numpy opencv-python tqdm h5py scipy laspy
pip install nuscenes-devkit

# Clone and install NavSim (if not already done)
cd /path/to/workspace
git clone https://github.com/autonomousvision/navsim.git
cd navsim
pip install -e .
```

2. **Download Bench2Drive Dataset**:

```bash
# Using Hugging Face CLI
huggingface-cli download --repo-type dataset --resume-download \
    rethinklab/Bench2Drive --local-dir /path/to/Bench2Drive-Dataset
```

## Quick Start

### 1. Setup and Convert Data

```bash
# Set paths
export B2D_ROOT=/path/to/Bench2Drive-Dataset
export NAVSIM_ROOT=/path/to/navsim
export EXP_ROOT=/path/to/experiments/diffusiondrive_b2d

# Run the training setup script
python train_diffusiondrive_bench2drive.py \
    --b2d-root $B2D_ROOT \
    --navsim-root $NAVSIM_ROOT \
    --exp-root $EXP_ROOT \
    --max-scenes 10  # Start with 10 scenes for testing
```

This will:

- Convert Bench2Drive data to NavSim format
- Create training configuration
- Generate launch scripts

### 2. Start Training

```bash
cd $EXP_ROOT
bash run_training.sh
```

## Detailed Steps

### Step 1: Data Conversion

Convert Bench2Drive to NavSim format:

```bash
# Convert training data
python train_diffusiondrive_bench2drive.py \
    --b2d-root $B2D_ROOT \
    --navsim-root $NAVSIM_ROOT \
    --exp-root $EXP_ROOT \
    --convert-only \
    --split train \
    --max-scenes 1000  # Adjust based on your needs

# Convert validation data
python train_diffusiondrive_bench2drive.py \
    --b2d-root $B2D_ROOT \
    --navsim-root $NAVSIM_ROOT \
    --exp-root $EXP_ROOT \
    --convert-only \
    --split val \
    --max-scenes 100
```

### Step 2: Configure Training

The script creates `bench2drive_training_config.yaml`. Key parameters to adjust:

```yaml
# GPU and batch settings
trainer:
  params:
    devices: 4  # Number of GPUs
    
dataloader:
  params:
    batch_size: 32  # Per GPU batch size
    num_workers: 8
    
# Model settings
agent:
  config:
    trajectory_sampling:
      time_horizon: 4.0  # Planning horizon in seconds
      interval_length: 0.5
    use_lidar: true
    use_camera: true
    
# Training settings
optimizer:
  lr: 6e-4  # Learning rate
  
trainer:
  params:
    max_epochs: 50
```

### Step 3: Run Training

```bash
# Set environment variables
export NAVSIM_DEVKIT_ROOT=$NAVSIM_ROOT
export NAVSIM_EXP_ROOT=$EXP_ROOT
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Select GPUs

# Run training
cd $EXP_ROOT
bash run_training.sh

# Or run with custom parameters
bash run_training.sh \
    dataloader.params.batch_size=16 \
    trainer.params.max_epochs=100 \
    optimizer.lr=1e-4
```

### Step 4: Monitor Training

1. **TensorBoard**:

```bash
tensorboard --logdir $EXP_ROOT/logs
```

2. **WandB** (if configured):

- Check your WandB dashboard
- Metrics tracked: loss, validation metrics, learning rate

3. **Checkpoints**:

- Saved in: `$EXP_ROOT/checkpoints/`
- Best model: `best_model.ckpt`
- Last model: `last.ckpt`

### Step 5: Evaluation

```bash
# Evaluate a checkpoint
bash run_evaluation.sh $EXP_ROOT/checkpoints/best_model.ckpt

# Or evaluate with PDM metrics
python $NAVSIM_ROOT/navsim/planning/script/run_pdm_score.py \
    agent=diffusiondrive_agent \
    agent.checkpoint_path=$EXP_ROOT/checkpoints/best_model.ckpt \
    +config=$EXP_ROOT/bench2drive_training_config.yaml
```

## Training Tips

### 1. **Start Small**

- Begin with 10-100 scenes to verify the pipeline
- Check GPU memory usage and adjust batch size
- Ensure data loading is working correctly

### 2. **Data Caching**

```bash
# First run: compute and cache features
bash run_training.sh force_cache_computation=true

# Subsequent runs: use cached features
bash run_training.sh force_cache_computation=false
```

### 3. **Multi-GPU Training**

- DDP strategy is automatically selected for multi-GPU
- Effective batch size = batch_size × num_gpus
- Scale learning rate accordingly: lr = base_lr × num_gpus

### 4. **Memory Management**

If running out of GPU memory:

```yaml
# Reduce batch size
dataloader.params.batch_size: 16

# Use gradient accumulation
trainer.params.accumulate_grad_batches: 2

# Limit trajectory length
agent.config.trajectory_sampling.time_horizon: 3.0
```

### 5. **Debugging**

```bash
# Run with debugging flags
bash run_training.sh \
    trainer.params.fast_dev_run=true \
    trainer.params.limit_train_batches=10 \
    trainer.params.limit_val_batches=5
```

## Common Issues and Solutions

### 1. **CUDA Out of Memory**

- Reduce batch size
- Use mixed precision training (already enabled)
- Reduce number of future frames predicted

### 2. **Slow Data Loading**

- Increase num_workers
- Use SSD for data storage
- Enable pin_memory
- Pre-cache features

### 3. **Poor Convergence**

- Check data normalization
- Adjust learning rate schedule
- Increase batch size for stable gradients
- Verify coordinate transformations are correct

### 4. **Missing Sensor Data**

The converter handles missing data by:

- Duplicating cameras for missing views
- Creating placeholder data when needed
- Logging warnings for missing modalities

## Advanced Configuration

### Custom Agent Parameters

```yaml
agent:
  config:
    # Architecture
    n_layer: 4
    n_head: 8
    n_embd: 256
    
    # Diffusion settings
    diffusion:
      num_diffusion_iters: 10
      variance_schedule: 'cosine'
      
    # Input encoding
    use_lidar: true
    use_camera: true
    lidar_resolution: [200, 200]  # BEV grid size
    lidar_range: [-32, -32, 32, 32]  # meters
```

### Custom Loss Weights

```yaml
agent:
  config:
    loss_weights:
      trajectory: 1.0
      velocity: 0.1
      acceleration: 0.1
      jerk: 0.01
```

## Next Steps

1. **Baseline Training**: Start with default settings
2. **Hyperparameter Tuning**: Adjust learning rate, batch size
3. **Architecture Changes**: Modify model architecture if needed
4. **Evaluation**: Run comprehensive evaluation on test set
5. **Deployment**: Export model for inference

## Resources

- [DiffusionDrive Paper](https://arxiv.org/abs/diffusiondrive)
- [NavSim Documentation](https://github.com/autonomousvision/navsim)
- [Bench2Drive Paper](https://arxiv.org/abs/2406.03877)

For issues specific to Bench2Drive conversion, check the converter logs in `$EXP_ROOT/conversion_logs/`.
