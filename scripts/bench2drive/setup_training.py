#!/usr/bin/env python3
"""
Training script for DiffusionDrive with Bench2Drive dataset
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
import subprocess
from typing import List

class DiffusionDriveBench2DriveTrainer:
    """Helper class to set up and run DiffusionDrive training with Bench2Drive data"""
    
    def __init__(self, 
                 bench2drive_root: Path,
                 navsim_root: Path,
                 experiment_root: Path):
        self.b2d_root = Path(bench2drive_root)
        self.navsim_root = Path(navsim_root)
        self.exp_root = Path(experiment_root)
        
        # Create directories
        self.exp_root.mkdir(parents=True, exist_ok=True)
        self.converted_data_root = self.exp_root / 'converted_data'
        self.converted_data_root.mkdir(exist_ok=True)
        
    def convert_dataset(self, split: str = 'train', max_scenes: int = None):
        """Convert Bench2Drive data to NavSim format"""
        
        print(f"Converting Bench2Drive {split} split to NavSim format...")
        
        converter_script = Path(__file__).parent / 'bench2drive_to_navsim_converter.py'
        
        cmd = [
            'python', str(converter_script),
            '--b2d-root', str(self.b2d_root),
            '--output-root', str(self.converted_data_root / split),
            '--split', split
        ]
        
        if max_scenes:
            cmd.extend(['--max-scenes', str(max_scenes)])
        
        subprocess.run(cmd, check=True)
        print(f"Conversion complete for {split} split!")
    
    def get_scene_list(self, split: str) -> List[str]:
        """Get list of converted scenes for a split"""
        
        split_index = self.converted_data_root / split / f'{split}_index.pkl'
        if not split_index.exists():
            print(f"Warning: {split_index} not found. Run conversion first.")
            return []
        
        import pickle
        with open(split_index, 'rb') as f:
            index = pickle.load(f)
        
        # Extract scene names and prepend 'bench2drive_' prefix
        scenes = [f"bench2drive_{name}" for name in index['scenes'].keys()]
        return scenes
    
    def create_training_config(self):
        """Create training configuration for DiffusionDrive"""
        
        train_scenes = self.get_scene_list('train')
        val_scenes = self.get_scene_list('val')
        
        if not train_scenes:
            print("No training scenes found. Please convert data first.")
            return None
        
        config = {
            'defaults': [
                'default_training',
                '_self_'
            ],
            
            # Data paths
            'navsim_log_path': str(self.converted_data_root),
            'sensor_blobs_path': str(self.converted_data_root / 'sensor_blobs'),
            'cache_path': str(self.exp_root / 'training_cache'),
            
            # Data splits
            'train_logs': train_scenes[:800] if len(train_scenes) > 800 else train_scenes,
            'val_logs': val_scenes[:100] if len(val_scenes) > 100 else val_scenes,
            
            # Training parameters
            'dataloader': {
                'params': {
                    'batch_size': 32,  # Adjust based on GPU memory
                    'num_workers': 8,
                    'pin_memory': True,
                    'drop_last': True
                }
            },
            
            'trainer': {
                'params': {
                    'max_epochs': 50,
                    'devices': 1,  # Number of GPUs
                    'accelerator': 'gpu',
                    'strategy': 'ddp' if torch.cuda.device_count() > 1 else 'auto',
                    'precision': '16-mixed',
                    'gradient_clip_val': 1.0,
                    'accumulate_grad_batches': 1,
                    'log_every_n_steps': 50,
                    'val_check_interval': 0.25,  # Validate 4 times per epoch
                    'limit_val_batches': 100,  # Limit validation batches
                }
            },
            
            # Model parameters
            'agent': {
                'config': {
                    'trajectory_sampling': {
                        'time_horizon': 4.0,  # seconds
                        'interval_length': 0.5,  # seconds
                        'num_poses': 8
                    },
                    'latent': False,  # Start with non-latent version
                    'use_lidar': True,
                    'use_camera': True,
                    'camera_channels': 3,
                    'lidar_channels': 1,
                    'history_num_frames': 4,
                    'future_num_frames': 8,
                    
                    # Diffusion parameters
                    'diffusion': {
                        'num_diffusion_iters': 10,
                        'variance_schedule': 'cosine',
                        'clip_denoised': True,
                        'predict_epsilon': True
                    }
                }
            },
            
            # Optimization
            'optimizer': {
                '_target_': 'torch.optim.AdamW',
                'lr': 6e-4,
                'weight_decay': 0.01,
                'betas': [0.9, 0.999]
            },
            
            'lr_scheduler': {
                '_target_': 'torch.optim.lr_scheduler.CosineAnnealingLR',
                'T_max': 50,
                'eta_min': 1e-6
            },
            
            # Experiment settings
            'experiment_name': 'diffusiondrive_bench2drive',
            'seed': 42,
            'force_cache_computation': False,  # Set True for first run
            
            # Logging
            'wandb': {
                'project': 'diffusiondrive-bench2drive',
                'entity': None,  # Your wandb entity
                'mode': 'online'  # or 'offline'
            }
        }
        
        # Save config
        config_path = self.exp_root / 'bench2drive_training_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Training config saved to: {config_path}")
        return config_path
    
    def create_launch_script(self, config_path: Path):
        """Create a launch script for training"""
        
        script_content = f"""#!/bin/bash
# DiffusionDrive training script for Bench2Drive

# Set environment variables
export NAVSIM_DEVKIT_ROOT={self.navsim_root}
export NAVSIM_EXP_ROOT={self.exp_root}
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on available GPUs
export CUDA_LAUNCH_BLOCKING=0

# Run training
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \\
    agent=diffusiondrive_agent \\
    +config={config_path} \\
    experiment_name=diffusiondrive_bench2drive_$(date +%Y%m%d_%H%M%S) \\
    train_test_split=trainval \\
    force_cache_computation=false \\
    "$@"  # Allow passing additional arguments
"""
        
        script_path = self.exp_root / 'run_training.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"Launch script created: {script_path}")
        return script_path
    
    def create_evaluation_script(self):
        """Create evaluation script for trained models"""
        
        script_content = f"""#!/bin/bash
# DiffusionDrive evaluation script for Bench2Drive

# Set environment variables
export NAVSIM_DEVKIT_ROOT={self.navsim_root}
export NAVSIM_EXP_ROOT={self.exp_root}
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH

# Check if checkpoint path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

CHECKPOINT_PATH=$1

# Run evaluation
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \\
    agent=diffusiondrive_agent \\
    agent.checkpoint_path=$CHECKPOINT_PATH \\
    +config={self.exp_root}/bench2drive_training_config.yaml \\
    split=val \\
    "${{@:2}}"  # Pass additional arguments
"""
        
        script_path = self.exp_root / 'run_evaluation.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"Evaluation script created: {script_path}")
        return script_path
    
    def print_training_guide(self):
        """Print a comprehensive training guide"""
        
        print("\n" + "="*60)
        print("DiffusionDrive + Bench2Drive Training Guide")
        print("="*60)
        
        print("\n1. Data Conversion:")
        print(f"   - Source: {self.b2d_root}")
        print(f"   - Target: {self.converted_data_root}")
        print("   - Run conversion for each split:")
        print("     python train_diffusiondrive_bench2drive.py --convert-only --split train")
        print("     python train_diffusiondrive_bench2drive.py --convert-only --split val")
        
        print("\n2. Training:")
        print(f"   - Config: {self.exp_root}/bench2drive_training_config.yaml")
        print(f"   - Script: {self.exp_root}/run_training.sh")
        print("   - Command: bash run_training.sh")
        
        print("\n3. Monitoring:")
        print("   - Logs: Check experiment folder for tensorboard logs")
        print("   - WandB: If configured, check your wandb project")
        
        print("\n4. Evaluation:")
        print(f"   - Script: {self.exp_root}/run_evaluation.sh")
        print("   - Command: bash run_evaluation.sh <checkpoint_path>")
        
        print("\n5. Tips:")
        print("   - Start with a small subset (--max-scenes 10) for testing")
        print("   - Adjust batch_size based on GPU memory")
        print("   - Use mixed precision training for faster convergence")
        print("   - Cache features on first run (force_cache_computation=true)")
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Train DiffusionDrive with Bench2Drive')
    parser.add_argument('--b2d-root', type=str, required=True,
                        help='Path to Bench2Drive dataset root')
    parser.add_argument('--navsim-root', type=str, required=True,
                        help='Path to NavSim/DiffusionDrive root')
    parser.add_argument('--exp-root', type=str, required=True,
                        help='Path to experiment root directory')
    parser.add_argument('--convert-only', action='store_true',
                        help='Only convert data, do not start training')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to convert')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Maximum number of scenes to convert (for testing)')
    
    args = parser.parse_args()
    
    # Check if torch is available
    try:
        import torch
    except ImportError:
        print("PyTorch not found. Please install PyTorch first.")
        sys.exit(1)
    
    trainer = DiffusionDriveBench2DriveTrainer(
        bench2drive_root=args.b2d_root,
        navsim_root=args.navsim_root,
        experiment_root=args.exp_root
    )
    
    if args.convert_only:
        # Just convert data
        trainer.convert_dataset(split=args.split, max_scenes=args.max_scenes)
    else:
        # Full pipeline
        print("Setting up DiffusionDrive training with Bench2Drive dataset...")
        
        # Convert data if needed
        if not (trainer.converted_data_root / 'train').exists():
            print("Converting training data...")
            trainer.convert_dataset('train', max_scenes=args.max_scenes)
        
        if not (trainer.converted_data_root / 'val').exists():
            print("Converting validation data...")
            trainer.convert_dataset('val', max_scenes=args.max_scenes)
        
        # Create configs and scripts
        config_path = trainer.create_training_config()
        if config_path:
            trainer.create_launch_script(config_path)
            trainer.create_evaluation_script()
            trainer.print_training_guide()
        else:
            print("Failed to create training configuration.")

if __name__ == '__main__':
    main()