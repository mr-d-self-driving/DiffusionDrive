# Bench2Drive to NavSim/nuScenes Dataset Conversion

This guide provides methods to convert Bench2Drive CARLA dataset to formats compatible with DiffusionDrive training.

## Overview

DiffusionDrive uses NavSim dataset format, which is based on nuPlan/nuScenes infrastructure. Since Bench2Drive uses a different format (HDF5-based with CARLA-specific structure), we need to convert the data.

## Conversion Methods

### Method 1: Direct Bench2Drive → NavSim Conversion

Use `bench2drive_to_navsim_converter.py` for direct conversion:

```bash
python bench2drive_to_navsim_converter.py \
    --b2d-root /path/to/bench2drive/dataset \
    --output-root /path/to/output/navsim \
    --split train \
    --max-scenes 10  # For testing, remove for full conversion
```

**Advantages:**
- Direct conversion, no intermediate format
- Preserves all sensor modalities
- Optimized for NavSim training

**Limitations:**
- Map information needs manual handling
- Some NavSim metrics may not work without proper map data

### Method 2: Using nuScenes as Bridge Format

First convert to nuScenes, then use existing NavSim tools:

```bash
# Step 1: Convert Bench2Drive to nuScenes
python bench2drive_to_nuscenes_converter.py \
    --b2d-root /path/to/bench2drive/dataset \
    --output-root /path/to/nuscenes/output \
    --max-scenes 10

# Step 2: Use NavSim's existing nuScenes support
# NavSim already has nuScenes dataloaders that can be adapted
```

**Advantages:**
- nuScenes is well-supported by many tools
- Can leverage existing nuScenes → NavSim pipelines
- Better compatibility with evaluation metrics

**Limitations:**
- Two-step process
- Some data loss in conversion

## Data Format Mapping

### Sensor Mapping

| Bench2Drive | NavSim/nuScenes | Notes |
|-------------|-----------------|-------|
| rgb_front | cam_f0 / CAM_FRONT | Direct mapping |
| rgb_front_left | cam_l0 / CAM_FRONT_LEFT | Direct mapping |
| rgb_front_right | cam_r0 / CAM_FRONT_RIGHT | Direct mapping |
| rgb_back | cam_b0 / CAM_BACK | Direct mapping |
| rgb_back_left | cam_l2 / CAM_BACK_LEFT | Direct mapping |
| rgb_back_right | cam_r2 / CAM_BACK_RIGHT | Direct mapping |
| - | cam_l1, cam_r1 | Side cameras - duplicated from front |
| lidar | LIDAR_TOP | Format conversion needed |
| radar_* | - | Not used in NavSim |
| depth_* | - | Not used in NavSim |
| semantic_* | - | Not standard in NavSim |

### Coordinate System Conversion

Bench2Drive (CARLA) uses:
- X: Forward
- Y: Right  
- Z: Up
- Left-handed system

NavSim/nuScenes uses:
- X: Right
- Y: Forward
- Z: Up  
- Right-handed system

The converters handle this transformation automatically.

## Installation Requirements

```bash
# Basic requirements
pip install numpy opencv-python tqdm h5py scipy

# For LiDAR conversion
pip install laspy

# For nuScenes format
pip install nuscenes-devkit

# For NavSim compatibility
# Follow NavSim installation guide
```

## Usage for DiffusionDrive Training

After conversion, you can use the converted dataset with DiffusionDrive:

1. **Update dataset configuration**:
   ```yaml
   # In NavSim config file
   dataset:
     data_root: /path/to/converted/navsim/dataset
     split: train
   ```

2. **Modify dataloader if needed**:
   - The converted data maintains NavSim structure
   - Camera intrinsics/extrinsics are preserved
   - Map information may need special handling

3. **Training considerations**:
   - Start with a small subset to verify conversion
   - Monitor training metrics carefully
   - Some planning metrics may not work without proper map data

## Handling Missing Components

### Map Data
Bench2Drive uses CARLA town maps, while NavSim expects nuPlan maps. Options:
1. Create dummy map data (affects planning metrics)
2. Use simplified lane representations
3. Convert CARLA OpenDRIVE to nuPlan format (complex)

### Roadblock IDs
These are nuPlan-specific and used for route planning:
- Can be left empty for pure learning tasks
- May affect some evaluation metrics

### Traffic Light States
Bench2Drive provides traffic light info which is mapped to NavSim format.

## Validation

After conversion, validate the dataset:

```python
# Check converted data
import pickle
import numpy as np

# Load metadata
with open('output/train_index.pkl', 'rb') as f:
    meta = pickle.load(f)
    
print(f"Total scenes: {len(meta['scenes'])}")
print(f"Total frames: {meta['total_frames']}")

# Load a sample frame
with open('output/train_frames.pkl', 'rb') as f:
    frames = pickle.load(f)
    
sample_frame = frames[0]
print("Frame keys:", sample_frame.keys())
print("Ego pose:", sample_frame['ego2global_translation'])
print("Cameras:", sample_frame['cams'].keys())
```

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**:
   - Process scenes in batches
   - Use `--max-scenes` parameter

2. **Missing sensor data**:
   - Check original Bench2Drive data completeness
   - Converters handle missing data gracefully

3. **Coordinate system issues**:
   - Verify transformations visually
   - Check ego vehicle trajectories

4. **LiDAR conversion errors**:
   - Ensure `laspy` is installed
   - LAZ files need proper decompression

## Advanced Usage

### Custom Map Integration

To add proper map support:

1. Export CARLA maps to OpenDRIVE
2. Convert to nuPlan map format
3. Update converter to use map API

### Extending the Converter

Add custom fields or sensors:

```python
# In converter class
def add_custom_data(self, frame_dict, b2d_data):
    # Add radar data
    frame_dict['radar'] = self.convert_radar(b2d_data['radar'])
    
    # Add semantic segmentation
    frame_dict['semantic'] = self.convert_semantic(b2d_data['semantic'])
```

## Performance Considerations

- Conversion speed: ~10-20 scenes/minute
- Storage: NavSim format uses more space due to less compression
- Consider using SSD for faster I/O during conversion

## Next Steps

1. Run conversion on a small subset first
2. Verify data integrity with visualization tools  
3. Test with DiffusionDrive training pipeline
4. Scale up to full dataset

For issues or questions, refer to:
- Bench2Drive documentation
- NavSim documentation  
- DiffusionDrive training guide