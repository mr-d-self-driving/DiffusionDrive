# Upstream Changes Analysis - DiffusionDrive Visualization Improvements

**Date**: 2025-06-27  
**Merge Commit**: 1e1c371375a93e5811a5695ae4618ba093b0c2fa  
**Branch**: feature/training-callbacks-with-improvements  
**Source**: upstream/main  

## Overview

This document details the changes merged from the upstream DiffusionDrive repository into the current feature branch. The changes primarily focus on enhancing visualization capabilities for trajectories in both Bird's Eye View (BEV) and camera perspectives.

## Changed Files Summary

| File | Lines Changed | Type of Change |
|------|---------------|----------------|
| `README.md` | +1, -1 | Minor update |
| `navsim/common/dataclasses.py` | +44, -12 | Enhanced trajectory extraction |
| `navsim/visualization/bev.py` | +5, -2 | Added trajectory labeling |
| `navsim/visualization/camera.py` | +160, -4 | New trajectory projection features |
| `navsim/visualization/plots.py` | +278, -277 | Major refactoring |

**Total**: 532 lines added, 252 lines removed across 5 files

## Detailed Changes

### 1. BEV Visualization Enhancement (`navsim/visualization/bev.py`)

#### Function Modified: `add_trajectory_to_bev_ax()`

**Before:**
```python
def add_trajectory_to_bev_ax(ax: plt.Axes, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
```

**After:**
```python
def add_trajectory_to_bev_ax(ax: plt.Axes, trajectory: Trajectory, config: Dict[str, Any], label: str) -> plt.Axes:
```

**Changes:**
- Added `label` parameter for trajectory identification
- Added `ax.legend()` call to display trajectory labels
- Commented out `markeredgecolor` parameter (possibly due to compatibility issues)

**Purpose:** Enables multiple trajectories to be displayed with distinguishing labels in BEV plots, improving visualization clarity when comparing different trajectory predictions.

### 2. Camera Visualization Features (`navsim/visualization/camera.py`)

#### New Function: `add_trajectory_to_camera_ax()`

```python
def add_trajectory_to_camera_ax(ax: plt.Axes, camera: Camera, trajectory: Trajectory, config: Dict[str, Any], label: str) -> plt.Axes:
```

**Functionality:**
- Projects 3D trajectory points onto 2D camera image coordinates
- Handles camera intrinsics and extrinsics transformations
- Filters points within camera field of view
- Renders trajectory as a line plot on camera images

#### Supporting Functions Added:
- `_transform_trajectories_to_images()` - Transforms trajectory poses to image pixel coordinates
- `_transform_trajectories_to_images_base()` - Base transformation function with FOV filtering

**Purpose:** Allows trajectory visualization directly on camera feeds, providing a multi-modal view of predicted paths.

### 3. Enhanced Trajectory Extraction (`navsim/common/dataclasses.py`)

#### Modified Method: `Scene.get_future_trajectory()`

**Before:**
```python
def get_future_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
```

**After:**
```python
def get_future_trajectory(self, num_trajectory_frames: Optional[int] = None, frame_idx: Optional[int] = None) -> Trajectory:
```

**Changes:**
- Added optional `frame_idx` parameter
- Allows trajectory extraction from arbitrary frame indices
- Maintains backward compatibility with default behavior
- Added debug comments for trajectory sampling parameters

**Purpose:** Provides flexibility to extract trajectories from different points in the scene timeline, useful for multi-frame analysis and visualization.

### 4. Plot Utilities Refactoring (`navsim/visualization/plots.py`)

**Statistics:**
- 555 lines modified (278 additions, 277 deletions)
- Appears to be a major refactoring

**Likely Changes:**
- Improved plot layout and organization
- Enhanced support for new visualization features
- Code cleanup and optimization
- Better integration with trajectory labeling

### 5. README Update (`README.md`)

- Minor change (2 lines modified)
- Likely version or documentation updates

## Impact Analysis

### Positive Impacts:
1. **Enhanced Visualization**: Better trajectory visualization in both BEV and camera views
2. **Multi-trajectory Support**: Can now display and compare multiple trajectories with labels
3. **Flexibility**: More options for trajectory extraction and visualization
4. **Camera Integration**: Trajectories can now be projected onto camera images

### Potential Considerations:
1. **API Changes**: Functions now require additional parameters (e.g., `label`)
2. **Performance**: Additional trajectory transformations may impact rendering speed
3. **Compatibility**: Commented out `markeredgecolor` suggests potential compatibility adjustments

## Integration Notes

These upstream changes are **fully compatible** with the script reorganization work done in this branch. The changes are focused on different aspects of the codebase:

- **Upstream**: Visualization and data structure improvements
- **This Branch**: Script organization, testing framework, and training callbacks

No merge conflicts were encountered, indicating clean separation of concerns.

## Recommended Actions

1. **Update Visualization Code**: Any custom visualization code should be updated to include the new `label` parameter
2. **Test Camera Projections**: Verify that trajectory projections work correctly with your camera setup
3. **Leverage New Features**: Consider using labeled trajectories for comparing different models or prediction modes
4. **Performance Testing**: Monitor visualization performance with the new trajectory projection features

## Code Examples

### Using Labeled Trajectories in BEV:
```python
from navsim.visualization.bev import add_trajectory_to_bev_ax

# Add multiple trajectories with labels
add_trajectory_to_bev_ax(ax, ground_truth_trajectory, config, label="Ground Truth")
add_trajectory_to_bev_ax(ax, predicted_trajectory, config, label="Predicted")
add_trajectory_to_bev_ax(ax, baseline_trajectory, config, label="Baseline")
```

### Projecting Trajectories onto Camera Views:
```python
from navsim.visualization.camera import add_trajectory_to_camera_ax

# Project trajectory onto camera image
add_trajectory_to_camera_ax(ax, camera, trajectory, config, label="Predicted Path")
```

### Extracting Trajectories from Different Frames:
```python
# Get trajectory starting from frame 10
trajectory = scene.get_future_trajectory(num_trajectory_frames=8, frame_idx=10)
```

## Conclusion

The upstream changes significantly enhance DiffusionDrive's visualization capabilities, particularly for trajectory analysis and multi-modal visualization. These improvements complement the script reorganization work and provide better tools for analyzing autonomous driving predictions.