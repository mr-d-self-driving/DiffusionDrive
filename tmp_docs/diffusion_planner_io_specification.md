# Diffusion Planner Input/Output Specification

## Overview
This document provides a detailed specification of all inputs and outputs for the Diffusion Planner autonomous driving system, including data formats, sources, and processing locations.

## Inputs to Diffusion Planner

The diffusion planner takes the following inputs (defined in `/workspace/Project_Diffusion-Pannar/Diffusion-Planner/diffusion_planner_ros/diffusion_planner_ros/diffusion_planner_node.py`, lines 328-337):

### 1. **Ego Current State**
- **Processing Location**: `utils.py:create_current_ego_state()`
- **Data Shape**: `[batch_size, 7]`
- **Features**: `[x, y, vx, vy, ax, ay, heading]`
- **ROS Topics**: 
  - `/localization/kinematic_state` (Odometry)
  - `/localization/acceleration` (AccelWithCovarianceStamped)
- **Coordinate Frame**: Vehicle base_link
- **Update Rate**: ~10 Hz

### 2. **Neighbor Agents Past**
- **Processing Location**: `utils.py:convert_tracked_objects_to_tensor()`
- **Data Shape**: `[batch_size, max_objects(32), timesteps(21), 7]`
- **Features**: `[x, y, vx, vy, heading, width, length]`
- **ROS Topic**: `/perception/object_recognition/tracking/objects` (TrackedObjects)
- **Object Types**: Vehicles, pedestrians, bicycles
- **Coordinate Frame**: Vehicle-centric (transformed from map frame)
- **History**: 21 timesteps (~2.1 seconds at 10 Hz)

### 3. **Lanes**
- **Processing Location**: `lanelet2_utils/lanelet_converter.py:create_lane_tensor()`
- **Data Shape**: `[batch_size, num_lanes(70), num_points(10), 7]`
- **Features**: `[x, y, z, dx, dy, dz, traffic_light_state]`
- **Additional Tensors**:
  - `lanes_speed_limit`: `[batch_size, num_lanes]` - Speed limit in m/s
  - `lanes_has_speed_limit`: `[batch_size, num_lanes]` - Boolean mask
- **Source**: Lanelet2 vector map (loaded at startup)
- **Processing**: 
  - Extracts lanes within 100m radius
  - Sorts by relevance to ego vehicle
  - Includes traffic light states from `/perception/traffic_light_recognition/traffic_signals`

### 4. **Route Lanes**
- **Processing Location**: Same as lanes but filtered for route
- **Data Shape**: `[batch_size, num_route_lanes(25), num_points(10), 7]`
- **Features**: Same as lanes
- **ROS Topic**: `/planning/mission_planning/route` (LaneletRoute)
- **Processing**:
  - Finds closest route segment to current position
  - Extracts next 25 segments along route
  - No sorting (maintains route order)

### 5. **Static Objects**
- **Current Implementation**: Zero tensor `[batch_size, 5, 10]`
- **Status**: Placeholder for future static obstacle integration
- **Planned Features**: Static obstacles like parked cars, barriers

## Input Processing Pipeline

```python
# Input dictionary structure passed to model
input_dict = {
    "ego_current_state": ego_current_state,          # Vehicle state
    "neighbor_agents_past": neighbor,                 # Tracked objects
    "lanes": lanes_tensor,                           # Map lanes
    "lanes_speed_limit": lanes_speed_limit,          # Speed limits
    "lanes_has_speed_limit": lanes_has_speed_limit,  # Speed limit mask
    "route_lanes": route_tensor,                     # Route lanes
    "static_objects": torch.zeros((1, 5, 10))        # Placeholder
}
```

## Outputs from Diffusion Planner

### Raw Model Output
- **Location**: Model inference at lines 363-367
- **Data Shape**: `[batch_size, 11, time_horizon, 4]`
- **Dimensions**:
  - `batch_size`: Number of trajectory samples
  - `11`: Ego vehicle + 10 potential objects (only ego used currently)
  - `time_horizon`: Future timesteps (typically 50-80 steps)
  - `4`: Features `[x, y, cos(heading), sin(heading)]`
- **Coordinate Frame**: Vehicle-centric

### Processed Outputs

#### 1. **Main Trajectory** (Backward Compatibility)
- **ROS Topic**: `/planning/scenario_planning/lane_driving/trajectory`
- **Message Type**: `PlanningTrajectory`
- **Processing Location**: Lines 405-408
- **Contents**:
  - Header with timestamp and frame_id
  - Array of trajectory points with:
    - Pose (position + orientation)
    - Longitudinal velocity
    - Lateral velocity
    - Heading rate
    - Acceleration
- **Selection**: Uses highest scoring trajectory (first in batch)

#### 2. **Multiple Trajectories** (New Format)
- **ROS Topic**: `/diffusion_planner/trajectories`
- **Message Type**: `Trajectories`
- **Processing Location**: Lines 373-412
- **Contents**:
  - Generator info (UUID and name)
  - Array of trajectories, each with:
    - Individual trajectory data (same as main trajectory)
    - Score (1.0 for best, decreasing for alternatives)
    - Generator ID reference
- **Use Case**: Enables downstream selection and planning diversity

#### 3. **Debug Visualizations**
All visualization markers use `MarkerArray` message type:

- **Neighbor Markers**
  - Topic: `/diffusion_planner/debug/neighbor_marker`
  - Shows: Tracked objects with history trails
  - Color coding by object type

- **Route Markers**
  - Topic: `/diffusion_planner/debug/route_marker`
  - Shows: Selected route segments
  - Highlights upcoming path

- **Trajectory Markers**
  - Topic: `/diffusion_planner/debug/trajectory_marker`
  - Shows: All generated trajectories
  - Color/size indicates score

## Core Model Components

### Model Architecture Files
- **Main Model**: `/workspace/Project_Diffusion-Pannar/Diffusion-Planner/diffusion_planner/model/diffusion_planner.py`
  - Implements the diffusion process
  - Handles noising/denoising steps
  - Coordinates encoder/decoder

- **DiT Architecture**: `/workspace/Project_Diffusion-Pannar/Diffusion-Planner/diffusion_planner/model/module/dit.py`
  - Diffusion Transformer implementation
  - Self-attention mechanisms
  - Temporal modeling

- **Encoder**: `/workspace/Project_Diffusion-Pannar/Diffusion-Planner/diffusion_planner/model/module/encoder.py`
  - Processes input features
  - Creates embeddings for:
    - Ego state
    - Neighbor agents
    - Lane geometry
    - Route information

- **Decoder**: `/workspace/Project_Diffusion-Pannar/Diffusion-Planner/diffusion_planner/model/module/decoder.py`
  - Converts diffusion outputs to trajectories
  - Applies denormalization
  - Ensures kinematic constraints

### Normalization
- **Config Location**: `normalization.json`
- **Implementation**: `diffusion_planner/utils/normalizer.py`
- **Normalizes**:
  - Positions to [-1, 1] range
  - Velocities/accelerations to standard scales
  - Ensures numerical stability

## Data Flow Summary

```
1. ROS Topics → 2. Input Processing → 3. Normalization → 4. Diffusion Model → 5. Denormalization → 6. Output Messages
      ↓                  ↓                    ↓                   ↓                     ↓                    ↓
   Sensors        Vehicle Frame         Config JSON         PyTorch/ONNX          Map Frame            Autoware
```

### Processing Steps:
1. **Data Collection**: Gather sensor data from ROS topics
2. **Synchronization**: Align data by timestamps
3. **Transformation**: Convert to vehicle-centric coordinates
4. **Tensorization**: Create model-compatible tensors
5. **Normalization**: Scale inputs for model
6. **Inference**: Run diffusion model
7. **Denormalization**: Convert outputs to real units
8. **Transformation**: Convert back to map coordinates
9. **Publishing**: Send trajectories to planning stack

## Performance Characteristics

### Input Processing Times (typical)
- Ego state: ~0.1 ms
- Neighbor tracking: ~2-5 ms
- Lane extraction: ~5-10 ms
- Route processing: ~1-3 ms

### Model Inference
- PyTorch backend: ~30-40 ms
- ONNX Runtime: ~20-30 ms
- Target: <50 ms total (20 Hz)

### Output Generation
- Trajectory conversion: ~1-2 ms
- Marker generation: ~2-3 ms

## Integration Notes

### Coordinate Systems
- **Map Frame**: Global fixed frame (typically "map")
- **Base Link**: Vehicle body frame (typically "base_link")
- All model computations in base_link frame
- Outputs transformed back to map frame

### Time Synchronization
- Uses ROS timestamps for alignment
- Maintains message history for synchronization
- Drops old messages to prevent memory growth

### Error Handling
- Checks for missing route before processing
- Validates transformation matrices
- Logs warnings for missing data
- Continues operation with partial data when possible