# Diffusion Planner ROS2 Node - Technical Documentation

## Overview
The `diffusion_planner_node.py` is the core ROS2 interface that bridges perception/localization data with the diffusion-based trajectory planning model. This node serves as the critical link between Autoware's perception/localization stack and the advanced diffusion-based planning algorithm, enabling real-time autonomous driving decisions.

## Architecture Overview

```
Perception/Localization Data → ROS2 Topics → Diffusion Planner Node → Model Inference → Trajectory Output
```

## Key Components

### 1. **Initialization (lines 51-211)**
The node sets up:
- **Vector map loading**: Loads Lanelet2 map for lane information
- **Model configuration**: Loads diffusion planner config and model weights
- **Backend selection**: Supports both PyTorch and ONNX Runtime for inference
- **ROS subscribers**: Sets up listeners for vehicle state, perception, and route data
- **ROS publishers**: Creates publishers for trajectories and debug visualizations

### 2. **Input Subscribers**

#### **Vehicle State** (lines 111-124)
- **Topic**: `/localization/kinematic_state`
  - Message Type: `Odometry`
  - Contains: Current position, orientation, velocity
- **Topic**: `/localization/acceleration`
  - Message Type: `AccelWithCovarianceStamped`
  - Contains: Current acceleration

#### **Perception** (lines 127-141)
- **Topic**: `/perception/object_recognition/tracking/objects`
  - Message Type: `TrackedObjects`
  - Contains: Tracked vehicles, pedestrians, bicycles
- **Topic**: `/perception/traffic_light_recognition/traffic_signals`
  - Message Type: `TrafficLightGroupArray`
  - Contains: Traffic light states

#### **Route** (lines 144-156)
- **Topic**: `/planning/mission_planning/route`
  - Message Type: `LaneletRoute`
  - Contains: Planned route from mission planner
  - Uses transient QoS for persistence across restarts

### 3. **Main Processing Pipeline** (`cb_tracked_objects`, lines 224-414)

This callback is triggered when new tracked objects arrive and orchestrates the entire planning pipeline:

#### **Step 1: Data Synchronization** (lines 230-247)
- Finds the nearest kinematic state and acceleration messages by timestamp
- Ensures all inputs are temporally aligned
- Parses traffic light recognition data

#### **Step 2: Coordinate Transformation** (line 244)
- Computes transformation matrices between map and vehicle base_link frames
- Essential for converting between global map coordinates and vehicle-centric coordinates

#### **Step 3: Input Tensor Creation**

**Ego State** (lines 251-258):
- Creates current vehicle state tensor with position, velocity, acceleration
- Includes wheel base parameter for kinematic modeling
- Processing time logged: ~0.X ms

**Neighbors** (lines 261-273):
- Maintains tracking history of surrounding objects
- Converts to vehicle-centric coordinates using transformation matrix
- Creates tensor with past trajectories:
  - Max 32 objects
  - 21 timesteps of history
- Publishes debug markers for visualization
- Processing time logged: ~X.X ms

**Lanes** (lines 276-290):
- Extracts lane segments within 100m radius
- Includes traffic light states for each lane
- Sorts by relevance (70 segments max)
- Tensor includes:
  - Lane geometry
  - Speed limits
  - Traffic light states
- Processing time logged: ~X.X ms

**Route** (lines 293-325):
- Finds closest route segment to current position
- Extracts upcoming route lanes (25 segments)
- Converts to vehicle-centric tensor
- Publishes debug markers for route visualization
- Processing time logged: ~X.X ms

#### **Step 4: Model Inference** (lines 328-370)

**Input Preparation**:
```python
input_dict = {
    "ego_current_state": ego_current_state,
    "neighbor_agents_past": neighbor,
    "lanes": lanes_tensor,
    "lanes_speed_limit": lanes_speed_limit,
    "lanes_has_speed_limit": lanes_has_speed_limit,
    "route_lanes": route_tensor,
    "static_objects": torch.zeros((1, 5, 10), device=dev),
}
```

**Inference Process**:
- Normalizes inputs using configured normalizers
- Supports batch processing (configurable batch size)
- Runs diffusion model:
  - PyTorch backend for development
  - ONNX Runtime backend for deployment
- Processing time logged: ~XX.X ms (target: <50ms for 20Hz)

#### **Step 5: Output Publishing** (lines 372-413)

**Model Output Format**:
- Shape: `[batch_size, 11, T, 4]`
  - batch_size: Number of trajectory samples
  - 11: Fixed dimension (ego + 10 potential objects)
  - T: Time horizon
  - 4: [x, y, cos(heading), sin(heading)]

**Publishing Process**:
1. Converts predictions back to map coordinates
2. Extracts heading from cos/sin representation
3. Creates trajectory messages
4. Publishes:
   - **Main trajectory** (`/planning/scenario_planning/lane_driving/trajectory`)
     - PlanningTrajectory format for backward compatibility
     - Uses highest scoring trajectory
   - **Multiple trajectories** (`/diffusion_planner/trajectories`)
     - New Trajectories format
     - Includes all sampled trajectories with scores
   - **Debug visualizations**:
     - Neighbor markers (`/diffusion_planner/debug/neighbor_marker`)
     - Route markers (`/diffusion_planner/debug/route_marker`)
     - Trajectory markers (`/diffusion_planner/debug/trajectory_marker`)

## Key Features

### Real-time Performance
- Designed for ~50ms total latency (20Hz operation)
- Logs individual component processing times
- Supports GPU acceleration via CUDA

### Multi-trajectory Output
- Generates multiple trajectory options
- Each trajectory has an associated score
- Enables downstream selection based on additional criteria

### Coordinate Handling
- Seamlessly converts between map and vehicle frames
- Maintains consistency across all inputs and outputs
- Critical for accurate planning in real-world scenarios

### Debug Visualization
- Comprehensive marker publishing for debugging
- Visualizes:
  - Tracked objects and their histories
  - Selected route segments
  - Generated trajectories
- Essential for development and validation

### Flexible Backends
- **PyTorch**: Full model capabilities for development
- **ONNX Runtime**: Optimized inference for deployment
- Configurable via ROS parameters

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vector_map_path` | string | "None" | Path to Lanelet2 map file |
| `config_json_path` | string | "None" | Path to model configuration JSON |
| `backend` | string | "PYTHORCH" | Inference backend: "PYTHORCH" or "ONNXRUNTIME" |
| `ckpt_path` | string | "None" | Path to PyTorch checkpoint (if using PyTorch) |
| `onnx_path` | string | "None" | Path to ONNX model (if using ONNX Runtime) |
| `wheel_base` | float | 2.79 | Vehicle wheel base in meters |
| `batch_size` | int | 1 | Number of trajectory samples to generate |

## Usage Example

```bash
ros2 run diffusion_planner_ros diffusion_planner_node \
    --ros-args \
    -p vector_map_path:=/path/to/lanelet2_map.osm \
    -p config_json_path:=/path/to/config.json \
    -p ckpt_path:=/path/to/model.pth \
    -p backend:=PYTHORCH \
    -p batch_size:=5
```

## Integration with Autoware

This node is designed to integrate seamlessly with the Autoware.Universe autonomous driving stack:
- Consumes standard Autoware perception and localization messages
- Outputs trajectories compatible with Autoware's motion planning framework
- Supports Autoware's coordinate system conventions
- Compatible with Autoware's visualization tools

## Performance Considerations

1. **GPU Memory**: Model requires significant GPU memory, especially with larger batch sizes
2. **Latency**: Total processing should remain under 50ms for 20Hz operation
3. **CPU Load**: Coordinate transformations and data preparation can be CPU-intensive
4. **Network Traffic**: Debug visualizations can generate significant ROS network traffic

## Future Improvements

1. **Asynchronous Processing**: Implement parallel processing for input preparation
2. **Dynamic Batching**: Adjust batch size based on computational resources
3. **Caching**: Cache static map data transformations
4. **Profiling**: Add detailed profiling for optimization
5. **Error Handling**: Enhance robustness to missing or delayed messages