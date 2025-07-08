# Driving Command Analysis in DiffusionDrive/NAVSIM

## Overview

This document contains the analysis of driving command format in DiffusionDrive (NAVSIM) dataset, including the scripts used to inspect the data and the findings.

## Key Findings

### 1. Command Format

NAVSIM uses a **4-dimensional one-hot encoded vector** for driving commands:

```python
[1, 0, 0, 0]  # LEFT TURN
[0, 1, 0, 0]  # GO STRAIGHT  
[0, 0, 1, 0]  # RIGHT TURN
[0, 0, 0, 1]  # STOP/OTHER
```

### 2. Data Structure

- **Type**: `numpy.ndarray[int]` with shape (4,)
- **Location in code**: `navsim/common/dataclasses.py`, line 138
- **Access path**: `agent_input.ego_statuses[-1].driving_command`

### 3. Usage in Model

The driving command is concatenated with velocity and acceleration to create an 8D status feature vector:

```python
features["status_feature"] = torch.concatenate([
    torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),  # 4D
    torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),     # 2D
    torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32), # 2D
])
```

## Analysis Scripts

### Script 1: Basic Pickle File Inspection

```python
import pickle
import numpy as np

# Load a sample pickle file
with open('/workspace/navsim_workspace/dataset/navsim_logs/test/2021.05.25.14.16.10_veh-35_00083_00485.pkl', 'rb') as f:
    data = pickle.load(f)

# Print the structure
print('Type:', type(data))
print('Length:', len(data) if hasattr(data, '__len__') else 'N/A')

# If it's a list, show first frame
if isinstance(data, list) and len(data) > 0:
    print('\nFirst frame keys:', list(data[0].keys()))
    if 'driving_command' in data[0]:
        print('\nDriving command in first few frames:')
        for i in range(min(5, len(data))):
            print(f'  Frame {i}: {data[i]["driving_command"]}')
```

### Script 2: Analyze Command Encoding

```python
import pickle
import numpy as np

# Load a sample pickle file
with open('/workspace/navsim_workspace/dataset/navsim_logs/test/2021.05.25.14.16.10_veh-35_00083_00485.pkl', 'rb') as f:
    data = pickle.load(f)

# Analyze driving commands
commands = set()
for frame in data[:100]:  # Check first 100 frames
    cmd = tuple(frame['driving_command'])
    commands.add(cmd)

print('Unique driving commands found:')
for cmd in sorted(commands):
    print(f'  {list(cmd)}')

# Check if it's one-hot encoded
print('\nChecking pattern (one-hot encoding?):')
for cmd in sorted(commands):
    sum_val = sum(cmd)
    print(f'  {list(cmd)} -> sum = {sum_val}')

# Map to simple commands
print('\nMapping to simple commands:')
for cmd in sorted(commands):
    if cmd == (1, 0, 0, 0):
        print(f'  {list(cmd)} -> LEFT (index 0)')
    elif cmd == (0, 1, 0, 0):
        print(f'  {list(cmd)} -> STRAIGHT (index 1)')
    elif cmd == (0, 0, 1, 0):
        print(f'  {list(cmd)} -> RIGHT (index 2)')
    elif cmd == (0, 0, 0, 1):
        print(f'  {list(cmd)} -> UNKNOWN/STOP (index 3)')
```

### Script 3: Find All Command Types Across Dataset

```python
import pickle
import os
import glob

# Sample multiple files to find all command types
log_dir = '/workspace/navsim_workspace/dataset/navsim_logs/trainval/'
files = glob.glob(os.path.join(log_dir, '*.pkl'))[:20]  # Check first 20 files

all_commands = {}

for file_path in files:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Collect all unique commands from this file
    file_commands = set()
    for frame in data:
        cmd = tuple(frame['driving_command'])
        file_commands.add(cmd)
    
    # Store file name if it has interesting commands
    for cmd in file_commands:
        if cmd not in all_commands:
            all_commands[cmd] = []
        all_commands[cmd].append(os.path.basename(file_path))

print('All unique driving commands found:')
for cmd, files in sorted(all_commands.items()):
    print(f'\n{list(cmd)} found in {len(files)} files')
    print(f'  Example: {files[0]}')

print('\n\nCommand encoding summary:')
print('NAVSIM uses 4D one-hot encoding for driving commands:')
print('  [1, 0, 0, 0] = LEFT TURN')
print('  [0, 1, 0, 0] = GO STRAIGHT') 
print('  [0, 0, 1, 0] = RIGHT TURN')
print('  [0, 0, 0, 1] = STOP/OTHER')
```

### Script 4: Analyze Command Distribution

```python
import pickle
import numpy as np
from collections import Counter

# Load multiple files and analyze command distribution
files = [
    '/workspace/navsim_workspace/dataset/navsim_logs/trainval/2021.06.09.12.39.51_veh-26_00055_00360.pkl',
    '/workspace/navsim_workspace/dataset/navsim_logs/trainval/2021.06.09.14.50.36_veh-26_00063_00350.pkl',
    '/workspace/navsim_workspace/dataset/navsim_logs/test/2021.09.09.19.10.24_veh-39_00148_00372.pkl'
]

command_counter = Counter()
total_frames = 0

for file_path in files:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    for frame in data:
        cmd = tuple(frame['driving_command'])
        command_counter[cmd] += 1
        total_frames += 1

print('Command distribution:')
for cmd, count in command_counter.most_common():
    percentage = (count / total_frames) * 100
    print(f'  {list(cmd)}: {count} frames ({percentage:.1f}%)')
```

## Mapping for Bench2Drive Integration

When converting from CARLA/Bench2Drive commands to NAVSIM format:

```python
def carla_to_navsim_command(carla_command: str) -> np.ndarray:
    """Convert CARLA command string to NAVSIM one-hot encoding"""
    
    # Define mapping
    command_map = {
        # Left commands
        'CHANGELANELEFT': [1, 0, 0, 0],
        'TURNLEFT': [1, 0, 0, 0],
        'LEFT': [1, 0, 0, 0],
        
        # Straight commands
        'STRAIGHT': [0, 1, 0, 0],
        'LANEFOLLOW': [0, 1, 0, 0],
        
        # Right commands
        'CHANGELANERIGHT': [0, 0, 1, 0],
        'TURNRIGHT': [0, 0, 1, 0],
        'RIGHT': [0, 0, 1, 0],
        
        # Stop/Other
        'STOP': [0, 0, 0, 1],
        'UNKNOWN': [0, 0, 0, 1]
    }
    
    # Default to straight if command not found
    return np.array(command_map.get(carla_command.upper(), [0, 1, 0, 0]), dtype=np.int32)
```

## Dataset File Structure

Each pickle file in NAVSIM contains a list of frames, where each frame is a dictionary with keys:

- `token`: Unique frame identifier
- `driving_command`: 4D one-hot encoded command vector
- `ego2global_translation`: Ego vehicle position
- `ego2global_rotation`: Ego vehicle rotation (quaternion)
- `ego_dynamic_state`: [vx, vy, ax, ay] - velocity and acceleration
- `cams`: Camera data dictionary
- `lidar_path`: Path to LiDAR data
- `anns`: Annotations (bounding boxes, etc.)
- And many more...

## Notes

1. The one-hot encoding ensures that exactly one command is active at any time (sum = 1)
2. The 4D vector is treated as a continuous feature in the neural network
3. Most frames in highway/straight road scenarios will have [0, 1, 0, 0] (straight)
4. Intersection and turning scenarios will have more varied commands
5. The [0, 0, 0, 1] command appears to be used for stop signs, red lights, or other special cases

## Usage Example

To use this in your code:

```python
# Access driving command from agent input
driving_cmd = agent_input.ego_statuses[-1].driving_command
print(f"Current command: {driving_cmd}")

# Interpret the command
if np.array_equal(driving_cmd, [1, 0, 0, 0]):
    action = "Turn Left"
elif np.array_equal(driving_cmd, [0, 1, 0, 0]):
    action = "Go Straight"
elif np.array_equal(driving_cmd, [0, 0, 1, 0]):
    action = "Turn Right"
else:
    action = "Stop/Other"
print(f"Action: {action}")
```
