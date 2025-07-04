#!/usr/bin/env python3
"""
Bench2Drive to nuScenes Format Converter

This script converts Bench2Drive data to nuScenes format, which can then be used
as a bridge to NavSim format since NavSim has better support for nuScenes data.
"""

import os
import json
import gzip
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from datetime import datetime
from tqdm import tqdm
import shutil
from scipy.spatial.transform import Rotation as R


class Bench2DriveToNuScenesConverter:
    """Convert Bench2Drive dataset to nuScenes format"""
    
    def __init__(self, b2d_root: Path, output_root: Path):
        self.b2d_root = Path(b2d_root)
        self.output_root = Path(output_root)
        
        # nuScenes directory structure
        self.setup_nuscenes_dirs()
        
        # Initialize database tables
        self.scene_table = []
        self.sample_table = []
        self.sample_data_table = []
        self.ego_pose_table = []
        self.calibrated_sensor_table = []
        self.sensor_table = []
        self.instance_table = []
        self.sample_annotation_table = []
        self.category_table = []
        self.attribute_table = []
        self.log_table = []
        self.map_table = []
        
        # Token generators
        self.token_counter = 0
        
        # Setup sensors and categories
        self.setup_sensors()
        self.setup_categories()
    
    def setup_nuscenes_dirs(self):
        """Create nuScenes directory structure"""
        dirs = ['v1.0-mini', 'samples', 'sweeps', 'maps']
        for d in dirs:
            (self.output_root / d).mkdir(parents=True, exist_ok=True)
        
        # Create sensor subdirectories
        sensors = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP']
        for sensor in sensors:
            (self.output_root / 'samples' / sensor).mkdir(exist_ok=True)
            (self.output_root / 'sweeps' / sensor).mkdir(exist_ok=True)
    
    def generate_token(self) -> str:
        """Generate unique token"""
        self.token_counter += 1
        return f"{self.token_counter:032x}"
    
    def setup_sensors(self):
        """Setup sensor configuration matching nuScenes"""
        self.sensors = {
            'CAM_FRONT': {'channel': 'CAM_FRONT', 'modality': 'camera'},
            'CAM_FRONT_LEFT': {'channel': 'CAM_FRONT_LEFT', 'modality': 'camera'},
            'CAM_FRONT_RIGHT': {'channel': 'CAM_FRONT_RIGHT', 'modality': 'camera'},
            'CAM_BACK': {'channel': 'CAM_BACK', 'modality': 'camera'},
            'CAM_BACK_LEFT': {'channel': 'CAM_BACK_LEFT', 'modality': 'camera'},
            'CAM_BACK_RIGHT': {'channel': 'CAM_BACK_RIGHT', 'modality': 'camera'},
            'LIDAR_TOP': {'channel': 'LIDAR_TOP', 'modality': 'lidar'},
        }
        
        # Camera mapping from Bench2Drive to nuScenes
        self.camera_mapping = {
            'rgb_front': 'CAM_FRONT',
            'rgb_front_left': 'CAM_FRONT_LEFT',
            'rgb_front_right': 'CAM_FRONT_RIGHT',
            'rgb_back': 'CAM_BACK',
            'rgb_back_left': 'CAM_BACK_LEFT',
            'rgb_back_right': 'CAM_BACK_RIGHT',
        }
        
        # Create sensor table entries
        for sensor_key, sensor_info in self.sensors.items():
            self.sensor_table.append({
                'token': self.generate_token(),
                'channel': sensor_info['channel'],
                'modality': sensor_info['modality']
            })
    
    def setup_categories(self):
        """Setup object categories matching nuScenes"""
        # Simplified category mapping
        categories = [
            'vehicle.car', 'vehicle.truck', 'vehicle.bus', 'vehicle.bicycle',
            'vehicle.motorcycle', 'human.pedestrian', 'movable_object.trafficcone',
            'static_object.bicycle_rack'
        ]
        
        for category in categories:
            self.category_table.append({
                'token': self.generate_token(),
                'name': category,
                'description': f"Category: {category}"
            })
        
        # Simple attributes
        self.attribute_table.append({
            'token': self.generate_token(),
            'name': 'vehicle.moving',
            'description': 'Vehicle is moving'
        })
        self.attribute_table.append({
            'token': self.generate_token(),
            'name': 'vehicle.stopped',
            'description': 'Vehicle is stopped'
        })
    
    def carla_to_nuscenes_transform(self, carla_transform: Dict) -> Dict:
        """Convert CARLA transform to nuScenes format"""
        # CARLA uses different coordinate system than nuScenes
        loc = carla_transform['location']
        rot = carla_transform['rotation']
        
        # Convert location (CARLA to nuScenes coordinate system)
        translation = [loc['x'], -loc['y'], loc['z']]
        
        # Convert rotation
        # CARLA: pitch, yaw, roll in degrees
        # nuScenes: quaternion [w, x, y, z]
        r = R.from_euler('zyx', [rot['yaw'], rot['pitch'], rot['roll']], degrees=True)
        quat = r.as_quat()  # [x, y, z, w]
        rotation = [quat[3], quat[0], quat[1], quat[2]]  # [w, x, y, z]
        
        return {
            'translation': translation,
            'rotation': rotation
        }
    
    def convert_camera_intrinsics(self, cam_params: Dict) -> np.ndarray:
        """Convert camera parameters to nuScenes intrinsic matrix"""
        if 'intrinsic' in cam_params:
            return np.array(cam_params['intrinsic'])
        
        # Default intrinsics if not provided
        width = cam_params.get('width', 640)
        height = cam_params.get('height', 480)
        fov = cam_params.get('fov', 90)
        
        # Calculate focal length from FOV
        focal_length = width / (2 * np.tan(np.radians(fov) / 2))
        
        return np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ])
    
    def process_frame(self, scene_path: Path, frame_idx: int, scene_token: str, 
                      log_token: str, prev_sample_token: Optional[str] = None) -> str:
        """Process a single frame and create nuScenes entries"""
        
        # Load annotation
        anno_path = scene_path / 'anno' / f'{frame_idx:05d}.json.gz'
        with gzip.open(anno_path, 'rt') as f:
            anno = json.load(f)
        
        # Generate tokens
        sample_token = self.generate_token()
        timestamp = int(frame_idx * 1e6 / 10)  # 10Hz to microseconds
        
        # Create sample entry
        sample_entry = {
            'token': sample_token,
            'timestamp': timestamp,
            'prev': prev_sample_token or '',
            'next': '',  # Will be updated later
            'scene_token': scene_token
        }
        self.sample_table.append(sample_entry)
        
        # Process ego pose
        ego_transform = anno['ego_vehicle']
        ego_pose = self.carla_to_nuscenes_transform(ego_transform)
        ego_pose_token = self.generate_token()
        
        self.ego_pose_table.append({
            'token': ego_pose_token,
            'timestamp': timestamp,
            'rotation': ego_pose['rotation'],
            'translation': ego_pose['translation']
        })
        
        # Process each sensor
        sample_data_tokens = {}
        
        # Process cameras
        for b2d_cam, nus_cam in self.camera_mapping.items():
            cam_path = scene_path / 'camera' / b2d_cam / f'{frame_idx:05d}.jpg'
            if cam_path.exists():
                # Copy image to nuScenes structure
                rel_path = f"samples/{nus_cam}/{sample_token}.jpg"
                dst_path = self.output_root / rel_path
                shutil.copy(cam_path, dst_path)
                
                # Get calibration
                cam_params = anno.get('sensors', {}).get(b2d_cam.replace('rgb_', ''), {})
                calibration = self.convert_camera_intrinsics(cam_params)
                
                # Create calibrated sensor entry
                calib_token = self.generate_token()
                sensor_transform = cam_params.get('transform', ego_transform)
                sensor_pose = self.carla_to_nuscenes_transform(sensor_transform)
                
                self.calibrated_sensor_table.append({
                    'token': calib_token,
                    'sensor_token': next(s['token'] for s in self.sensor_table if s['channel'] == nus_cam),
                    'translation': sensor_pose['translation'],
                    'rotation': sensor_pose['rotation'],
                    'camera_intrinsic': calibration.tolist()
                })
                
                # Create sample data entry
                sample_data_token = self.generate_token()
                sample_data_tokens[nus_cam] = sample_data_token
                
                self.sample_data_table.append({
                    'token': sample_data_token,
                    'sample_token': sample_token,
                    'ego_pose_token': ego_pose_token,
                    'calibrated_sensor_token': calib_token,
                    'timestamp': timestamp,
                    'fileformat': 'jpg',
                    'is_key_frame': True,
                    'height': cam_params.get('height', 480),
                    'width': cam_params.get('width', 640),
                    'filename': rel_path,
                    'prev': '',  # Would need to track
                    'next': ''   # Would need to track
                })
        
        # Process LiDAR
        lidar_path = scene_path / 'lidar' / f'{frame_idx:05d}.laz'
        if lidar_path.exists():
            # Convert LiDAR to binary format
            rel_path = f"samples/LIDAR_TOP/{sample_token}.pcd.bin"
            dst_path = self.output_root / rel_path
            
            # Convert LAZ to binary (simplified - you'd need proper conversion)
            self.convert_lidar_to_bin(lidar_path, dst_path)
            
            # Create entries similar to cameras
            lidar_token = self.generate_token()
            sample_data_tokens['LIDAR_TOP'] = lidar_token
            
            # ... (similar calibration and sample_data entries)
        
        # Process annotations (objects)
        actors = anno.get('actors', [])
        for actor in actors:
            if actor.get('type_id', '').startswith('vehicle') or actor.get('type_id', '').startswith('walker'):
                self.create_annotation(actor, sample_token, sample_data_tokens)
        
        # Update sample with data tokens
        sample_entry['data'] = sample_data_tokens
        
        return sample_token
    
    def convert_lidar_to_bin(self, laz_path: Path, bin_path: Path):
        """Convert LAZ lidar to nuScenes binary format"""
        try:
            import laspy
            las = laspy.read(laz_path)
            
            # nuScenes expects 5 channels: x, y, z, intensity, ring
            points = np.zeros((len(las.x), 5), dtype=np.float32)
            points[:, 0] = las.x
            points[:, 1] = las.y
            points[:, 2] = las.z
            points[:, 3] = las.intensity if hasattr(las, 'intensity') else 0
            points[:, 4] = 0  # ring index not available in CARLA
            
            points.tofile(bin_path)
        except:
            # Create dummy file
            np.zeros((100, 5), dtype=np.float32).tofile(bin_path)
    
    def create_annotation(self, actor: Dict, sample_token: str, sample_data_tokens: Dict):
        """Create annotation entry for an actor"""
        
        # Map CARLA type to nuScenes category
        carla_type = actor.get('type_id', '').split('.')[0]
        category_map = {
            'vehicle': 'vehicle.car',
            'walker': 'human.pedestrian',
            'traffic': 'movable_object.trafficcone'
        }
        category = category_map.get(carla_type, 'vehicle.car')
        
        # Get category token
        category_token = next(
            (c['token'] for c in self.category_table if c['name'] == category),
            self.category_table[0]['token']
        )
        
        # Get bounding box info
        bbox = actor.get('bounding_box', {})
        location = bbox.get('location', {})
        extent = bbox.get('extent', {})
        rotation = actor.get('rotation', {})
        
        # Create instance if needed
        instance_token = self.generate_token()
        self.instance_table.append({
            'token': instance_token,
            'category_token': category_token,
            'nbr_annotations': 1,  # Will be updated
            'first_annotation_token': '',  # Will be updated
            'last_annotation_token': ''   # Will be updated
        })
        
        # Create annotation
        annotation_token = self.generate_token()
        
        # Convert to nuScenes coordinate system
        translation = [location.get('x', 0), -location.get('y', 0), location.get('z', 0)]
        size = [2 * extent.get('x', 1), 2 * extent.get('y', 1), 2 * extent.get('z', 1)]
        
        # Convert rotation
        r = R.from_euler('z', rotation.get('yaw', 0), degrees=True)
        quat = r.as_quat()
        rotation_quat = [quat[3], quat[0], quat[1], quat[2]]
        
        # Velocity
        velocity = actor.get('velocity', {})
        velocity_vals = [velocity.get('x', 0), -velocity.get('y', 0)]
        
        # Determine visibility (simplified)
        visibility_token = next(
            (sd for sd in sample_data_tokens.values()),
            ''
        )
        
        self.sample_annotation_table.append({
            'token': annotation_token,
            'sample_token': sample_token,
            'instance_token': instance_token,
            'visibility_token': visibility_token,
            'attribute_tokens': [self.attribute_table[0]['token']],  # moving
            'translation': translation,
            'size': size,
            'rotation': rotation_quat,
            'velocity': velocity_vals,
            'prev': '',  # Would need tracking
            'next': '',  # Would need tracking
            'num_lidar_pts': 100,  # Placeholder
            'num_radar_pts': 0     # No radar in output
        })
    
    def convert_scene(self, scene_path: Path) -> None:
        """Convert a complete scene"""
        
        scene_name = scene_path.name
        print(f"Converting scene: {scene_name}")
        
        # Create log entry
        log_token = self.generate_token()
        self.log_table.append({
            'token': log_token,
            'logfile': scene_name,
            'vehicle': 'ego_vehicle',
            'date_captured': datetime.now().strftime('%Y-%m-%d'),
            'location': scene_name.split('_')[0]  # Extract town name
        })
        
        # Create scene entry
        scene_token = self.generate_token()
        scene_entry = {
            'token': scene_token,
            'log_token': log_token,
            'nbr_samples': 0,  # Will be updated
            'first_sample_token': '',  # Will be updated
            'last_sample_token': '',   # Will be updated
            'name': scene_name,
            'description': f"Converted from Bench2Drive scene {scene_name}"
        }
        self.scene_table.append(scene_entry)
        
        # Get all frames
        anno_files = sorted((scene_path / 'anno').glob('*.json.gz'))
        
        prev_sample_token = None
        first_sample_token = None
        
        for i, frame_file in enumerate(tqdm(anno_files, desc=f"Processing {scene_name}")):
            frame_idx = int(frame_file.stem)
            
            try:
                sample_token = self.process_frame(
                    scene_path, frame_idx, scene_token, log_token, prev_sample_token
                )
                
                if i == 0:
                    first_sample_token = sample_token
                    scene_entry['first_sample_token'] = sample_token
                
                # Update previous sample's next pointer
                if prev_sample_token:
                    for sample in self.sample_table:
                        if sample['token'] == prev_sample_token:
                            sample['next'] = sample_token
                            break
                
                prev_sample_token = sample_token
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
        
        # Update scene entry
        scene_entry['last_sample_token'] = prev_sample_token
        scene_entry['nbr_samples'] = len(anno_files)
    
    def save_tables(self):
        """Save all tables to JSON files"""
        
        tables = {
            'scene': self.scene_table,
            'sample': self.sample_table,
            'sample_data': self.sample_data_table,
            'ego_pose': self.ego_pose_table,
            'calibrated_sensor': self.calibrated_sensor_table,
            'sensor': self.sensor_table,
            'instance': self.instance_table,
            'sample_annotation': self.sample_annotation_table,
            'category': self.category_table,
            'attribute': self.attribute_table,
            'log': self.log_table,
            'map': self.map_table
        }
        
        for name, table in tables.items():
            output_path = self.output_root / 'v1.0-mini' / f'{name}.json'
            with open(output_path, 'w') as f:
                json.dump(table, f, indent=2)
    
    def convert_dataset(self, max_scenes: Optional[int] = None):
        """Convert entire dataset"""
        
        # Find all scenes
        scenes = []
        for town_dir in self.b2d_root.iterdir():
            if town_dir.is_dir() and town_dir.name.startswith('Town'):
                scenes.extend(list(town_dir.iterdir()))
        
        if max_scenes:
            scenes = scenes[:max_scenes]
        
        print(f"Found {len(scenes)} scenes to convert")
        
        for scene_path in scenes:
            try:
                self.convert_scene(scene_path)
            except Exception as e:
                print(f"Error converting scene {scene_path.name}: {e}")
                continue
        
        # Save all tables
        self.save_tables()
        
        print(f"Conversion complete! Created nuScenes dataset at {self.output_root}")


def main():
    """Main conversion script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Bench2Drive to nuScenes format')
    parser.add_argument('--b2d-root', type=str, required=True,
                        help='Path to Bench2Drive dataset root')
    parser.add_argument('--output-root', type=str, required=True,
                        help='Path to output nuScenes dataset')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Maximum number of scenes to convert')
    
    args = parser.parse_args()
    
    converter = Bench2DriveToNuScenesConverter(
        b2d_root=Path(args.b2d_root),
        output_root=Path(args.output_root)
    )
    
    converter.convert_dataset(max_scenes=args.max_scenes)


if __name__ == '__main__':
    main()