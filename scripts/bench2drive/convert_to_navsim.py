#!/usr/bin/env python3
"""
Bench2Drive to NavSim Dataset Converter

This script converts Bench2Drive CARLA dataset to NavSim format for training DiffusionDrive.
"""

import os
import json
import gzip
import pickle
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import uuid


class CoordinateTransformer:
    """Handles coordinate transformations between CARLA and NavSim/nuPlan formats"""
    
    @staticmethod
    def carla_to_nuplan_pose(carla_transform: Dict) -> Tuple[List[float], List[float]]:
        """
        Convert CARLA transform to nuPlan pose format
        
        Args:
            carla_transform: Dict with 'location' and 'rotation' keys
            
        Returns:
            translation: [x, y, z] in nuPlan coordinates
            rotation: quaternion [w, x, y, z]
        """
        # CARLA uses left-handed coordinate system, nuPlan uses right-handed
        # CARLA: x-forward, y-right, z-up
        # nuPlan: x-right, y-forward, z-up
        
        loc = carla_transform['location']
        rot = carla_transform['rotation']  # pitch, yaw, roll in degrees
        
        # Convert CARLA location to nuPlan
        translation = [loc['y'], -loc['x'], loc['z']]  # Swap and negate
        
        # Convert CARLA rotation to quaternion
        # CARLA rotation is in degrees, convert to radians
        pitch_rad = np.radians(rot['pitch'])
        yaw_rad = np.radians(rot['yaw'])
        roll_rad = np.radians(rot['roll'])
        
        # Create rotation matrix and convert to quaternion
        r = Rotation.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad])
        quaternion = r.as_quat()  # [x, y, z, w]
        quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # [w, x, y, z]
        
        return translation, quaternion
    
    @staticmethod
    def carla_velocity_to_nuplan(velocity: Dict, heading: float) -> List[float]:
        """Convert CARLA velocity to nuPlan velocity format"""
        # CARLA velocity is in world frame, convert to ego frame
        vx = velocity['x']
        vy = velocity['y']
        vz = velocity['z']
        
        # Rotate to ego frame
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        vx_ego = vx * cos_h + vy * sin_h
        vy_ego = -vx * sin_h + vy * cos_h
        
        return [vx_ego, vy_ego, 0.0]  # NavSim uses 2D velocity


class SensorDataMapper:
    """Maps Bench2Drive sensor data to NavSim format"""
    
    # Camera mapping from Bench2Drive to NavSim
    # NavSim uses: cam_f0, cam_l0, cam_l1, cam_l2, cam_r0, cam_r1, cam_r2, cam_b0
    # Bench2Drive has: front, front_left, front_right, back, back_left, back_right
    CAMERA_MAPPING = {
        'cam_f0': 'rgb_front',
        'cam_l0': 'rgb_front_left',
        'cam_l1': 'rgb_front_left',  # Duplicate as we don't have side cameras
        'cam_l2': 'rgb_back_left',
        'cam_r0': 'rgb_front_right',
        'cam_r1': 'rgb_front_right',  # Duplicate
        'cam_r2': 'rgb_back_right',
        'cam_b0': 'rgb_back'
    }
    
    @classmethod
    def map_cameras(cls, b2d_camera_data: Dict, camera_params: Dict) -> Dict:
        """Map Bench2Drive camera data to NavSim format"""
        navsim_cameras = {}
        
        for navsim_cam, b2d_cam in cls.CAMERA_MAPPING.items():
            if b2d_cam in b2d_camera_data:
                # Load image
                img_path = b2d_camera_data[b2d_cam]
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # Create placeholder image
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Get camera parameters
                cam_param = camera_params.get(b2d_cam.replace('rgb_', ''), {})
                
                navsim_cameras[navsim_cam] = {
                    'img': img,
                    'intrinsic': cam_param.get('intrinsic', np.eye(3)),
                    'extrinsic': cam_param.get('extrinsic', np.eye(4)),
                }
        
        return navsim_cameras
    
    @staticmethod
    def convert_lidar_data(lidar_path: str) -> np.ndarray:
        """Convert Bench2Drive LiDAR data to NavSim format"""
        # Bench2Drive uses LAZ format, NavSim expects numpy array
        # Format: [N, 6] with (x, y, z, intensity, ring_index, lidar_id)
        
        try:
            import laspy
            las = laspy.read(lidar_path)
            
            # Extract point cloud data
            points = np.vstack([
                las.x,
                las.y, 
                las.z,
                las.intensity if hasattr(las, 'intensity') else np.zeros(len(las.x)),
                np.zeros(len(las.x)),  # ring_index placeholder
                np.zeros(len(las.x))   # lidar_id placeholder
            ]).T
            
            return points.astype(np.float32)
            
        except ImportError:
            print("Warning: laspy not installed, using dummy LiDAR data")
            return np.zeros((1000, 6), dtype=np.float32)


class AnnotationConverter:
    """Convert Bench2Drive annotations to NavSim format"""
    
    # Mapping from CARLA object types to NavSim categories
    OBJECT_TYPE_MAPPING = {
        'vehicle': 'vehicle',
        'car': 'vehicle.car',
        'truck': 'vehicle.truck',
        'bicycle': 'vehicle.bicycle',
        'motorcycle': 'vehicle.motorcycle',
        'pedestrian': 'human.pedestrian',
        'traffic_light': 'traffic_light',
        'traffic_sign': 'traffic_sign',
    }
    
    @classmethod
    def convert_annotations(cls, b2d_annotations: Dict) -> Dict:
        """Convert Bench2Drive annotations to NavSim format"""
        
        gt_boxes = []
        gt_names = []
        gt_velocity_3d = []
        instance_tokens = []
        track_tokens = []
        
        # Extract actors from Bench2Drive format
        actors = b2d_annotations.get('actors', [])
        
        for actor in actors:
            # Get bounding box
            bbox = actor.get('bounding_box', {})
            location = bbox.get('location', {})
            extent = bbox.get('extent', {})
            
            # Convert to NavSim box format [x, y, z, l, w, h, heading]
            box = [
                location.get('x', 0),
                location.get('y', 0),
                location.get('z', 0),
                2 * extent.get('x', 1),  # length
                2 * extent.get('y', 1),  # width
                2 * extent.get('z', 1),  # height
                np.radians(actor.get('rotation', {}).get('yaw', 0))
            ]
            gt_boxes.append(box)
            
            # Map object type
            carla_type = actor.get('type_id', '').split('.')[0]
            navsim_type = cls.OBJECT_TYPE_MAPPING.get(carla_type, 'unknown')
            gt_names.append(navsim_type)
            
            # Get velocity
            velocity = actor.get('velocity', {})
            gt_velocity_3d.append([
                velocity.get('x', 0),
                velocity.get('y', 0),
                velocity.get('z', 0)
            ])
            
            # Generate tokens
            instance_token = str(uuid.uuid4())
            track_token = actor.get('id', str(uuid.uuid4()))
            instance_tokens.append(instance_token)
            track_tokens.append(track_token)
        
        return {
            'gt_boxes': np.array(gt_boxes, dtype=np.float32),
            'gt_names': gt_names,
            'gt_velocity_3d': np.array(gt_velocity_3d, dtype=np.float32),
            'instance_tokens': instance_tokens,
            'track_tokens': track_tokens
        }


class Bench2DriveToNavSimConverter:
    """Main converter class"""
    
    def __init__(self, b2d_root: Path, output_root: Path, split: str = 'train'):
        self.b2d_root = Path(b2d_root)
        self.output_root = Path(output_root)
        self.split = split
        
        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / 'scenes').mkdir(exist_ok=True)
        (self.output_root / 'sensor_blobs').mkdir(exist_ok=True)
        
        # Initialize helpers
        self.coord_transformer = CoordinateTransformer()
        self.sensor_mapper = SensorDataMapper()
        self.annotation_converter = AnnotationConverter()
    
    def load_b2d_frame(self, scene_path: Path, frame_idx: int) -> Dict:
        """Load a single frame from Bench2Drive dataset"""
        
        # Load annotation
        anno_path = scene_path / 'anno' / f'{frame_idx:05d}.json.gz'
        with gzip.open(anno_path, 'rt') as f:
            anno = json.load(f)
        
        # Collect camera paths
        camera_data = {}
        for cam_type in ['front', 'front_left', 'front_right', 'back', 'back_left', 'back_right']:
            cam_path = scene_path / 'camera' / f'rgb_{cam_type}' / f'{frame_idx:05d}.jpg'
            if cam_path.exists():
                camera_data[f'rgb_{cam_type}'] = str(cam_path)
        
        # LiDAR path
        lidar_path = scene_path / 'lidar' / f'{frame_idx:05d}.laz'
        
        # Radar data (optional)
        radar_path = scene_path / 'radar' / f'{frame_idx:05d}.h5'
        
        return {
            'annotation': anno,
            'camera_data': camera_data,
            'lidar_path': str(lidar_path) if lidar_path.exists() else None,
            'radar_path': str(radar_path) if radar_path.exists() else None,
        }
    
    def convert_scene(self, scene_path: Path) -> List[Dict]:
        """Convert a single Bench2Drive scene to NavSim format"""
        
        frames = []
        scene_name = scene_path.name
        
        # Get all frame indices
        anno_files = sorted((scene_path / 'anno').glob('*.json.gz'))
        
        print(f"Converting scene: {scene_name} with {len(anno_files)} frames")
        
        for frame_file in tqdm(anno_files, desc=f"Processing {scene_name}"):
            frame_idx = int(frame_file.stem)
            
            try:
                # Load frame data
                frame_data = self.load_b2d_frame(scene_path, frame_idx)
                anno = frame_data['annotation']
                
                # Convert ego pose
                ego_transform = anno['ego_vehicle']
                translation, rotation = self.coord_transformer.carla_to_nuplan_pose(ego_transform)
                
                # Convert ego velocity
                ego_velocity = anno['ego_vehicle'].get('velocity', {})
                ego_heading = np.radians(ego_transform['rotation']['yaw'])
                velocity_2d = self.coord_transformer.carla_velocity_to_nuplan(ego_velocity, ego_heading)
                
                # Create frame token
                frame_token = f"{scene_name}_{frame_idx:05d}"
                
                # Convert cameras
                camera_params = anno.get('sensors', {})
                cameras = self.sensor_mapper.map_cameras(frame_data['camera_data'], camera_params)
                
                # Convert LiDAR
                lidar_data = None
                if frame_data['lidar_path']:
                    lidar_data = self.sensor_mapper.convert_lidar_data(frame_data['lidar_path'])
                
                # Convert annotations
                annotations = self.annotation_converter.convert_annotations(anno)
                
                # Extract traffic light states
                traffic_lights = []
                for tl in anno.get('traffic_lights', []):
                    traffic_lights.append({
                        'id': tl.get('id'),
                        'state': tl.get('state', 'unknown'),
                        'position': tl.get('location', {})
                    })
                
                # Build NavSim frame
                navsim_frame = {
                    'token': frame_token,
                    'timestamp': frame_idx / 10.0,  # 10Hz to seconds
                    'scene_token': scene_name,
                    'log_name': f"bench2drive_{scene_name}",
                    'map_location': self._extract_map_name(scene_name),
                    
                    # Ego state
                    'ego2global_translation': translation,
                    'ego2global_rotation': rotation,
                    'ego_dynamic_state': velocity_2d + [0.0, 0.0],  # [vx, vy, ax, ay]
                    
                    # Planning info
                    'driving_command': anno.get('command', 0),
                    'roadblock_ids': [],  # Would need map API
                    'traffic_lights': traffic_lights,
                    
                    # Annotations
                    'anns': annotations,
                    
                    # Sensor references
                    'cams': {cam: f"{frame_token}_{cam}" for cam in cameras.keys()},
                    'lidar_path': f"{frame_token}_lidar" if lidar_data is not None else None,
                }
                
                # Save sensor data
                self._save_sensor_data(frame_token, cameras, lidar_data)
                
                frames.append(navsim_frame)
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
        
        return frames
    
    def _extract_map_name(self, scene_name: str) -> str:
        """Extract map name from scene name"""
        # Scene names are like: Town01_weather_0_route_00000
        parts = scene_name.split('_')
        if parts:
            return parts[0]  # e.g., Town01
        return "unknown"
    
    def _save_sensor_data(self, frame_token: str, cameras: Dict, lidar_data: Optional[np.ndarray]):
        """Save sensor data to disk"""
        
        # Save camera data
        for cam_name, cam_data in cameras.items():
            cam_path = self.output_root / 'sensor_blobs' / f"{frame_token}_{cam_name}.pkl"
            with open(cam_path, 'wb') as f:
                pickle.dump(cam_data, f)
        
        # Save LiDAR data
        if lidar_data is not None:
            lidar_path = self.output_root / 'sensor_blobs' / f"{frame_token}_lidar.pkl"
            with open(lidar_path, 'wb') as f:
                pickle.dump(lidar_data, f)
    
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
        
        all_frames = []
        scene_index = {}
        
        for scene_path in tqdm(scenes, desc="Converting scenes"):
            scene_frames = self.convert_scene(scene_path)
            scene_name = scene_path.name
            
            # Create scene metadata
            scene_meta = {
                'name': scene_name,
                'token': scene_name,
                'log_token': f"bench2drive_{self.split}",
                'frame_tokens': [f['token'] for f in scene_frames],
                'map_name': self._extract_map_name(scene_name),
            }
            
            scene_index[scene_name] = scene_meta
            all_frames.extend(scene_frames)
        
        # Save dataset index
        dataset_meta = {
            'split': self.split,
            'scenes': scene_index,
            'total_frames': len(all_frames),
            'sensor_config': {
                'cameras': list(SensorDataMapper.CAMERA_MAPPING.keys()),
                'lidar_channels': 6,
            }
        }
        
        with open(self.output_root / f'{self.split}_index.pkl', 'wb') as f:
            pickle.dump(dataset_meta, f)
        
        # Save all frames
        with open(self.output_root / f'{self.split}_frames.pkl', 'wb') as f:
            pickle.dump(all_frames, f)
        
        print(f"Conversion complete! Saved {len(all_frames)} frames from {len(scenes)} scenes")
        print(f"Output directory: {self.output_root}")


def main():
    """Main conversion script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Bench2Drive to NavSim format')
    parser.add_argument('--b2d-root', type=str, required=True,
                        help='Path to Bench2Drive dataset root')
    parser.add_argument('--output-root', type=str, required=True,
                        help='Path to output NavSim dataset')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Maximum number of scenes to convert (for testing)')
    
    args = parser.parse_args()
    
    converter = Bench2DriveToNavSimConverter(
        b2d_root=Path(args.b2d_root),
        output_root=Path(args.output_root),
        split=args.split
    )
    
    converter.convert_dataset(max_scenes=args.max_scenes)


if __name__ == '__main__':
    main()