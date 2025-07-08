import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange
import os
import numpy
import torch
from datasets.random_cuboid import RandomCuboid
from PIL import Image
from collections import Counter
import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml
import cv2
import copy
import json
import re

# Import the provided utilities
from thes.paths import iterate_scannetpp, scannetpp_raw_dir, scannetpp_processed_dir

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

pointcloud_dims = (256, 192)


class ScannetppStage1Dataset(Dataset):
    """ScanNet++ Stage1 Dataset for training on 2D SAM masks."""

    def __init__(
        self,
        dataset_name="scannetpp",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannetpp",
        label_db_filepath: Optional[
            str
        ] = "data/processed/scannetpp/label_database.yaml",
        sam_folder: Optional[str] = "gt_mask",
        scenes_to_exclude: str = "",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        filter_out_classes=[],
        label_offset=0,
        add_clip=False,
        is_elastic_distortion=True,
        color_drop=0.0,
        max_frames: Optional[int] = None,
    ):
        self.excluded_scenes = set()
        if scenes_to_exclude:
            self.excluded_scenes.update(scene.strip() for scene in scenes_to_exclude.split(',') if scene.strip())
        
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "unknown task"

        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop
        self.max_frames = max_frames

        if self.dataset_name == "scannetpp":
            self.color_map = {0: [0, 255, 0]}
        else:
            assert False, "dataset not known"

        self.task = task

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        self.area = area
        self.eval_inner_core = eval_inner_core

        self.reps_per_epoch = reps_per_epoch

        self.cropping = cropping
        self.cropping_args = cropping_args
        self.is_tta = is_tta
        self.on_crops = on_crops
        self.sam_folder = sam_folder

        self.crop_min_size = crop_min_size
        self.crop_length = crop_length

        self.version1 = cropping_v1

        self.random_cuboid = RandomCuboid(
            self.crop_min_size,
            crop_length=self.crop_length,
            version1=self.version1,
        )

        self.mode = mode
        self.data_dir = data_dir
        self.add_unlabeled_pc = add_unlabeled_pc
        if add_unlabeled_pc:
            raise NotImplementedError("add_unlabeled_pc not implemented for ScanNet++")
            
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points

        # Discover scene/frame pairs using iterate_scannetpp
        self._data = self._discover_scene_frame_pairs()
        
        # Load labels (simplified for binary classification)
        self._labels = {0: {'color': [0, 255, 0], 'name': 'object', 'validation': True}}

        # Load scene metadata (intrinsics and poses)
        self.scene_metadata = self._load_scene_metadata()
        
        if data_percent < 1.0:
            self._data = sample(
                self._data, int(len(self._data) * data_percent)
            )

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color augmentation
        if add_colors:
            # use imagenet stats
            color_mean = (0.485, 0.456, 0.406)
            color_std = (0.229, 0.224, 0.225)
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.cache_data = cache_data

    def get_raw_sample(self, idx):
        original_mode = self.mode
        self.mode = "validation"
        try:
            return self.__getitem__(idx)
        finally:
            self.mode = original_mode

    def _discover_scene_frame_pairs(self):
        pairs = []
        total_rgb_count = 0
        scene_data = []
        
        split_map = {"train": "train", "validation": "val", "val": "val"}
        split = split_map.get(self.mode, self.mode)
        
        for raw_scene_path, processed_scene_path in iterate_scannetpp(split):
            scene_id = raw_scene_path.name
            
            if scene_id in self.excluded_scenes:
                continue
            
            rgb_dir = processed_scene_path / "iphone" / "rgb"
            if not rgb_dir.exists():
                raise FileNotFoundError(f"RGB directory missing: {rgb_dir}")
                
            rgb_frames = [f.stem for f in rgb_dir.glob("*.jpg")] + [f.stem for f in rgb_dir.glob("*.png")]
            if not rgb_frames:
                raise FileNotFoundError(f"No RGB frames found: {rgb_dir}")
                
            total_rgb_count += len(rgb_frames)
            scene_data.append((scene_id, processed_scene_path, rgb_frames))
        
        if total_rgb_count == 0:
            raise ValueError(f"No RGB frames found for split '{self.mode}'")
        
        if self.max_frames and self.max_frames < total_rgb_count:
            sample_rate = self.max_frames / total_rgb_count
            print(f"Sampling: {total_rgb_count} -> {self.max_frames} frames (rate={sample_rate:.3f})")
            
            for scene_id, processed_path, rgb_frames in scene_data:
                n_samples = max(1, int(len(rgb_frames) * sample_rate))
                
                frame_data = [(f, self._extract_frame_number(f)) for f in rgb_frames]
                frame_data.sort(key=lambda x: x[1])
                
                indices = np.linspace(0, len(frame_data) - 1, n_samples, dtype=int)
                selected = [frame_data[i][0] for i in indices]
                
                self._validate_frames(scene_id, processed_path, selected)
                pairs.extend([f"{scene_id} {f}" for f in selected])
        else:
            for scene_id, processed_path, rgb_frames in scene_data:
                depth_dir = processed_path / "iphone" / "depth"
                mask_dir = processed_path / self.sam_folder
                
                if not depth_dir.exists():
                    raise FileNotFoundError(f"Depth directory missing: {depth_dir}")
                if not mask_dir.exists():
                    raise FileNotFoundError(f"Mask directory missing: {mask_dir}")
                
                depth_frames = {f.stem for f in depth_dir.glob("*.png")}
                mask_frames = {f.stem for f in mask_dir.glob("*.png")}
                valid_frames = set(rgb_frames) & depth_frames & mask_frames
                
                if not valid_frames:
                    raise FileNotFoundError(f"No matching frames for scene {scene_id}")
                
                valid_frame_data = [(f, self._extract_frame_number(f)) for f in valid_frames]
                valid_frame_data.sort(key=lambda x: x[1])
                pairs.extend([f"{scene_id} {f[0]}" for f in valid_frame_data])
        
        print(f"Found {len(pairs)} valid scene/frame pairs")
        return pairs

    def _extract_frame_number(self, frame_id):
        match = re.search(r'frame_(\d+)', frame_id)
        return int(match.group(1)) if match else 0

    def _validate_frames(self, scene_id, scene_path, frame_ids):
        depth_dir = scene_path / "iphone" / "depth"
        mask_dir = scene_path / self.sam_folder
        
        if not depth_dir.exists():
            raise FileNotFoundError(f"Depth directory missing: {depth_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory missing: {mask_dir}")
        
        for frame_id in frame_ids:
            if not (depth_dir / f"{frame_id}.png").exists():
                raise FileNotFoundError(f"Missing depth: {scene_id}/{frame_id}")
            if not (mask_dir / f"{frame_id}.png").exists():
                raise FileNotFoundError(f"Missing mask: {scene_id}/{frame_id}")

    def _load_scene_metadata(self):
        """Load intrinsics and poses for all scenes"""
        metadata = {}
        
        split_map = {"train": "train", "validation": "val", "val": "val"}
        split = split_map.get(self.mode, self.mode)
        
        for raw_scene_path, _ in iterate_scannetpp(split):
            scene_id = raw_scene_path.name
            
            # Skip excluded scenes
            if scene_id in self.excluded_scenes:
                continue
            
            # Check if raw scene path exists
            if not raw_scene_path.exists():
                raise FileNotFoundError(f"Raw scene directory not found: {raw_scene_path}")
            
            # Load pose_intrinsic_imu.json from raw directory
            pose_intrinsic_file = raw_scene_path / "iphone" / "pose_intrinsic_imu.json"
            
            if not pose_intrinsic_file.exists():
                raise FileNotFoundError(f"Missing pose_intrinsic_imu.json for scene {scene_id}: {pose_intrinsic_file}")
            
            try:
                with open(pose_intrinsic_file, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in pose_intrinsic_imu.json for scene {scene_id}: {e}")
            except Exception as e:
                raise IOError(f"Error reading pose_intrinsic_imu.json for scene {scene_id}: {e}")
            
            # Extract frame data - each frame has its own intrinsic and pose
            frame_to_pose = {}
            frame_to_intrinsic = {}
            
            try:
                for frame_key, frame_data in data.items():
                    if not isinstance(frame_data, dict):
                        continue
                    
                    frame_id = frame_key
                    
                    # Extract intrinsic matrix for this frame
                    if "intrinsic" in frame_data:
                        intrinsic_matrix = np.array(frame_data["intrinsic"])
                        if intrinsic_matrix.shape != (3, 3):
                            raise ValueError(f"Intrinsics matrix should be 3x3, got {intrinsic_matrix.shape} for scene {scene_id}, frame {frame_id}")
                        frame_to_intrinsic[frame_id] = intrinsic_matrix
                    
                    # Extract pose matrix for this frame (prefer aligned_pose if available)
                    pose_matrix = None
                    if "aligned_pose" in frame_data:
                        pose_matrix = np.array(frame_data["aligned_pose"])
                    elif "pose" in frame_data:
                        pose_matrix = np.array(frame_data["pose"])
                    
                    if pose_matrix is not None:
                        if pose_matrix.shape == (4, 4):
                            frame_to_pose[frame_id] = pose_matrix
                        elif pose_matrix.shape == (16,):
                            frame_to_pose[frame_id] = pose_matrix.reshape(4, 4)
                        else:
                            raise ValueError(f"Invalid pose shape {pose_matrix.shape} for scene {scene_id}, frame {frame_id}")
                            
            except Exception as e:
                raise ValueError(f"Error parsing frame data for scene {scene_id}: {e}")
            
            if not frame_to_pose:
                raise ValueError(f"No valid poses found for scene {scene_id}")
            
            if not frame_to_intrinsic:
                raise ValueError(f"No valid intrinsics found for scene {scene_id}")
            
            # Store both poses and intrinsics per frame
            metadata[scene_id] = {
                "frame_to_pose": frame_to_pose,
                "frame_to_intrinsic": frame_to_intrinsic
            }
            
        if not metadata:
            raise ValueError(f"No scene metadata loaded for split '{self.mode}'")
            
        return metadata

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)
    
    def num_to_natural(self, group_ids):
        '''
        Change the group number to natural number arrangement
        '''
        if np.all(group_ids == -1):
            return group_ids
        array = copy.deepcopy(group_ids)
        unique_values = np.unique(array[array != -1])
        mapping = np.full(np.max(unique_values) + 2, -1)
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]
        return array

    def __len__(self):
        if self.is_tta:
            return 5 * len(self._data)
        else:
            return self.reps_per_epoch * len(self._data)

    def __getitem__(self, idx: int):
        idx = idx % len(self._data)
        if self.is_tta:
            idx = idx % len(self._data)

        fname = self._data[idx]
        scene_id, frame_id = fname.split()
        
        # Get paths for this scene
        raw_scene_path = scannetpp_raw_dir / "data" / scene_id
        processed_scene_path = scannetpp_processed_dir / "data" / scene_id
        
        # Verify scene directories exist
        if not raw_scene_path.exists():
            raise FileNotFoundError(f"Raw scene directory not found: {raw_scene_path}")
        if not processed_scene_path.exists():
            raise FileNotFoundError(f"Processed scene directory not found: {processed_scene_path}")
        
        # Load RGB
        color_path = processed_scene_path / "iphone" / "rgb" / f"{frame_id}.jpg"
        if not color_path.exists():
            color_path = processed_scene_path / "iphone" / "rgb" / f"{frame_id}.png"
            
        if not color_path.exists():
            raise FileNotFoundError(f"RGB frame not found for scene {scene_id}, frame {frame_id}. Checked: {processed_scene_path / 'iphone' / 'rgb' / frame_id}.jpg and .png")
        
        try:
            with Image.open(color_path) as img:
                color_image = np.array(img.convert('RGB'))
            color_image = cv2.resize(color_image, pointcloud_dims)
        except Exception as e:
            raise IOError(f"Error loading RGB image {color_path}: {e}")

        # Load Depth - critical file that must exist and be valid
        depth_path = processed_scene_path / "iphone" / "depth" / f"{frame_id}.png"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth image not found for scene {scene_id}, frame {frame_id}: {depth_path}")
        
        try:
            with Image.open(depth_path) as img:
                depth_image = np.array(img, dtype=np.uint16)
            
            if depth_image.size == 0:
                raise ValueError(f"Depth image is empty: {depth_path}")
            
            depth_image = cv2.resize(depth_image, pointcloud_dims, interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            raise IOError(f"Error loading depth image {depth_path}: {e}")

        # Get pose from metadata
        if scene_id not in self.scene_metadata:
            raise FileNotFoundError(f"No metadata loaded for scene {scene_id}")
            
        scene_meta = self.scene_metadata[scene_id]
        
        if frame_id not in scene_meta["frame_to_pose"]:
            available_frames = list(scene_meta["frame_to_pose"].keys())[:10]
            raise FileNotFoundError(f"No pose found for scene {scene_id}, frame {frame_id}. Available frames (first 10): {available_frames}")
        
        pose = scene_meta["frame_to_pose"][frame_id]

        # Validate pose
        if pose.shape != (4, 4):
            raise ValueError(f"Invalid pose shape {pose.shape} for scene {scene_id}, frame {frame_id}. Expected (4, 4)")

        if frame_id not in scene_meta["frame_to_intrinsic"]:
            available_frames = list(scene_meta["frame_to_intrinsic"].keys())[:10]
            raise FileNotFoundError(f"No intrinsic found for scene {scene_id}, frame {frame_id}. Available frames (first 10): {available_frames}")
        
        depth_intrinsic = scene_meta["frame_to_intrinsic"][frame_id].copy()
        
        # Scale intrinsics from original resolution (1920x1440) to pointcloud_dims (256x192)
        scale_x = pointcloud_dims[0] / 1920.0
        scale_y = pointcloud_dims[1] / 1440.0
        depth_intrinsic[0, 0] *= scale_x
        depth_intrinsic[1, 1] *= scale_y
        depth_intrinsic[0, 2] *= scale_x
        depth_intrinsic[1, 2] *= scale_y

        # Load SAM mask
        sam_path = processed_scene_path / self.sam_folder / f"{frame_id}.png"
        if not sam_path.exists():
            raise FileNotFoundError(f"SAM mask not found for scene {scene_id}, frame {frame_id}: {sam_path}")
        
        try:
            with open(sam_path, 'rb') as image_file:
                img = Image.open(image_file)
                sam_groups = np.array(img, dtype=np.int16)
                
            if sam_groups.size == 0:
                raise ValueError(f"SAM mask is empty: {sam_path}")
            
            sam_groups = cv2.resize(sam_groups, pointcloud_dims, interpolation=cv2.INTER_NEAREST)
                
        except Exception as e:
            raise IOError(f"Error loading SAM mask {sam_path}: {e}")

        # Validate depth mask and check for valid depth values
        mask = (depth_image != 0)
        if not np.any(mask):
            raise ValueError(f"Depth image contains no valid depth values (all zeros): {depth_path}")
        
        try:
            colors = np.reshape(color_image[mask], [-1, 3])
            sam_groups = sam_groups[mask]
        except Exception as e:
            raise ValueError(f"Error applying depth mask for scene {scene_id}, frame {frame_id}: {e}")

        depth_shift = 1000.0
        x, y = np.meshgrid(
            np.linspace(0, depth_image.shape[1] - 1, depth_image.shape[1]), 
            np.linspace(0, depth_image.shape[0] - 1, depth_image.shape[0])
        )
        uv_depth = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
        uv_depth[:, :, 0] = x
        uv_depth[:, :, 1] = y
        uv_depth[:, :, 2] = depth_image / depth_shift
        uv_depth = np.reshape(uv_depth, [-1, 3])
        uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()
        
        if uv_depth.size == 0:
            raise ValueError(f"No valid depth points after filtering for scene {scene_id}, frame {frame_id}")
        
        fx = depth_intrinsic[0, 0]
        fy = depth_intrinsic[1, 1]
        cx = depth_intrinsic[0, 2]
        cy = depth_intrinsic[1, 2]
        
        n = uv_depth.shape[0]
        points = np.ones((n, 4))
        X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx
        Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy
        points[:, 0] = X
        points[:, 1] = Y
        points[:, 2] = uv_depth[:, 2]
        
        try:
            points_world = np.dot(points, np.transpose(pose))
        except Exception as e:
            raise ValueError(f"Error transforming points to world coordinates for scene {scene_id}, frame {frame_id}: {e}")
            
        sam_groups = self.num_to_natural(sam_groups)

        counts = Counter(sam_groups)
        for num, count in counts.items():
            if count < 100:
                sam_groups[sam_groups == num] = -1
        sam_groups = self.num_to_natural(sam_groups)

        coordinates = points_world[:, :3]
        color = colors
        normals = np.ones_like(coordinates)
        segments = np.ones(coordinates.shape[0])
        labels = np.concatenate([np.zeros(coordinates.shape[0]).reshape(-1, 1), sam_groups.reshape(-1, 1)], axis=1)

        # Validate final point cloud
        if coordinates.shape[0] == 0:
            raise ValueError(f"No valid points generated for scene {scene_id}, frame {frame_id}")

        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # volume and image augmentations for train
        if "train" in self.mode or self.is_tta:
            if self.cropping:
                new_idx = self.random_cuboid(
                    coordinates,
                    labels[:, 1],
                    self._remap_from_zero(labels[:, 0].copy()),
                )
                coordinates = coordinates[new_idx]
                color = color[new_idx]
                labels = labels[new_idx]
                segments = segments[new_idx]
                raw_color = raw_color[new_idx]
                raw_normals = raw_normals[new_idx]
                normals = normals[new_idx]

            coordinates -= coordinates.mean(0)
            try:
                coordinates += (
                    np.random.uniform(coordinates.min(0), coordinates.max(0))
                    / 2
                )
            except OverflowError as err:
                print(coordinates)
                print(coordinates.shape)
                raise err

            if self.instance_oversampling > 0.0:
                raise NotImplementedError("Instance oversampling not implemented for ScanNet++")

            if self.flip_in_center:
                coordinates = flip_in_center(coordinates)

            for i in (0, 1):
                if random() < 0.5:
                    coord_max = np.max(coordinates[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]

            if random() < 0.95:
                if self.is_elastic_distortion:
                    for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )
            aug = self.volume_augmentations(
                points=coordinates,
                normals=normals,
                features=color,
                labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(
                self.image_augmentations(image=pseudo_image)["image"]
            )

            if self.point_per_cut != 0:
                number_of_cuts = int(len(coordinates) / self.point_per_cut)
                for _ in range(number_of_cuts):
                    size_of_cut = np.random.uniform(0.05, self.max_cut_region)
                    # not wall, floor or empty
                    point = choice(coordinates)
                    x_min = point[0] - size_of_cut
                    x_max = x_min + size_of_cut
                    y_min = point[1] - size_of_cut
                    y_max = y_min + size_of_cut
                    z_min = point[2] - size_of_cut
                    z_max = z_min + size_of_cut
                    indexes = crop(
                        coordinates, x_min, y_min, z_min, x_max, y_max, z_max
                    )
                    coordinates, normals, color, labels = (
                        coordinates[~indexes],
                        normals[~indexes],
                        color[~indexes],
                        labels[~indexes],
                    )

            if (self.resample_points > 0) or (self.noise_rate > 0):
                coordinates, color, normals, labels = random_around_points(
                    coordinates,
                    color,
                    normals,
                    labels,
                    self.resample_points,
                    self.noise_rate,
                    self.ignore_label,
                )

            if random() < self.color_drop:
                color[:] = 255

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])
        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))

        return (
            coordinates,
            features,
            labels,
            f'{scene_id}/{frame_id}',
            raw_color,
            raw_normals,
            raw_coordinates,
            idx,
        )

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.load(f, Loader=yaml.SafeLoader)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        labels[
            ~np.isin(labels, list(self.label_info.keys()))
        ] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(
            noise, blurx, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blury, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blurz, mode="constant", cval=0
        )

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def flip_in_center(coordinates):
    # moving coordinates to center
    coordinates -= coordinates.mean(0)
    aug = V.Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    # x -y
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    # -x y
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])[
            "points"
        ]
        coordinates[second_crop] += minimum
    if third_crop.size > 1:
        minimum = coordinates[third_crop].min(0)
        minimum[2] = 0
        minimum[1] = 0
        coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
        coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        minimum = coordinates[fourth_crop].min(0)
        minimum[2] = 0
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])[
            "points"
        ]
        coordinates[fourth_crop] += minimum

    return coordinates


def random_around_points(
    coordinates,
    color,
    normals,
    labels,
    rate=0.2,
    noise_rate=0,
    ignore_label=255,
):
    coord_indexes = sample(
        list(range(len(coordinates))), k=int(len(coordinates) * rate)
    )
    noisy_coordinates = deepcopy(coordinates[coord_indexes])
    noisy_coordinates += np.random.uniform(
        -0.2 - noise_rate, 0.2 + noise_rate, size=noisy_coordinates.shape
    )

    if noise_rate > 0:
        noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
        noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
        noisy_labels = np.full(labels[coord_indexes].shape, ignore_label)

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))
    else:
        noisy_color = deepcopy(color[coord_indexes])
        noisy_normals = deepcopy(normals[coord_indexes])
        noisy_labels = deepcopy(labels[coord_indexes])

        coordinates = np.vstack((coordinates, noisy_coordinates))
        color = np.vstack((color, noisy_color))
        normals = np.vstack((normals, noisy_normals))
        labels = np.vstack((labels, noisy_labels))

    return coordinates, color, normals, labels


def random_points(
    coordinates, color, normals, labels, noise_rate=0.6, ignore_label=255
):
    max_boundary = coordinates.max(0) + 0.1
    min_boundary = coordinates.min(0) - 0.1

    noisy_coordinates = int(
        (max(max_boundary) - min(min_boundary)) / noise_rate
    )

    noisy_coordinates = np.array(
        list(
            product(
                np.linspace(
                    min_boundary[0], max_boundary[0], noisy_coordinates
                ),
                np.linspace(
                    min_boundary[1], max_boundary[1], noisy_coordinates
                ),
                np.linspace(
                    min_boundary[2], max_boundary[2], noisy_coordinates
                ),
            )
        )
    )
    noisy_coordinates += np.random.uniform(
        -noise_rate, noise_rate, size=noisy_coordinates.shape
    )

    noisy_color = np.random.randint(0, 255, size=noisy_coordinates.shape)
    noisy_normals = np.random.rand(*noisy_coordinates.shape) * 2 - 1
    noisy_labels = np.full(
        (noisy_coordinates.shape[0], labels.shape[1]), ignore_label
    )

    coordinates = np.vstack((coordinates, noisy_coordinates))
    color = np.vstack((color, noisy_color))
    normals = np.vstack((normals, noisy_normals))
    labels = np.vstack((labels, noisy_labels))
    return coordinates, color, normals, labels