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

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SemanticSegmentationDataset(Dataset):
    """ScanNet++ Stage 1 Dataset for training with SAM masks."""

    def __init__(
        self,
        dataset_name="scannetpp",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannetpp",
        label_db_filepath: Optional[
            str
        ] = "configs/scannetpp_preprocessing/label_database.yaml",
        sam_folder: Optional[str] = "gt_mask",
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
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "unknown task"

        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop

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
        print('SAM folder is', self.sam_folder)

        self.crop_min_size = crop_min_size
        self.crop_length = crop_length

        self.version1 = cropping_v1

        self.random_cuboid = RandomCuboid(
            self.crop_min_size,
            crop_length=self.crop_length,
            version1=self.version1,
        )

        self.mode = mode
        self.data_dir = Path(data_dir)
        self.add_unlabeled_pc = add_unlabeled_pc
        if add_unlabeled_pc:
            matterport_path = self.data_dir.parent / "matterport" / "train_database.yaml"
            if not matterport_path.exists():
                raise FileNotFoundError(f"Matterport database not found at {matterport_path}")
            self.other_database = self._load_yaml(matterport_path)
            
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

        # loading database files
        self._data = []
        self._labels = {0: {'color': [0, 255, 0], 'name': 'object', 'validation': True}}

        # Load split file based on mode
        if self.mode == "train":
            split_file = self.data_dir.parent / 'splits' / 'nvs_sem_train.txt'
        elif self.mode == "validation":
            split_file = self.data_dir.parent / 'splits' / 'nvs_sem_val.txt'
        else:
            split_file = self.data_dir.parent / 'splits' / 'nvs_sem_test.txt'
            
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, "r") as f:
            scenes = [line.strip() for line in f if line.strip()]
        
        if len(scenes) == 0:
            raise ValueError(f"No scenes found in split file: {split_file}")
        
        # For each scene, get all frames
        data_subdir = self.data_dir / 'data'
        if not data_subdir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_subdir}")
            
        for scene in scenes:
            scene_path = data_subdir / scene
            if not scene_path.exists():
                raise FileNotFoundError(f"Scene directory not found: {scene_path}")
            
            # Check for required subdirectories
            rgb_dir = scene_path / 'iphone' / 'rgb'
            depth_dir = scene_path / 'iphone' / 'depth'
            pose_dir = scene_path / 'iphone' / 'pose'
            mask_dir = scene_path / self.sam_folder
            
            for dir_path, dir_name in [(rgb_dir, 'RGB'), (depth_dir, 'Depth'), 
                                       (pose_dir, 'Pose'), (mask_dir, 'SAM mask')]:
                if not dir_path.exists():
                    raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")
            
            # Get all RGB files to determine frames
            rgb_files = sorted([f for f in rgb_dir.glob('*.jpg')])
            if len(rgb_files) == 0:
                raise ValueError(f"No RGB images found in {rgb_dir}")
                
            for rgb_file in rgb_files:
                frame_id = rgb_file.stem
                # Check if all required files exist for this frame
                depth_file = depth_dir / f"{frame_id}.png"
                pose_file = pose_dir / f"{frame_id}.txt"
                mask_file = mask_dir / f"{frame_id}.png"
                
                missing_files = []
                if not depth_file.exists():
                    missing_files.append(f"depth: {depth_file}")
                if not pose_file.exists():
                    missing_files.append(f"pose: {pose_file}")
                if not mask_file.exists():
                    missing_files.append(f"mask: {mask_file}")
                    
                if missing_files:
                    raise FileNotFoundError(f"Missing files for frame {scene}/{frame_id}: " + 
                                          ", ".join(missing_files))
                
                self._data.append(f"{scene} {frame_id}")

        if len(self._data) == 0:
            raise ValueError(f"No valid frames found for mode {self.mode}")
            
        print(f"Loaded {len(self._data)} frames for {self.mode} mode")

        if data_percent < 1.0:
            self._data = sample(
                self._data, int(len(self._data) * data_percent)
            )
        
        # Load intrinsics for all scenes
        self.intrinsics = {}
        for scene in scenes:
            intrinsics_path = data_subdir / scene / 'iphone' / 'intrinsics.txt'
            if not intrinsics_path.exists():
                raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
            self.intrinsics[scene] = np.loadtxt(intrinsics_path)

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
        # new_data = []
        if self.cache_data:
            raise NotImplementedError("Cache data not implemented for ScanNet++")

    def splitPointCloud(self, cloud, size=50.0, stride=50, inner_core=-1):
        if inner_core == -1:
            limitMax = np.amax(cloud[:, 0:3], axis=0)
            width = int(np.ceil((limitMax[0] - size) / stride)) + 1
            depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
            cells = [
                (x * stride, y * stride)
                for x in range(width)
                for y in range(depth)
            ]
            blocks = []
            for (x, y) in cells:
                xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
                ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
                cond = xcond & ycond
                block = cloud[cond, :]
                blocks.append(block)
            return blocks
        else:
            limitMax = np.amax(cloud[:, 0:3], axis=0)
            width = int(np.ceil((limitMax[0] - inner_core) / stride)) + 1
            depth = int(np.ceil((limitMax[1] - inner_core) / stride)) + 1
            cells = [
                (x * stride, y * stride)
                for x in range(width)
                for y in range(depth)
            ]
            blocks_outer = []
            conds_inner = []
            for (x, y) in cells:
                xcond_outer = (
                    cloud[:, 0] <= x + inner_core / 2.0 + size / 2
                ) & (cloud[:, 0] >= x + inner_core / 2.0 - size / 2)
                ycond_outer = (
                    cloud[:, 1] <= y + inner_core / 2.0 + size / 2
                ) & (cloud[:, 1] >= y + inner_core / 2.0 - size / 2)

                cond_outer = xcond_outer & ycond_outer
                block_outer = cloud[cond_outer, :]

                xcond_inner = (block_outer[:, 0] <= x + inner_core) & (
                    block_outer[:, 0] >= x
                )
                ycond_inner = (block_outer[:, 1] <= y + inner_core) & (
                    block_outer[:, 1] >= y
                )

                cond_inner = xcond_inner & ycond_inner

                conds_inner.append(cond_inner)
                blocks_outer.append(block_outer)
            return conds_inner, blocks_outer

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
            return 5 * len(self.data)
        else:
            return self.reps_per_epoch * len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)

        fname = self.data[idx]
        scene_id, image_id = fname.split()
        
        # Build paths
        base_path = self.data_dir / 'data' / scene_id
        color_path = base_path / 'iphone' / 'rgb' / f'{image_id}.jpg'
        depth_path = base_path / 'iphone' / 'depth' / f'{image_id}.png'
        pose_path = base_path / 'iphone' / 'pose' / f'{image_id}.txt'
        sam_path = base_path / self.sam_folder / f'{image_id}.png'
        
        # Load RGB
        color_image = cv2.imread(str(color_path))
        if color_image is None:
            raise FileNotFoundError(f"Failed to load RGB image: {color_path}")
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Load depth
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth_image is None:
            raise FileNotFoundError(f"Failed to load depth image: {depth_path}")

        # Load pose
        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4):
            raise ValueError(f"Invalid pose shape {pose.shape} for {pose_path}")

        # Load intrinsics
        depth_intrinsic = self.intrinsics[scene_id]

        # Load SAM masks
        with open(sam_path, 'rb') as image_file:
            img = Image.open(image_file)
            sam_groups = np.array(img, dtype=np.int16)

        # Process point cloud
        mask = (depth_image != 0)
        colors = color_image[mask]
        sam_groups = sam_groups[mask]

        # Depth to 3D conversion
        depth_shift = 1000.0
        y, x = np.where(mask)
        z = depth_image[mask] / depth_shift
        
        fx = depth_intrinsic[0, 0]
        fy = depth_intrinsic[1, 1]
        cx = depth_intrinsic[0, 2]
        cy = depth_intrinsic[1, 2]
        
        # Convert to camera coordinates
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        
        # Create homogeneous coordinates
        points = np.ones((len(X), 4))
        points[:, 0] = X
        points[:, 1] = Y
        points[:, 2] = z
        
        # Transform to world coordinates
        points_world = np.dot(points, pose.T)
        
        # Process instance masks
        sam_groups = self.num_to_natural(sam_groups)
        
        # Filter small instances
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
                points = points[new_idx]

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
                (
                    coordinates,
                    color,
                    normals,
                    labels,
                ) = self.augment_individual_instance(
                    coordinates,
                    color,
                    normals,
                    labels,
                    self.instance_oversampling,
                )

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

            if self.add_unlabeled_pc:
                if random() < 0.8:
                    new_points = np.load(
                        self.other_database[
                            np.random.randint(0, len(self.other_database) - 1)
                        ]["filepath"]
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        new_points[:, :3],
                        new_points[:, 3:6],
                        new_points[:, 6:9],
                        new_points[:, 9:],
                    )
                    unlabeled_coords -= unlabeled_coords.mean(0)
                    unlabeled_coords += (
                        np.random.uniform(
                            unlabeled_coords.min(0), unlabeled_coords.max(0)
                        )
                        / 2
                    )

                    aug = self.volume_augmentations(
                        points=unlabeled_coords,
                        normals=unlabeled_normals,
                        features=unlabeled_color,
                        labels=unlabeled_labels,
                    )
                    (
                        unlabeled_coords,
                        unlabeled_color,
                        unlabeled_normals,
                        unlabeled_labels,
                    ) = (
                        aug["points"],
                        aug["features"],
                        aug["normals"],
                        aug["labels"],
                    )
                    pseudo_image = unlabeled_color.astype(np.uint8)[
                        np.newaxis, :, :
                    ]
                    unlabeled_color = np.squeeze(
                        self.image_augmentations(image=pseudo_image)["image"]
                    )

                    coordinates = np.concatenate(
                        (coordinates, unlabeled_coords)
                    )
                    color = np.concatenate((color, unlabeled_color))
                    normals = np.concatenate((normals, unlabeled_normals))
                    labels = np.concatenate(
                        (
                            labels,
                            np.full_like(unlabeled_labels, self.ignore_label),
                        )
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
            f'{scene_id}/{image_id}',
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
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
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

    def augment_individual_instance(
        self, coordinates, color, normals, labels, oversampling=1.0
    ):
        if self.instance_oversampling == 0:
            return coordinates, color, normals, labels
            
        max_instance = int(len(np.unique(labels[:, 1])))
        # randomly selecting half of non-zero instances
        for instance in range(0, int(max_instance * oversampling)):
            if self.place_around_existing:
                center = choice(
                    coordinates[
                        labels[:, 1] == choice(np.unique(labels[:, 1]))
                    ]
                )
            else:
                center = np.array(
                    [uniform(-5, 5), uniform(-5, 5), uniform(-0.5, 2)]
                )
            
            if not hasattr(self, 'instance_data'):
                raise AttributeError("Instance oversampling enabled but no instance_data loaded")
                
            instance = choice(choice(self.instance_data))
            instance = np.load(instance["instance_filepath"])
            # centering two objects
            instance[:, :3] = (
                instance[:, :3] - instance[:, :3].mean(axis=0) + center
            )
            max