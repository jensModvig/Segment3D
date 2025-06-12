import logging
import os
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
import cv2
import numpy as np
import torch
import yaml
import albumentations as A
import scipy
import volumentations as V
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset
from datasets.random_cuboid import RandomCuboid

logger = logging.getLogger(__name__)

# You can change this name to match your custom dataset
sam_folder = 'sam_masks'  # Folder where your SAM masks are stored

class CustomScannetppStage1Dataset(Dataset):
    """Dataset for training Stage 1 with custom data similar to ScanNet++."""

    def __init__(
        self,
        dataset_name="custom_scannetpp",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/custom_scannetpp",
        label_db_filepath: Optional[str] = "configs/custom_scannetpp/label_database.yaml",
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
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
        self.color_map = {0: [0, 255, 0]}  # Default color map for instances
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
            self.other_database = self._load_yaml(
                Path(data_dir).parent / "matterport" / "train_database.yaml"
            )
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

        # Loading scene and frame information
        self._data = []
        
        # ===== PLACEHOLDER: DATA LOADING =====
        # Replace this with your own data loading logic
        # This should load a list of scenes and image IDs to process
        if self.mode == "train":
            data_path = f'data/custom_scannetpp/splits/scannetpp_train.txt'
        else:
            data_path = f'data/custom_scannetpp/splits/scannetpp_val.txt'
            
        if os.path.exists(data_path):
            with open(data_path, "r") as scene_file:
                self._data = scene_file.read().splitlines()
        else:
            # Example fallback - you should replace this with your own data
            print(f"Warning: {data_path} not found. Using example data.")
            self._data = ["scene0000/0000", "scene0000/0001"] 
        # ===== END PLACEHOLDER =====

        if data_percent < 1.0:
            self._data = sample(
                self._data, int(len(self._data) * data_percent)
            )
        
        # Load intrinsics
        # ===== PLACEHOLDER: INTRINSICS LOADING =====
        # Replace this with your own camera intrinsics loading logic
        intrinsics_path = os.path.join(self.data_dir, 'intrinsics', 'intrinsics.txt')
        if os.path.exists(intrinsics_path):
            self.depth_intrinsic = np.loadtxt(intrinsics_path)
        else:
            # Example fallback intrinsics - replace with your actual camera parameters
            print(f"Warning: {intrinsics_path} not found. Using example intrinsics.")
            self.depth_intrinsic = np.array([
                [577.870, 0, 319.5, 0],
                [0, 577.870, 239.5, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        # ===== END PLACEHOLDER =====

        # Label info
        self._labels = {0: {'color': [0, 255, 0], 'name': 'object', 'validation': True}}

        # Augmentations setup
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
            
        # Color normalization
        if add_colors:
            color_mean = (0.485, 0.456, 0.406)
            color_std = (0.229, 0.224, 0.225)
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.cache_data = cache_data

    def num_to_natural(self, group_ids):
        '''
        Change the group number to natural number arrangement
        '''
        if np.all(group_ids == -1):
            return group_ids
        array = deepcopy(group_ids)
        unique_values = np.unique(array[array != -1])
        mapping = np.full(np.max(unique_values) + 2, -1)
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]
        return array

    def map2color(self, labels):
        output_colors = list()
        for label in labels:
            output_colors.append(self.color_map[label])
        return torch.tensor(output_colors)

    def __len__(self):
        if self.is_tta:
            return 5 * len(self.data)
        else:
            return self.reps_per_epoch * len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)

        # Get scene and frame ID
        scene_id, image_id = self.data[idx].split('/')
        
        # ===== PLACEHOLDER: LOAD RGB, DEPTH, POSE, and SAM MASKS =====
        # Replace these paths with your own data paths
        color_path = os.path.join(self.data_dir, scene_id, 'color', f'{image_id}.jpg')
        depth_path = os.path.join(self.data_dir, scene_id, 'depth', f'{image_id}.png')
        pose_path = os.path.join(self.data_dir, scene_id, 'pose', f'{image_id}.txt')
        sam_mask_path = os.path.join(self.data_dir, scene_id, sam_folder, f'{image_id}.png')
        
        # Load RGB image
        if os.path.exists(color_path):
            color_image = cv2.imread(color_path)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            color_image = cv2.resize(color_image, (640, 480))
        else:
            # Generate fake image if not found
            print(f"Warning: {color_path} not found. Using random image.")
            color_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Load depth image
        if os.path.exists(depth_path):
            depth_image = cv2.imread(depth_path, -1)
        else:
            # Generate fake depth if not found
            print(f"Warning: {depth_path} not found. Using random depth.")
            depth_image = np.random.randint(0, 5000, (480, 640), dtype=np.uint16)
        
        # Load camera pose
        if os.path.exists(pose_path):
            pose = np.loadtxt(pose_path)
        else:
            # Generate identity pose if not found
            print(f"Warning: {pose_path} not found. Using identity pose.")
            pose = np.eye(4)
        
        # Load SAM masks
        if os.path.exists(sam_mask_path):
            with open(sam_mask_path, 'rb') as image_file:
                img = Image.open(image_file)
                sam_groups = np.array(img, dtype=np.int16)
        else:
            # Generate empty masks if not found
            print(f"Warning: {sam_mask_path} not found. Using empty masks.")
            sam_groups = np.zeros((480, 640), dtype=np.int16) - 1
        # ===== END PLACEHOLDER =====

        # Create a mask for valid depth values
        mask = (depth_image != 0)
        colors = np.reshape(color_image[mask], [-1, 3])
        
        # Extract SAM groups for valid depth points
        sam_groups = sam_groups[mask]
        
        # Convert to natural numbers (consecutive instance IDs)
        sam_groups = self.num_to_natural(sam_groups)
        
        # Filter out instances with too few points (less than 100)
        counts = Counter(sam_groups)
        for num, count in counts.items():
            if count < 100:
                sam_groups[sam_groups == num] = -1
        sam_groups = self.num_to_natural(sam_groups)

        # Convert depth to 3D points
        # ===== PLACEHOLDER: DEPTH TO POINT CLOUD CONVERSION =====
        # This is a standard depth-to-pointcloud conversion, but you might need
        # to adjust it based on your specific depth format and camera intrinsics
        depth_shift = 1000.0  # Adjust based on your depth units
        x, y = np.meshgrid(
            np.linspace(0, depth_image.shape[1]-1, depth_image.shape[1]),
            np.linspace(0, depth_image.shape[0]-1, depth_image.shape[0])
        )
        uv_depth = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
        uv_depth[:,:,0] = x
        uv_depth[:,:,1] = y
        uv_depth[:,:,2] = depth_image / depth_shift
        uv_depth = np.reshape(uv_depth, [-1, 3])
        uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
        
        # Extract camera parameters
        fx = self.depth_intrinsic[0,0]
        fy = self.depth_intrinsic[1,1]
        cx = self.depth_intrinsic[0,2]
        cy = self.depth_intrinsic[1,2]
        bx = self.depth_intrinsic[0,3] if self.depth_intrinsic.shape[1] > 3 else 0
        by = self.depth_intrinsic[1,3] if self.depth_intrinsic.shape[1] > 3 else 0
        
        # Project to 3D
        n = uv_depth.shape[0]
        points = np.ones((n, 4))
        X = (uv_depth[:,0] - cx) * uv_depth[:,2] / fx + bx
        Y = (uv_depth[:,1] - cy) * uv_depth[:,2] / fy + by
        points[:,0] = X
        points[:,1] = Y
        points[:,2] = uv_depth[:,2]
        
        # Transform to world coordinates using camera pose
        points_world = np.dot(points, np.transpose(pose))
        # ===== END PLACEHOLDER =====

        # Prepare data arrays
        coordinates = points_world[:,:3]
        color = colors
        normals = np.ones_like(coordinates)  # Default normals pointing up
        segments = np.ones(coordinates.shape[0])
        labels = np.concatenate([
            np.zeros(coordinates.shape[0]).reshape(-1, 1),  # Semantic labels (all 0 for instance segmentation)
            sam_groups.reshape(-1, 1)                      # Instance labels from SAM
        ], axis=1)

        # Save original data
        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        # If colors not needed, use ones
        if not self.add_colors:
            color = np.ones((len(color), 3))

        # Apply training augmentations
        if "train" in self.mode or self.is_tta:
            # Random cuboid cropping
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
            
            # Center the point cloud
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

            # Instance oversampling if enabled
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

            # Flip in center if enabled
            if self.flip_in_center:
                coordinates = flip_in_center(coordinates)

            # Random flipping
            for i in (0, 1):
                if random() < 0.5:
                    coord_max = np.max(coordinates[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]

            # Elastic distortion
            if random() < 0.95:
                if self.is_elastic_distortion:
                    for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )
            
            # Volume augmentations
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
            
            # Image augmentations
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(
                self.image_augmentations(image=pseudo_image)["image"]
            )

            # Random region cutting
            if self.point_per_cut != 0:
                number_of_cuts = int(len(coordinates) / self.point_per_cut)
                for _ in range(number_of_cuts):
                    size_of_cut = np.random.uniform(0.05, self.max_cut_region)
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

            # Point resampling or noise
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

            # Color drop
            if random() < self.color_drop:
                color[:] = 255

        # Color normalization
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])
        
        # Prepare final labels and features
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
        """database file containing information about preprocessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

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
            instance = choice(choice(self.instance_data))
            instance = np.load(instance["instance_filepath"])
            # centering two objects
            instance[:, :3] = (
                instance[:, :3] - instance[:, :3].mean(axis=0) + center
            )
            max_instance = max_instance + 1
            instance[:, -1] = max_instance
            aug = V.Compose(
                [
                    V.Scale3d(),
                    V.RotateAroundAxis3d(
                        rotation_limit=np.pi / 24, axis=(1, 0, 0)
                    ),
                    V.RotateAroundAxis3d(
                        rotation_limit=np.pi / 24, axis=(0, 1, 0)
                    ),
                    V.RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 0, 1)),
                ]
            )(
                points=instance[:, :3],
                features=instance[:, 3:6],
                normals=instance[:, 6:9],
                labels=instance[:, 9:],
            )
            coordinates = np.concatenate((coordinates, aug["points"]))
            color = np.concatenate((color, aug["features"]))
            normals = np.concatenate((normals, aug["normals"]))
            labels = np.concatenate((labels, aug["labels"]))

        return coordinates, color, normals, labels


# Helper functions (same as original)
def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space."""
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

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
    """Crop points within the given 3D box."""
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max."
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
    """Flip point cloud in center."""
    coordinates -= coordinates.mean(0)
    aug = V.Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])["points"]
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
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])["points"]
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
    """Add random points around existing points."""
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