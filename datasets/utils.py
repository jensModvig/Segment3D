import MinkowskiEngine as ME
import numpy as np
import torch
from random import random


class VoxelizeCollate:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        probing=False,
        task="instance_segmentation",
        ignore_class_threshold=100,
        filter_out_classes=[],
        label_offset=0,
        num_queries=None,
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "task not known"
        self.task = task
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.batch_instance = batch_instance
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold
        self.num_queries = num_queries

    def __call__(self, batch):
        """
        Process batch through cropping and voxelization.
        
        Input batch: List of samples from dataset, each sample:
        [
            coordinates,     # (N, 3) - world coordinates  
            features,        # (N, F) - features (color + coords)
            labels,          # (N, 4) - [semantic, instance, segment, confidence]
            filename,        # string
            raw_color,       # (N, 3)
            raw_normals,     # (N, 3)
            raw_coordinates, # (N, 3)
            idx              # int
        ]
        """
        if ("train" in self.mode) and (
            self.small_crops or self.very_small_crops
        ):
            batch = make_crops(batch)
        if ("train" in self.mode) and self.very_small_crops:
            batch = make_crops(batch)
        return voxelize(
            batch,
            self.ignore_label,
            self.voxel_size,
            self.probing,
            self.mode,
            task=self.task,
            ignore_class_threshold=self.ignore_class_threshold,
            filter_out_classes=self.filter_out_classes,
            label_offset=self.label_offset,
            num_queries=self.num_queries,
        )


class VoxelizeCollateMerge:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        scenes=2,
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        make_one_pc_noise=False,
        place_nearby=False,
        place_far=False,
        proba=1,
        probing=False,
        task="instance_segmentation",
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "task not known"
        self.task = task
        self.mode = mode
        self.scenes = scenes
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.ignore_label = ignore_label
        self.voxel_size = voxel_size
        self.batch_instance = batch_instance
        self.make_one_pc_noise = make_one_pc_noise
        self.place_nearby = place_nearby
        self.place_far = place_far
        self.proba = proba
        self.probing = probing

    def __call__(self, batch):
        if (
            ("train" in self.mode)
            and (not self.make_one_pc_noise)
            and (self.proba > random())
        ):
            if self.small_crops or self.very_small_crops:
                batch = make_crops(batch)
            if self.very_small_crops:
                batch = make_crops(batch)
            if self.batch_instance:
                batch = batch_instances(batch)
            new_batch = []
            for i in range(0, len(batch), self.scenes):
                batch_coordinates = []
                batch_features = []
                batch_labels = []

                batch_filenames = ""
                batch_raw_color = []
                batch_raw_normals = []

                offset_instance_id = 0
                offset_segment_id = 0

                for j in range(min(len(batch[i:]), self.scenes)):
                    batch_coordinates.append(batch[i + j][0])
                    batch_features.append(batch[i + j][1])

                    if j == 0:
                        batch_filenames = batch[i + j][3]  # filename at index 3
                    else:
                        batch_filenames = batch_filenames + f"+{batch[i + j][3]}"

                    batch_raw_color.append(batch[i + j][4])    # raw_color at index 4
                    batch_raw_normals.append(batch[i + j][5])  # raw_normals at index 5

                    # make instance ids and segment ids unique
                    # take care that -1 instances stay at -1
                    # Note: labels now have 4 columns (semantic, instance, segment, confidence)
                    labels_copy = batch[i + j][2].copy()
                    labels_copy[:, 1:3] += [offset_instance_id, offset_segment_id]  # Only offset instance and segment columns
                    labels_copy[batch[i + j][2][:, 1] == -1, 1] = -1  # Keep -1 instances as -1
                    batch_labels.append(labels_copy)

                    # Calculate max from instance and segment columns only (columns 1 and 2)
                    max_instance_id, max_segment_id = batch[i + j][2][:, 1:3].max(axis=0)
                    offset_segment_id = offset_segment_id + max_segment_id + 1
                    offset_instance_id = offset_instance_id + max_instance_id + 1

                if (len(batch_coordinates) == 2) and self.place_nearby:
                    border = batch_coordinates[0][:, 0].max()
                    border -= batch_coordinates[1][:, 0].min()
                    batch_coordinates[1][:, 0] += border
                elif (len(batch_coordinates) == 2) and self.place_far:
                    batch_coordinates[1] += (
                        np.random.uniform((-10, -10, -10), (10, 10, 10)) * 200
                    )
                new_batch.append(
                    (
                        np.vstack(batch_coordinates),
                        np.vstack(batch_features),
                        np.concatenate(batch_labels),
                        batch_filenames,
                        np.vstack(batch_raw_color),
                        np.vstack(batch_raw_normals),
                    )
                )
            # TODO WHAT ABOUT POINT2SEGMENT AND SO ON ...
            batch = new_batch
        elif ("train" in self.mode) and self.make_one_pc_noise:
            new_batch = []
            for i in range(0, len(batch), 2):
                if (i + 1) < len(batch):
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    batch[i][2],
                                    np.full_like(
                                        batch[i + 1][2], self.ignore_label
                                    ),
                                )
                            ),
                        ]
                    )
                    new_batch.append(
                        [
                            np.vstack((batch[i][0], batch[i + 1][0])),
                            np.vstack((batch[i][1], batch[i + 1][1])),
                            np.concatenate(
                                (
                                    np.full_like(
                                        batch[i][2], self.ignore_label
                                    ),
                                    batch[i + 1][2],
                                )
                            ),
                        ]
                    )
                else:
                    new_batch.append([batch[i][0], batch[i][1], batch[i][2]])
            batch = new_batch
        
        return voxelize(
            batch,
            self.ignore_label,
            self.voxel_size,
            self.probing,
            self.mode,
            task=self.task,
        )


def batch_instances(batch):
    """
    Updated batch_instances to handle 4-column labels with confidence.
    
    Splits batch by instances, keeping only semantic labels for each instance.
    
    Input batch: List of samples [coordinates, features, labels_with_confidence]
    Output batch: List of instance samples [coordinates, features, semantic_labels]
    """
    new_batch = []
    for sample in batch:
        # sample[2] has shape (N, 4) - [semantic, instance, segment, confidence]
        unique_instance_ids = np.unique(sample[2][:, 1])  # Get unique instance IDs
        
        for instance_id in unique_instance_ids:
            instance_mask = sample[2][:, 1] == instance_id  # (N,) - bool mask for this instance
            new_batch.append(
                (
                    sample[0][instance_mask],           # (N_instance, 3) - coordinates for this instance
                    sample[1][instance_mask],           # (N_instance, F) - features for this instance  
                    sample[2][instance_mask][:, 0],     # (N_instance,) - semantic labels only for this instance
                ),
            )
    return new_batch


def voxelize(
    batch,
    ignore_label,
    voxel_size,
    probing,
    mode,
    task,
    ignore_class_threshold,
    filter_out_classes,
    label_offset,
    num_queries,
):
    """
    Voxelize batch with confidence support.
    
    Input batch: List of samples, each sample is:
    [
        coordinates,              # (N, 3) - world coordinates
        features,                 # (N, F) - typically (N, 6) for color+coords  
        labels,                   # (N, 4) - [semantic, instance, segment, confidence]
        filename,                 # string
        raw_color,                # (N, 3)
        raw_normals,              # (N, 3)
        raw_coordinates,          # (N, 3)
        idx                       # int
    ]
    """
    (
        coordinates,              # List of voxelized coordinates per batch
        features,                 # List of voxelized features per batch
        labels,                   # List of voxelized labels per batch (N_vox, 3) - no confidence
        original_labels,          # List of original labels per batch (N_orig, 4) - with confidence
        inverse_maps,             # List of inverse mapping indices per batch
        original_colors,          # List of original colors per batch
        original_normals,         # List of original normals per batch
        original_coordinates,     # List of original coordinates per batch
        idx,                      # List of sample indices
        voxelized_confidences,    # List of voxelized confidences per batch (N_vox, 1)
        original_confidences,     # List of original confidences per batch (N_orig, 1)
    ) = ([], [], [], [], [], [], [], [], [], [], [])
    
    voxelization_dict = {
        "ignore_label": ignore_label,
        "return_index": True,
        "return_inverse": True,
    }

    full_res_coords = []  # List of full resolution coordinates per batch

    for sample in batch:
        # Extract elements from sample tuple - 8 elements total (indices 0-7)
        sample_coords = sample[0]           # (N, 3) - coordinates
        sample_features = sample[1]         # (N, F) - features
        sample_labels_with_conf = sample[2] # (N, 4) - [semantic, instance, segment, confidence]
        sample_filename = sample[3]         # string
        sample_raw_color = sample[4]        # (N, 3)
        sample_raw_normals = sample[5]      # (N, 3) 
        sample_raw_coords = sample[6]       # (N, 3)
        sample_idx = sample[7]              # int
        
        # Store original data
        idx.append(sample_idx)
        original_coordinates.append(sample_raw_coords)           # (N, 3)
        original_labels.append(sample_labels_with_conf)         # (N, 4) - keep confidence
        full_res_coords.append(sample_coords)                   # (N, 3)
        original_colors.append(sample_raw_color)                # (N, 3)
        original_normals.append(sample_raw_normals)             # (N, 3)
        
        # Extract confidence and labels separately - ALWAYS expect confidence in column 3
        sample_confidence = sample_labels_with_conf[:, 3:4].astype(np.float32)  # (N, 1) - confidence column
        sample_labels_no_conf = sample_labels_with_conf[:, :3]  # (N, 3) - [semantic, instance, segment]
        original_confidences.append(sample_confidence)          # (N, 1)

        # Prepare coordinates for voxelization
        coords = np.floor(sample_coords / voxel_size)           # (N, 3) - voxel coordinates
        voxelization_dict.update(
            {
                "coordinates": torch.from_numpy(coords).to("cpu").contiguous(),
                "features": sample_features,
            }
        )

        # Perform voxelization to get unique voxel mapping
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(**voxelization_dict)
        # unique_map: (N_vox,) - indices of points that represent each unique voxel
        # inverse_map: (N,) - mapping from original points to voxel indices
        inverse_maps.append(inverse_map)

        # Apply voxelization mapping to get unique voxel data
        sample_coordinates = coords[unique_map]                 # (N_vox, 3) - voxelized coordinates
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        
        sample_features_vox = sample_features[unique_map]       # (N_vox, F) - voxelized features
        features.append(torch.from_numpy(sample_features_vox).float())
        
        if len(sample_labels_no_conf) > 0:
            sample_labels_vox = sample_labels_no_conf[unique_map]  # (N_vox, 3) - voxelized labels
            labels.append(torch.from_numpy(sample_labels_vox).long())
            
            # Apply same voxelization to confidence - ALWAYS present
            sample_conf_vox = sample_confidence[unique_map]        # (N_vox, 1) - voxelized confidence
            voxelized_confidences.append(torch.from_numpy(sample_conf_vox).float())  # Always add confidence

    # Concatenate all batch data using MinkowskiEngine sparse collate
    input_dict = {"coords": coordinates, "feats": features}
    if len(labels) > 0:
        input_dict["labels"] = labels
        # Collate coordinates, features, and labels
        coordinates, features, labels = ME.utils.sparse_collate(**input_dict)
        # After collate:
        # coordinates: (B*N_vox, 4) - [batch_idx, x, y, z] 
        # features: (B*N_vox, F)
        # labels: (B*N_vox, 3) - [semantic, instance, segment]
        
        # We don't collate confidences with ME because it doesn't support arbitrary tensors
        # Keep as list of tensors: voxelized_confidences: List[(N_vox_i, 1)]
        # Confidence should ALWAYS be present when labels are present
        assert len(voxelized_confidences) == len(input_dict["labels"]), "Confidence must be present for all batches with labels"
    else:
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        labels = torch.Tensor([])
        voxelized_confidences = []  # Empty if no labels

    if probing:
        return (
            NoGpu(
                coordinates,
                features,
                original_labels,
                inverse_maps,
            ),
            labels,
        )

    # Handle segment remapping for test mode
    if mode == "test":
        for i in range(len(input_dict["labels"])):
            # Remap semantic labels (column 0) to consecutive indices
            _, ret_index, ret_inv = np.unique(
                input_dict["labels"][i][:, 0],
                return_index=True,
                return_inverse=True,
            )
            input_dict["labels"][i][:, 0] = torch.from_numpy(ret_inv)
    else:
        # Training mode - handle segment remapping
        input_dict["segment2label"] = []

        if "labels" in input_dict:
            for i in range(len(input_dict["labels"])):
                # Remap segment IDs (column 2) to consecutive indices
                _, ret_index, ret_inv = np.unique(
                    input_dict["labels"][i][:, -1],  # segment column
                    return_index=True,
                    return_inverse=True,
                )
                input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
                # segment2label: (N_segments, 2) - [semantic, instance] for each segment
                input_dict["segment2label"].append(
                    input_dict["labels"][i][ret_index][:, :-1]
                )

    # Generate target masks for training/validation
    if "labels" in input_dict:
        list_labels = input_dict["labels"]                    # List[(N_vox_i, 3)]
        list_confidences = voxelized_confidences             # List[(N_vox_i, 1)]

        target = []
        target_full = []

        if len(list_labels[0].shape) == 1:
            # Semantic segmentation case - labels are 1D
            for batch_id in range(len(list_labels)):
                label_ids = list_labels[batch_id].unique()
                if 255 in label_ids:
                    label_ids = label_ids[:-1]

                target.append({
                    "labels": label_ids,                                     # (N_classes,)
                    "masks": list_labels[batch_id] == label_ids.unsqueeze(1), # (N_classes, N_vox)
                })
        else:
            # Instance segmentation case - labels are 2D
            if mode == "test":
                for i in range(len(input_dict["labels"])):
                    target.append(
                        {"point2segment": input_dict["labels"][i][:, 0]}  # (N_vox,) - semantic labels
                    )
                    target_full.append(
                        {
                            "point2segment": torch.from_numpy(
                                original_labels[i][:, 0]  # (N_orig,) - original semantic labels
                            ).long()
                        }
                    )
            else:
                # Training/validation mode - generate instance masks
                target = get_instance_masks(
                    list_labels,                              # List[(N_vox_i, 3)]
                    list_confidences,                         # List[(N_vox_i, 1)] - ALWAYS present
                    task,
                    list_segments=input_dict["segment2label"], # List[(N_segments_i, 2)]
                    ignore_class_threshold=ignore_class_threshold,
                    filter_out_classes=filter_out_classes,
                    label_offset=label_offset,
                )
                for i in range(len(target)):
                    # Add point-to-segment mapping
                    target[i]["point2segment"] = input_dict["labels"][i][:, 2]  # (N_vox,) - segment IDs
                    
                if "train" not in mode:
                    # Also generate full resolution targets for validation
                    target_full = get_instance_masks(
                        [torch.from_numpy(l[:, :3]) for l in original_labels],  # List[(N_orig_i, 3)] - no confidence
                        [torch.from_numpy(c).float() for c in original_confidences],  # List[(N_orig_i, 1)] - ALWAYS present
                        task,
                        ignore_class_threshold=ignore_class_threshold,
                        filter_out_classes=filter_out_classes,
                        label_offset=label_offset,
                    )
                    for i in range(len(target_full)):
                        target_full[i]["point2segment"] = torch.from_numpy(
                            original_labels[i][:, 2]  # (N_orig,) - original segment IDs
                        ).long()
    else:
        target = []
        target_full = []
        coordinates = []
        features = []

    # Return results
    if "train" not in mode:
        return (
            NoGpu(
                coordinates,                # (B*N_vox, 4) - collated coordinates
                features,                   # (B*N_vox, F) - collated features
                original_labels,            # List[(N_orig_i, 4)] - original labels with confidence
                inverse_maps,               # List[(N_orig_i,)] - voxel mapping per batch
                full_res_coords,            # List[(N_orig_i, 3)] - full res coordinates per batch
                target_full,                # List[Dict] - full resolution targets
                original_colors,            # List[(N_orig_i, 3)] - original colors per batch
                original_normals,           # List[(N_orig_i, 3)] - original normals per batch
                original_coordinates,       # List[(N_orig_i, 3)] - original coordinates per batch
                idx,                        # List[int] - sample indices
            ),
            target,                         # List[Dict] - voxelized targets
            [sample[3] for sample in batch], # List[str] - filenames
        )
    else:
        return (
            NoGpu(
                coordinates,                # (B*N_vox, 4) - collated coordinates
                features,                   # (B*N_vox, F) - collated features
                original_labels,            # List[(N_orig_i, 4)] - original labels with confidence
                inverse_maps,               # List[(N_orig_i,)] - voxel mapping per batch
                full_res_coords,            # List[(N_orig_i, 3)] - full res coordinates per batch
            ),
            target,                         # List[Dict] - voxelized targets
            [sample[3] for sample in batch], # List[str] - filenames
        )


def get_instance_masks(
    list_labels,                    # List[(N_vox_i, 3)] - [semantic, instance, segment] per batch
    confidences,                    # List[(N_vox_i, 1)] - confidence per point per batch - ALWAYS required
    task,
    list_segments=None,             # List[(N_segments_i, 2)] - [semantic, instance] per segment per batch
    ignore_class_threshold=100,
    filter_out_classes=[],
    label_offset=0,
):
    """
    Generate instance masks and confidence masks for each batch.
    
    Args:
        list_labels: List of label tensors, each (N_vox_i, 3) with [semantic, instance, segment]
        confidences: List of confidence tensors, each (N_vox_i, 1) with confidence values - ALWAYS required
        
    Returns:
        target: List of dictionaries, each containing:
            - "labels": (N_instances_i,) - class labels for each instance
            - "masks": (N_instances_i, N_vox_i) - binary masks for each instance  
            - "confidence": (N_instances_i, N_vox_i) - confidence masks for each instance - ALWAYS present
            - "segment_mask": (N_instances_i, N_segments_i) - segment masks if segments provided
    """
    # Confidence must always be provided
    assert confidences is not None, "Confidence must always be provided - no fallbacks allowed"
    assert len(confidences) == len(list_labels), "Confidence must be provided for all batches"
    
    target = []

    for batch_id in range(len(list_labels)):
        batch_labels = list_labels[batch_id]        # (N_vox, 3) - [semantic, instance, segment]
        batch_confidence = confidences[batch_id].float()  # (N_vox, 1) - ALWAYS present
        
        label_ids = []                              # List of class labels for each instance
        masks = []                                  # List of binary masks (N_vox,) for each instance
        segment_masks = []                          # List of segment masks (N_segments,) for each instance  
        confidence_masks = []                       # List of confidence masks (N_vox,) for each instance
        
        # Get unique instance IDs in this batch
        instance_ids = batch_labels[:, 1].unique()  # (N_unique_instances,) - unique instance IDs
        
        for instance_id in instance_ids:
            if instance_id == -1:
                continue  # Skip invalid instances

            # Get all points belonging to this instance
            instance_points_mask = batch_labels[:, 1] == instance_id  # (N_vox,) - bool mask
            instance_points = batch_labels[instance_points_mask]      # (N_instance_points, 3)
                
            # Get the class label for this instance (should be consistent across all points)
            label_id = instance_points[0, 0]  # scalar - semantic class label

            # Apply filtering
            if label_id in filter_out_classes:
                continue

            if (
                255 in filter_out_classes
                and label_id.item() == 255
                and instance_points.shape[0] < ignore_class_threshold
            ):
                continue

            # Store instance information
            label_ids.append(label_id)              # scalar
            masks.append(instance_points_mask)      # (N_vox,) - binary mask for this instance
            
            # Create confidence mask for this instance - ALWAYS create
            conf_mask = torch.zeros(batch_labels.shape[0], dtype=torch.float32)  # (N_vox,) - init to 0
            instance_confidence = batch_confidence[instance_points_mask].squeeze().float()  # Convert to float32
            conf_mask[instance_points_mask] = instance_confidence  # Fill instance points with their confidence
            confidence_masks.append(conf_mask)  # (N_vox,) - confidence mask for this instance
            
            # Handle segment masks if provided
            if list_segments:
                # Get unique segment IDs for this instance
                instance_segment_ids = instance_points[:, 2].unique()  # (N_instance_segments,) - segments in this instance
                
                # Create segment mask: True for segments that belong to this instance
                segment_mask = torch.zeros(list_segments[batch_id].shape[0], dtype=torch.bool)  # (N_segments,) - init to False
                segment_mask[instance_segment_ids] = True  # Set True for segments in this instance
                
                segment_masks.append(segment_mask)         # (N_segments,) - segment mask for this instance

        if len(label_ids) == 0:
            return list()

        # Stack all instance data for this batch
        label_ids = torch.stack(label_ids)          # (N_instances,) - class labels
        masks = torch.stack(masks)                  # (N_instances, N_vox) - binary masks
        confidence_masks = torch.stack(confidence_masks)  # (N_instances, N_vox) - confidence masks - ALWAYS present

        if list_segments:
            segment_masks = torch.stack(segment_masks)  # (N_instances, N_segments) - segment masks

        # Handle semantic segmentation task - merge instances of same class
        if task == "semantic_segmentation":
            new_label_ids = []
            new_masks = []
            new_segment_masks = []
            new_confidence_masks = []
            
            # Group by unique class labels
            for label_id in label_ids.unique():
                class_mask = label_ids == label_id          # (N_instances,) - instances of this class
                
                new_label_ids.append(label_id)             # scalar
                # Combine all instance masks of this class with logical OR
                new_masks.append(masks[class_mask, :].sum(dim=0).bool())  # (N_vox,) - combined mask
                
                # For semantic segmentation, take max confidence across instances of same class
                class_conf_mask = confidence_masks[class_mask, :]      # (N_class_instances, N_vox)
                combined_conf_mask = class_conf_mask.max(dim=0)[0].float()  # (N_vox,) - max confidence per point
                new_confidence_masks.append(combined_conf_mask)
                
                if list_segments:
                    # Combine segment masks with logical OR
                    new_segment_masks.append(segment_masks[class_mask, :].sum(dim=0).bool())  # (N_segments,)

            # Replace with merged data
            label_ids = torch.stack(new_label_ids)          # (N_classes,)
            masks = torch.stack(new_masks)                  # (N_classes, N_vox)
            confidence_masks = torch.stack(new_confidence_masks)  # (N_classes, N_vox) - ALWAYS present

            if list_segments:
                segment_masks = torch.stack(new_segment_masks)  # (N_classes, N_segments)

        # Build target dictionary for this batch - ALWAYS include confidence
        target_dict = {
            "labels": torch.clamp(label_ids - label_offset, min=0),  # (N_instances,) - adjusted class labels
            "masks": masks,                                          # (N_instances, N_vox) - binary instance masks
            "confidence": confidence_masks,                          # (N_instances, N_vox) - confidence masks - ALWAYS present
        }
            
        if list_segments:
            target_dict["segment_mask"] = segment_masks              # (N_instances, N_segments) - segment masks
            
        target.append(target_dict)
        
    return target

def make_crops(batch):
    """
    Modified make_crops to handle confidence in labels.
    
    Input batch: List of samples, each with 8 elements including labels with confidence
    Output batch: List of cropped samples with 3 elements [coordinates, features, labels]
    """
    new_batch = []
    # Extract first 3 elements (coordinates, features, labels) for cropping
    for scene in batch:
        new_batch.append([scene[0], scene[1], scene[2]])  # [coords, features, labels]
    batch = new_batch
    new_batch = []
    
    for scene in batch:
        # Move to center for better quadrant split
        scene[0][:, :3] -= scene[0][:, :3].mean(0)

        # BUGFIX - Add points in each quadrant to ensure non-empty crops
        padding_coords = np.array([
            [0.1, 0.1, 0.1],    # +x, +y quadrant
            [0.1, -0.1, 0.1],   # +x, -y quadrant  
            [-0.1, 0.1, 0.1],   # -x, +y quadrant
            [-0.1, -0.1, 0.1],  # -x, -y quadrant
        ])
        scene[0] = np.vstack((scene[0], padding_coords))  # (N+4, 3)
        
        # Add padding features  
        padding_features = np.zeros((4, scene[1].shape[1]))  # (4, F)
        scene[1] = np.vstack((scene[1], padding_features))   # (N+4, F)
        
        # Add padding labels with structure [semantic, instance, segment, confidence]
        padding_labels = np.full((4, 4), 255, dtype=np.float32)  # (4, 4) - all columns to 255
        padding_labels[:, 3] = 1  # Set confidence column to default value 0.5
        scene[2] = np.vstack((scene[2], padding_labels))     # (N+4, 4)

        # Create crops for each quadrant
        # +x, +y quadrant
        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        # +x, -y quadrant  
        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        # -x, +y quadrant
        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        # -x, -y quadrant
        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

    # Center all crops
    for i in range(len(new_batch)):
        new_batch[i][0][:, :3] -= new_batch[i][0][:, :3].mean(0)
    
    return new_batch


class NoGpu:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        full_res_coords=None,
        target_full=None,
        original_colors=None,
        original_normals=None,
        original_coordinates=None,
        idx=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps
        self.full_res_coords = full_res_coords
        self.target_full = target_full
        self.original_colors = original_colors
        self.original_normals = original_normals
        self.original_coordinates = original_coordinates
        self.idx = idx


class NoGpuMask:
    def __init__(
        self,
        coordinates,
        features,
        original_labels=None,
        inverse_maps=None,
        masks=None,
        labels=None,
    ):
        """helper class to prevent gpu loading on lightning"""
        self.coordinates = coordinates
        self.features = features
        self.original_labels = original_labels
        self.inverse_maps = inverse_maps

        self.masks = masks
        self.labels = labels