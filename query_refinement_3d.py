import argparse
import numpy as np
import torch
from pathlib import Path
from hydra.experimental import initialize, compose
from demo_utils import get_model
from datasets.stage1_conf import Stage1DatasetConf
import albumentations as A
import MinkowskiEngine as ME
import cv2
from PIL import Image
import glob
from cuml.cluster import DBSCAN
import copy


class ModelSpaceTransformer:
    def __init__(self, cfg, device):
        self.voxel_size = cfg.data.voxel_size
        self.device = device
        
    def prepare_pointcloud(self, mesh):
        points = np.asarray(mesh.vertices)
        coords = np.floor(points / self.voxel_size)
        colors = np.asarray(mesh.vertex_colors) * 255
        colors = np.squeeze(A.Normalize(
            mean=(0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            std=(0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
        )(image=colors.astype(np.uint8)[np.newaxis, :, :])["image"])
        
        self.unique_map, self.inverse_map = ME.utils.sparse_quantize(
            coordinates=coords, features=colors, return_index=True, 
            return_inverse=True, return_maps_only=True)
        
        coordinates = [torch.from_numpy(coords[self.unique_map]).int()]
        features = [torch.from_numpy(colors[self.unique_map]).float()]
        coordinates, *_ = ME.utils.sparse_collate(coords=coordinates, feats=features)
        features = torch.cat(features, dim=0)
        self.raw_coordinates = torch.from_numpy(points[self.unique_map]).float().to(self.device)
        
        return ME.SparseTensor(coordinates=coordinates, features=features, device=self.device)


def get_input_path(scene_id, frame_id, shared_folder, sam_folder):
    shared_q2_files = sorted(glob.glob(f"{shared_folder}/{scene_id}_{frame_id}_i*_q2.png"))
    if shared_q2_files:
        return Path(shared_q2_files[-1])
    return Path(f'/work3/s173955/bigdata/processed/scannetpp/data/{scene_id}/{sam_folder}/{frame_id}.png')


def get_iteration_number(scene_id, frame_id, shared_folder):
    shared_files = glob.glob(f"{shared_folder}/{scene_id}_{frame_id}_i*_q*.png")
    if not shared_files:
        return 1
    iterations = [int(f.split('_i')[1].split('_')[0]) for f in shared_files]
    return max(iterations) + 1


def num_to_natural(group_ids):
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


def load_sam_labels_3d(scene_id, frame_id, input_path, depth_resolution):
    """Load and process SAM labels, remapping them to consecutive integers."""
    data_path = Path('/work3/s173955/bigdata/processed/scannetpp/data/')
    depth_path = data_path / scene_id / f'depth_pro/depth_map_fpxc_{depth_resolution}' / f'{frame_id}.npz'
    
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"SAM file not found: {input_path}")
    
    depth_data = np.load(str(depth_path))
    depth_image = depth_data['depth']
    sam_groups = cv2.resize(
        np.array(Image.open(input_path), dtype=np.int16), 
        depth_image.shape[::-1], 
        interpolation=cv2.INTER_NEAREST
    )[depth_image != 0]
    
    return num_to_natural(sam_groups)


def get_depth_mask(scene_id, frame_id, depth_resolution):
    depth_path = Path('/work3/s173955/bigdata/processed/scannetpp/data/') / scene_id / f'depth_pro/depth_map_fpxc_{depth_resolution}' / f'{frame_id}.npz'
    depth_data = np.load(str(depth_path))
    return (depth_data['depth'] != 0)

import numpy as np
from typing import List, Tuple

def find_contained_masks(masks: np.ndarray, containers: np.ndarray, iou_threshold: float = 0.9) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    """Find masks contained within container masks based on IoU threshold."""
    n_masks, w = masks.shape
    n_containers = containers.shape[0]
    
    if containers.shape[1:] != (w,):
        raise ValueError(f"Shape mismatch: masks {masks.shape} vs containers {containers.shape}")
    
    contained_by_container = [[] for _ in range(n_containers)]
    contained_indices = set()
    
    for container_idx in range(n_containers):
        container = containers[container_idx].astype(bool)
        
        for mask_idx in range(n_masks):
            if mask_idx in contained_indices:
                continue
                
            mask = masks[mask_idx].astype(bool)
            intersection = np.sum(mask & container)
            mask_area = np.sum(mask)
            
            if mask_area > 0 and (intersection / mask_area) > iou_threshold:
                contained_by_container[container_idx].append(mask)
                contained_indices.add(mask_idx)
    
    remaining_masks = [masks[i] for i in range(n_masks) if i not in contained_indices]
    
    return contained_by_container, remaining_masks


def get_model_masks(cfg, model, data, transformer):
    with torch.no_grad():
        outputs = model.model(data, point2segment=None, raw_coordinates=transformer.raw_coordinates)
    
    masks = outputs["pred_masks"][0].detach().cpu()
    logits = outputs["pred_logits"][0][:, 0].detach().cpu()  # Always take first column
    
    # Get valid masks (those with at least one point)
    result_pred_mask = (masks > 0).float()
    valid_masks = result_pred_mask.sum(0) > 0
    masks = masks[:, valid_masks]
    mask_cls = logits[valid_masks]
    result_pred_mask = result_pred_mask[:, valid_masks]
    
    # Calculate scores using the same approach as reference implementation
    heatmap = masks.float().sigmoid()  # Apply sigmoid to masks to get heatmap
    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
    scores = mask_cls * mask_scores_per_image  # No sigmoid on mask_cls
    
    # Get top-k predictions
    topk_count = min(cfg.general.topk_per_image, len(scores)) if cfg.general.topk_per_image != -1 else len(scores)
    _, topk_indices = scores.topk(topk_count, sorted=True)
    
    # Return just the masks (matching the original return format)
    return [(masks[:, idx][transformer.inverse_map].sigmoid().numpy() > 0.5).astype(np.float32) 
            for idx in topk_indices]


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def select_and_merge_masks(model_masks, sam_labels_3d, selection_threshold, merge_threshold):
    selected_masks = []
    
    sam_ids = np.unique(sam_labels_3d[sam_labels_3d != -1])
    
    for sam_id in sam_ids:
        sam_binary = (sam_labels_3d == sam_id).astype(np.float32)
        best_iou, best_mask = 0, None
        
        for model_mask in model_masks:
            iou = calculate_iou(model_mask, sam_binary)
            if iou > best_iou:
                best_iou, best_mask = iou, model_mask
        
        if best_iou > selection_threshold:
            selected_masks.append(best_mask)
    
    if not selected_masks:
        return []
    
    # Extract SAM masks
    sam_masks = [(sam_labels_3d == sam_id).astype(np.float32) for sam_id in sam_ids]
    
    selected_array = np.stack(selected_masks)
    sam_array = np.stack(sam_masks)
    
    # Find containments both ways
    selected_in_sam, _ = find_contained_masks(selected_array, sam_array, iou_threshold=0.9)
    sam_in_selected, _ = find_contained_masks(sam_array, selected_array, iou_threshold=0.9)
    
    # Merge contained masks with containers
    merged_set1 = []
    for container_idx, contained_masks in enumerate(selected_in_sam):
        if contained_masks:
            merged_mask = sam_masks[container_idx].copy()
            for contained_mask in contained_masks:
                merged_mask = np.logical_or(merged_mask, contained_mask).astype(np.float32)
            merged_set1.append(merged_mask)
    
    merged_set2 = []
    for container_idx, contained_masks in enumerate(sam_in_selected):
        if contained_masks:
            merged_mask = selected_masks[container_idx].copy()
            for contained_mask in contained_masks:
                merged_mask = np.logical_or(merged_mask, contained_mask).astype(np.float32)
            merged_set2.append(merged_mask)
    
    # Final merge between sets based on IoU > 0.5
    final_masks = merged_set1.copy()
    used_indices = set()
    
    for mask1 in merged_set2:
        best_iou, best_idx = 0, -1
        for idx, mask2 in enumerate(final_masks):
            if idx not in used_indices:
                iou = calculate_iou(mask1, mask2)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
        
        if best_iou > 0.5:
            final_masks[best_idx] = np.logical_or(final_masks[best_idx], mask1).astype(np.float32)
            used_indices.add(best_idx)
        else:
            final_masks.append(mask1)
    
    return final_masks


def map_to_2d_and_save(masks_3d, depth_mask, output_path):
    if not masks_3d:
        return
        
    h, w = depth_mask.shape
    output_image = np.full((h, w), -1, dtype=np.int16)
    
    for i, mask_3d in enumerate(masks_3d):
        mask_2d_indices = np.full((h, w), False, dtype=bool)
        mask_2d_indices[depth_mask] = mask_3d > 0
        # 50% transparency: random selection of pixels to write
        transparency_mask = np.random.random((h, w)) < 0.5
        final_mask = mask_2d_indices & transparency_mask
        output_image[final_mask] = i
    
    Image.fromarray(output_image.astype(np.uint16)).save(output_path)


def apply_dbscan_clustering(masks_3d, coordinates, cfg):
    if not masks_3d:
        return masks_3d
    
    clustered_masks = []
    for mask_3d in masks_3d:
        mask_coords = coordinates[mask_3d > 0]
        if len(mask_coords) < cfg.general.dbscan_min_points:
            continue
            
        clusters = DBSCAN(
            eps=cfg.general.dbscan_eps,
            min_samples=cfg.general.dbscan_min_points
        ).fit(mask_coords).labels_
        
        for cluster_id in np.unique(clusters):
            if cluster_id != -1:
                cluster_mask = np.zeros_like(mask_3d)
                mask_indices = np.where(mask_3d > 0)[0]
                cluster_points = mask_indices[clusters == cluster_id]
                cluster_mask[cluster_points] = 1.0
                clustered_masks.append(cluster_mask)
    
    return clustered_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--depth_resolution", required=True)
    parser.add_argument("--shared_folder", default="/work3/s173955/data/label_refinement")
    parser.add_argument("--sam_folder", default="sam")
    parser.add_argument("--selection_iou_threshold", type=float, default=0.001)
    parser.add_argument("--merge_iou_threshold", type=float, default=0.1)
    args = parser.parse_args()
    
    Path(args.shared_folder).mkdir(parents=True, exist_ok=True)
    
    input_path = get_input_path(args.scene_id, args.frame_id, args.shared_folder, args.sam_folder)
    print('using input path', input_path)
    iteration = get_iteration_number(args.scene_id, args.frame_id, args.shared_folder)
    
    with initialize(config_path="conf"):
        cfg = compose(config_name="config_base_instance_segmentation.yaml")
    
    cfg.general.checkpoint = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).eval().to(device)
    
    coords, colors, labels = Stage1DatasetConf.load_specific_frame(
        args.scene_id, args.frame_id,
        depth_folder=f'depth_pro/depth_map_fpxc_{args.depth_resolution}'
    )
    
    confidence = labels[:, 3]
    confidence_mask = confidence >= 0.5
    coords_filtered = coords[confidence_mask]
    colors_filtered = colors[confidence_mask]
    
    transformer = ModelSpaceTransformer(cfg, device)
    data = transformer.prepare_pointcloud(type('M', (), {'vertices': coords_filtered, 'vertex_colors': colors_filtered})())
    
    sam_labels_3d = load_sam_labels_3d(args.scene_id, args.frame_id, input_path, args.depth_resolution)
    sam_labels_3d_filtered = sam_labels_3d[confidence_mask]
    
    model_masks = get_model_masks(cfg, model, data, transformer)
    
    merged_masks = model_masks
    merged_masks = select_and_merge_masks(
        model_masks, sam_labels_3d_filtered, 
        args.selection_iou_threshold, args.merge_iou_threshold
    )
    
    # clustered_masks = apply_dbscan_clustering(merged_masks, coords_filtered, cfg)
    clustered_masks = merged_masks
    
    clustered_masks_full = []
    for mask in clustered_masks:
        full_mask = np.zeros(len(coords), dtype=np.float32)
        full_mask[confidence_mask] = mask
        clustered_masks_full.append(full_mask)
    
    depth_mask = get_depth_mask(args.scene_id, args.frame_id, args.depth_resolution)
    output_path = Path(args.shared_folder) / f"{args.scene_id}_{args.frame_id}_i{iteration}_q3.png"
    map_to_2d_and_save(clustered_masks_full, depth_mask, output_path)


if __name__ == "__main__":
    main()