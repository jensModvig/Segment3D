import argparse
import shutil
import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from hydra.experimental import initialize, compose
from demo_utils import get_model
from datasets.scannetpp_stage1 import ScannetppStage1Dataset
import albumentations as A
from albumentations import Normalize
from torch_scatter import scatter_mean
import MinkowskiEngine as ME

class ModelSpaceTransformer:
    def __init__(self, cfg, device):
        self.voxel_size = cfg.data.voxel_size
        self.device = device
        self.num_queries = cfg.model.num_queries
        
    def prepare_pointcloud(self, mesh):
        points = np.asarray(mesh.vertices)
        coords = np.floor(points / self.voxel_size)
        
        # Store original points
        self.original_points = points
        
        # normalization for point cloud features
        color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
        color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
        normalize_color = A.Normalize(mean=color_mean, std=color_std)
        colors = np.asarray(mesh.vertex_colors) * 255
        pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
        colors = np.squeeze(normalize_color(image=pseudo_image)["image"])
        
        self.unique_map, self.inverse_map = ME.utils.sparse_quantize(
            coordinates=coords,
            features=colors,
            return_index=True,
            return_inverse=True,
            return_maps_only=True
        )
        
        coordinates = [torch.from_numpy(coords[self.unique_map]).int()]
        features = [torch.from_numpy(colors[self.unique_map]).float()]
        coordinates, *_ = ME.utils.sparse_collate(coords=coordinates, feats=features)
        features = torch.cat(features, dim=0)
        self.raw_coordinates = torch.from_numpy(points[self.unique_map]).float().to(self.device)
        
        data = ME.SparseTensor(
            coordinates=coordinates,
            features=features,
            device=self.device,
        )
        return data


def save_pointcloud_with_sphere(coords, colors, sphere_pos, path):
    """Save point cloud with a sphere marker at query position"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
    sphere.translate(sphere_pos)
    sphere = sphere.sample_points_uniformly(number_of_points=500)
    sphere_points = np.asarray(sphere.points)
    sphere_colors = np.full((len(sphere_points), 3), [0, 0, 255])
    
    all_coords = np.vstack([coords, sphere_points])
    all_colors = np.vstack([colors, sphere_colors])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_coords)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(all_colors / 255.0, 0, 1))
    
    o3d.io.write_point_cloud(str(path), pcd)


def get_heatmap_colors(heatmap):
    """Convert heatmap values to viridis colormap"""
    intensity = np.clip(heatmap, 0, 1)
    colors = plt.cm.viridis(intensity)[:, :3] * 255
    return colors


def get_threshold_colors(original_colors, heatmap, threshold=0.5):
    """Color points red if above threshold"""
    colors = original_colors.copy()
    mask = heatmap > threshold
    colors[mask] = [255, 0, 0]
    return colors


def find_mask_by_point_activation(model, data, transformer, query_position):
    """Strategy 1: Select mask with highest activation at query point"""
    
    # Run model with default FPS sampling
    with torch.no_grad():
        outputs = model.model(
            data, 
            point2segment=None, 
            raw_coordinates=transformer.raw_coordinates
        )
    
    masks = outputs["pred_masks"][0].detach().cpu()  # [num_points, num_queries]
    logits = outputs["pred_logits"][0].detach().cpu()  # [num_queries, num_classes]
    
    # Find nearest sparse point to query
    query_tensor = torch.from_numpy(query_position.reshape(1, 3)).float()
    distances = torch.cdist(
        query_tensor.to(transformer.device), 
        transformer.raw_coordinates.unsqueeze(0)
    )[0, 0].cpu()
    nearest_sparse_idx = distances.argmin().item()
    
    print(f"\n=== Point Activation Strategy ===")
    print(f"Query position: {query_position.flatten()}")
    print(f"Nearest sparse point distance: {distances[nearest_sparse_idx]:.6f}")
    
    # Get activations at the query point for all masks
    point_activations = masks[nearest_sparse_idx, :].sigmoid()
    
    # Select mask with highest activation at query point
    best_mask_idx = point_activations.argmax().item()
    best_activation = point_activations[best_mask_idx].item()
    
    print(f"Best mask: {best_mask_idx} with activation {best_activation:.4f}")
    print(f"Top 5 activations: {point_activations.topk(5)[0].numpy()}")
    
    # Get full resolution mask
    selected_mask = masks[:, best_mask_idx]
    mask_full = selected_mask[transformer.inverse_map]
    heatmap = mask_full.sigmoid().numpy()
    
    return heatmap, best_mask_idx


def find_mask_by_cube_iou(model, data, transformer, query_position, cube_size=1.0):
    """Strategy 2: Select mask with highest IoU with cube around query point"""
    
    # Run model
    with torch.no_grad():
        outputs = model.model(
            data,
            point2segment=None,
            raw_coordinates=transformer.raw_coordinates
        )
    
    masks = outputs["pred_masks"][0].detach().cpu()
    logits = outputs["pred_logits"][0].detach().cpu()
    
    # Define cube bounds
    cube_min = query_position.flatten() - cube_size / 2
    cube_max = query_position.flatten() + cube_size / 2
    
    print(f"\n=== Cube IoU Strategy ===")
    print(f"Cube center: {query_position.flatten()}")
    print(f"Cube bounds: [{cube_min} to {cube_max}]")
    
    # Find points inside cube (using original points for full resolution)
    points_in_cube = np.all(
        (transformer.original_points >= cube_min) & 
        (transformer.original_points <= cube_max), 
        axis=1
    )
    
    # Map to sparse indices
    sparse_points_in_cube = points_in_cube[transformer.unique_map]
    cube_indices = np.where(sparse_points_in_cube)[0]
    
    print(f"Points in cube: {points_in_cube.sum()} (full), {len(cube_indices)} (sparse)")
    
    if len(cube_indices) == 0:
        print("No points in cube, falling back to nearest point")
        # Fall back to nearest point
        query_tensor = torch.from_numpy(query_position.reshape(1, 3)).float()
        distances = torch.cdist(
            query_tensor.to(transformer.device),
            transformer.raw_coordinates.unsqueeze(0)
        )[0, 0].cpu()
        cube_indices = [distances.argmin().item()]
    
    # Calculate IoU for each mask
    best_iou = 0
    best_mask_idx = 0
    
    for mask_idx in range(masks.shape[1]):
        mask_probs = masks[:, mask_idx].sigmoid()
        mask_binary = mask_probs > 0.5
        
        # Calculate intersection: points that are both in cube AND in mask
        intersection = mask_binary[cube_indices].sum().item()
        
        # Calculate union: points that are in cube OR in mask
        union = len(cube_indices) + mask_binary.sum().item() - intersection
        
        iou = intersection / (union + 1e-6)
        
        if iou > best_iou:
            best_iou = iou
            best_mask_idx = mask_idx
    
    print(f"Best mask: {best_mask_idx} with IoU {best_iou:.4f}")
    
    # Get full resolution mask
    selected_mask = masks[:, best_mask_idx]
    mask_full = selected_mask[transformer.inverse_map]
    heatmap = mask_full.sigmoid().numpy()
    
    return heatmap, best_mask_idx


def find_mask_by_internal_scoring(cfg, model, data, transformer, query_position):
    """Strategy 3: Use internal scoring from model (similar to demo.py)"""
    
    # Run model
    with torch.no_grad():
        outputs = model.model(
            data,
            point2segment=None,
            raw_coordinates=transformer.raw_coordinates
        )
    
    masks = outputs["pred_masks"][0].detach().cpu()
    logits = outputs["pred_logits"][0].detach().cpu()
    
    print(f"\n=== Internal Scoring Strategy ===")
    
    # Apply the model's internal scoring (from demo.py)
    mask_cls = logits[:, 0] if logits.shape[1] == 1 else logits[:, 1]  # Use 'thing' class
    
    # Filter valid masks
    result_pred_mask = (masks > 0).float()
    valid_masks = result_pred_mask.sum(0) > 0
    masks = masks[:, valid_masks]
    mask_cls = mask_cls[valid_masks]
    result_pred_mask = result_pred_mask[:, valid_masks]
    
    # Calculate scores
    heatmap = masks.float().sigmoid()
    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
        result_pred_mask.sum(0) + 1e-6
    )
    scores = mask_cls.sigmoid() * mask_scores_per_image
    
    # Apply topk filtering
    topk_count = min(cfg.general.topk_per_image, len(scores)) if cfg.general.topk_per_image != -1 else len(scores)
    scores, topk_indices = scores.topk(topk_count, sorted=True)
    
    # Now find which of these top masks best covers our query point
    query_tensor = torch.from_numpy(query_position.reshape(1, 3)).float()
    distances = torch.cdist(
        query_tensor.to(transformer.device),
        transformer.raw_coordinates.unsqueeze(0)
    )[0, 0].cpu()
    nearest_sparse_idx = distances.argmin().item()
    
    # Check activation at query point for top masks
    best_combined_score = -1
    best_mask_idx = 0
    best_original_idx = 0
    
    for i, idx in enumerate(topk_indices):
        point_activation = masks[nearest_sparse_idx, idx].sigmoid().item()
        combined_score = scores[i].item() * point_activation
        
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_mask_idx = i
            best_original_idx = idx.item()
    
    print(f"Selected mask {best_original_idx} (top-{best_mask_idx+1}) with:")
    print(f"  Model score: {scores[best_mask_idx]:.4f}")
    print(f"  Point activation: {masks[nearest_sparse_idx, best_original_idx].sigmoid().item():.4f}")
    print(f"  Combined score: {best_combined_score:.4f}")
    
    # Get full resolution mask
    selected_mask = masks[:, best_original_idx]
    mask_full = selected_mask[transformer.inverse_map]
    heatmap = mask_full.sigmoid().numpy()
    
    return heatmap, best_original_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--output_dir", default="debug_query_multi")
    parser.add_argument("--cube_size", type=float, default=1.0, help="Size of cube for IoU calculation")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    with initialize(config_path="conf"):
        cfg = compose(config_name="config_base_instance_segmentation.yaml")
    
    cfg.general.checkpoint = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).eval().to(device)
    
    # Load data
    coords, colors, labels = ScannetppStage1Dataset.load_specific_frame(args.scene_id, args.frame_id)
    
    mock_mesh = type('M', (), {
        'vertices': coords,
        'vertex_colors': colors
    })()
    
    transformer = ModelSpaceTransformer(cfg, device)
    data = transformer.prepare_pointcloud(mock_mesh)
    
    print(f"Generating 5 query samples with 3 strategies for {args.scene_id}_{args.frame_id}")
    
    for i in range(4):
        # Random query point
        random_idx = np.random.randint(low=0, high=coords.shape[0]-1, size=1)
        query_pos = coords[random_idx]
        o3d_translation = query_pos.flatten()
        
        print(f"\n{'='*60}")
        print(f"Sample {i+1} - Query position: {o3d_translation}")
        
        # Strategy 1: Point activation
        heatmap_point, mask_idx_point = find_mask_by_point_activation(
            model, data, transformer, query_pos
        )
        
        # Strategy 2: Cube IoU
        heatmap_cube, mask_idx_cube = find_mask_by_cube_iou(
            model, data, transformer, query_pos, cube_size=args.cube_size
        )
        
        # Strategy 3: Internal scoring
        heatmap_internal, mask_idx_internal = find_mask_by_internal_scoring(
            cfg, model, data, transformer, query_pos
        )
        
        # Save all three versions
        prefix = f'sample_{i+1}' #f"{args.scene_id}_{args.frame_id}_sample_{i+1}"
        
        # Original for reference
        save_pointcloud_with_sphere(
            coords, colors, o3d_translation, 
            output_dir / f"{prefix}_original.pcd"
        )
        
        # Point activation strategy
        save_pointcloud_with_sphere(
            coords, get_heatmap_colors(heatmap_point), o3d_translation,
            output_dir / f"{prefix}_point_heatmap.pcd"
        )
        save_pointcloud_with_sphere(
            coords, get_threshold_colors(colors, heatmap_point), o3d_translation,
            output_dir / f"{prefix}_point_threshold.pcd"
        )
        
        # Cube IoU strategy
        save_pointcloud_with_sphere(
            coords, get_heatmap_colors(heatmap_cube), o3d_translation,
            output_dir / f"{prefix}_cube_heatmap.pcd"
        )
        save_pointcloud_with_sphere(
            coords, get_threshold_colors(colors, heatmap_cube), o3d_translation,
            output_dir / f"{prefix}_cube_threshold.pcd"
        )
        
        # Internal scoring strategy
        save_pointcloud_with_sphere(
            coords, get_heatmap_colors(heatmap_internal), o3d_translation,
            output_dir / f"{prefix}_internal_heatmap.pcd"
        )
        save_pointcloud_with_sphere(
            coords, get_threshold_colors(colors, heatmap_internal), o3d_translation,
            output_dir / f"{prefix}_internal_threshold.pcd"
        )
        
        # Print statistics
        print(f"\nActivation statistics:")
        print(f"  Point strategy: {(heatmap_point > 0.5).sum()}/{len(heatmap_point)} points "
              f"({(heatmap_point > 0.5).sum()/len(heatmap_point)*100:.1f}%)")
        print(f"  Cube IoU strategy: {(heatmap_cube > 0.5).sum()}/{len(heatmap_cube)} points "
              f"({(heatmap_cube > 0.5).sum()/len(heatmap_cube)*100:.1f}%)")
        print(f"  Internal strategy: {(heatmap_internal > 0.5).sum()}/{len(heatmap_internal)} points "
              f"({(heatmap_internal > 0.5).sum()/len(heatmap_internal)*100:.1f}%)")
    
    print(f"\nAll samples saved to: {output_dir}")
    print(f"Files per sample:")
    print(f"  *_original.pcd - Original point cloud with query marker")
    print(f"  *_point_*.pcd - Point activation strategy results")
    print(f"  *_cube_*.pcd - Cube IoU strategy results")
    print(f"  *_internal_*.pcd - Internal scoring strategy results")


if __name__ == "__main__":
    main()