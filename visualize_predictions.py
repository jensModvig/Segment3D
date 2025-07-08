import argparse
import json
import colorsys
import numpy as np
import open3d as o3d
import torch
import cv2
from PIL import Image
from collections import Counter
from pathlib import Path
import copy
import yaml
import albumentations as A
from hydra.experimental import initialize, compose
from demo_utils import get_model, prepare_data

try:
    from thes.paths import scannetpp_raw_dir, scannetpp_processed_dir
except ImportError:
    scannetpp_raw_dir = Path("bigdata/scannetpp")
    scannetpp_processed_dir = Path("data/processed/scannetpp")

# Updated paths for actual preprocessed data
scannetpp_seg3d_processed_dir = Path("/work3/s173955/bigdata/processed/scannetpp_seg3d_test")


def generate_colors(n):
    return np.array([colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.8, 0.9) for i in range(n)]) * 255


def save_pcd(coords, colors, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors / 255.0, 0, 1))
    success = o3d.io.write_point_cloud(str(path), pcd)
    if not success:
        raise RuntimeError(f"Failed to save {path}")


def num_to_natural(ids):
    if np.all(ids == -1):
        return ids
    array = copy.deepcopy(ids)
    unique = np.unique(array[array != -1])
    if len(unique) == 0:
        return array
    mapping = np.full(np.max(unique) + 2, -1)
    mapping[unique + 1] = np.arange(len(unique))
    return mapping[array + 1]


def load_frame_metadata(scene_id, frame_id):
    pose_file = scannetpp_raw_dir / "data" / scene_id / "iphone" / "pose_intrinsic_imu.json"
    with open(pose_file) as f:
        data = json.load(f)
    
    frame_data = data[frame_id]
    pose = np.array(frame_data.get("aligned_pose", frame_data["pose"])).reshape(4, 4)
    intrinsic = np.array(frame_data["intrinsic"])
    intrinsic[[0, 1], [0, 1]] *= [256/1920, 192/1440]
    intrinsic[[0, 1], [2, 2]] *= [256/1920, 192/1440]
    
    return pose, intrinsic


def load_frame(scene_id, frame_id):
    scene_path = scannetpp_processed_dir / "data" / scene_id
    
    rgb_path = scene_path / "iphone" / "rgb" / f"{frame_id}.jpg"
    if not rgb_path.exists():
        rgb_path = scene_path / "iphone" / "rgb" / f"{frame_id}.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {frame_id}")
    rgb = cv2.resize(cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB), (256, 192))
    
    depth_path = scene_path / "iphone" / "depth" / f"{frame_id}.png"
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth image not found: {frame_id}")
    depth = cv2.resize(cv2.imread(str(depth_path), -1), (256, 192), cv2.INTER_NEAREST)
    
    mask_path = scene_path / "gt_mask" / f"{frame_id}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"GT mask not found: {frame_id}")
    masks = cv2.resize(np.array(Image.open(mask_path), dtype=np.int16), (256, 192), cv2.INTER_NEAREST)
    
    pose, intrinsic = load_frame_metadata(scene_id, frame_id)
    
    valid = depth != 0
    colors = rgb[valid]
    masks = masks[valid]
    
    y, x = np.mgrid[:depth.shape[0], :depth.shape[1]]
    coords_2d = np.column_stack([x[valid], y[valid], depth[valid] / 1000.0])
    
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    coords_3d = np.column_stack([
        (coords_2d[:, 0] - cx) * coords_2d[:, 2] / fx,
        (coords_2d[:, 1] - cy) * coords_2d[:, 2] / fy,
        coords_2d[:, 2],
        np.ones(len(coords_2d))
    ])
    
    coords = (coords_3d @ pose.T)[:, :3]
    
    masks = num_to_natural(masks)
    counts = Counter(masks)
    for num, count in counts.items():
        if count < 100:
            masks[masks == num] = -1
    masks = num_to_natural(masks)
    
    return coords, colors, masks


def load_scene_preprocessed(scene_id):
    # Try all possible database files
    for split in ["test", "validation", "train"]:
        database_path = scannetpp_seg3d_processed_dir / f"{split}_database.yaml"
        if not database_path.exists():
            continue
            
        with open(database_path) as f:
            database = yaml.load(f, Loader=yaml.SafeLoader)
        
        # Search for the scene in this split
        for entry in database:
            if entry.get("scene") == scene_id:
                points_file = Path(entry["filepath"])
                if not points_file.exists():
                    raise FileNotFoundError(f"Points file not found: {points_file}")
                
                points = np.load(points_file)
                coords = points[:, :3]
                colors = points[:, 3:6]
                labels = points[:, 10:12]  # semantic and instance labels
                print(f"Found scene {scene_id} in {split} split")
                return coords, colors, labels
    
    # If not found in any split
    available_splits = [f.stem.replace("_database", "") for f in scannetpp_seg3d_processed_dir.glob("*_database.yaml")]
    raise FileNotFoundError(f"Scene {scene_id} not found in any database. Available splits: {available_splits}")


def load_scene_raw(scene_id):
    raw_mesh_path = scannetpp_raw_dir / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    if not raw_mesh_path.exists():
        raise FileNotFoundError(f"Raw mesh not found: {raw_mesh_path}")
    
    mesh = o3d.io.read_triangle_mesh(str(raw_mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh: {raw_mesh_path}")
    
    coords = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) * 255
    if len(colors) == 0:
        colors = np.full((len(coords), 3), 255, dtype=np.uint8)
    
    # Load semantic labels from raw mesh
    semantic_mesh_path = scannetpp_raw_dir / "data" / scene_id / "scans" / "mesh_aligned_0.05_semantic.ply"
    if semantic_mesh_path.exists():
        semantic_mesh = o3d.io.read_triangle_mesh(str(semantic_mesh_path))
        if len(semantic_mesh.vertices) > 0:
            semantic_coords = np.asarray(semantic_mesh.vertices)
            semantic_labels = np.asarray(semantic_mesh.vertex_colors)[:, 0] * 255  # semantic labels stored in red channel
            
            # Verify coordinates match
            if np.allclose(coords, semantic_coords):
                return coords, colors, semantic_labels.astype(int)
            else:
                print(f"Warning: Coordinates mismatch between mesh and semantic mesh for {scene_id}")
    
    # Return without labels if semantic mesh not available
    print(f"Warning: No semantic labels found for raw mesh {scene_id}")
    return coords, colors, None


def infer(model, coords, colors, cfg, device):
    normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    colors_norm = normalize(image=colors.astype(np.uint8)[None])["image"][0]
    
    mock_mesh = type('M', (), {
        'vertices': coords, 'vertex_colors': colors_norm,
        'has_vertex_normals': lambda: False, 'compute_vertex_normals': lambda: None
    })()
    
    data, p2s, p2s_full, raw_coords, inv_map = prepare_data(cfg, mock_mesh, None, device)
    
    with torch.no_grad():
        outputs = model(data, point2segment=p2s, raw_coordinates=raw_coords)
    
    logits = outputs["pred_logits"][0][:, 0].cpu()
    masks = outputs["pred_masks"][0].cpu()
    
    if cfg.model.train_on_segments and p2s is not None:
        masks = masks[p2s.cpu()].squeeze(0)
    
    valid = (masks > 0).float()
    keep = valid.sum(0) > 0
    
    if keep.sum() == 0:
        return np.array([]), np.zeros((len(coords), 0), dtype=bool)
    
    masks, scores, valid = masks[:, keep], logits[keep], valid[:, keep]
    heatmap = masks.sigmoid()
    mask_scores = (heatmap * valid).sum(0) / (valid.sum(0) + 1e-6)
    scores = scores * mask_scores
    
    masks_full = masks[inv_map]
    if p2s_full is not None:
        from torch_scatter import scatter_mean
        masks_full = scatter_mean(masks_full, p2s_full.squeeze(0), dim=0)
        masks_full = (masks_full > 0.5).float()[p2s_full.squeeze(0)]
    
    return scores.numpy(), (masks_full.T > 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--output_dir", default="debug_predictions")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with initialize(config_path="conf"):
        cfg = compose(config_name="config_base_instance_segmentation.yaml")
    
    cfg.general.checkpoint = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).eval().to(device)
    
    # Process partial view (frame)
    coords, colors, labels = load_frame(args.scene_id, args.frame_id)
    prefix = f"{args.scene_id}_{args.frame_id}"
    
    # Save frame raw RGB
    save_pcd(coords, colors, output_dir / f"{prefix}_frame_rgb.pcd")
    
    # Save frame GT labels
    unique_labels = np.unique(labels[labels >= 0])
    if len(unique_labels) > 0:
        label_colors = np.zeros_like(coords)
        palette = generate_colors(len(unique_labels))
        for i, label in enumerate(unique_labels):
            label_colors[labels == label] = palette[i]
        save_pcd(coords, label_colors, output_dir / f"{prefix}_frame_gt.pcd")
    
    # Get frame predictions
    scores, masks = infer(model, coords, colors, cfg, device)
    if len(scores) > 0:
        pred_colors = np.zeros_like(coords)
        palette = generate_colors(len(scores))
        for i, mask in enumerate(masks):
            pred_colors[mask] = palette[i]
        save_pcd(coords, pred_colors, output_dir / f"{prefix}_frame_pred.pcd")
    
    # Process preprocessed scene
    try:
        scene_coords_proc, scene_colors_proc, scene_labels_proc = load_scene_preprocessed(args.scene_id)
        
        # Save preprocessed scene RGB
        save_pcd(scene_coords_proc, scene_colors_proc, output_dir / f"{args.scene_id}_preprocessed_rgb.pcd")
        
        # Save preprocessed scene GT labels
        unique_labels_proc = np.unique(scene_labels_proc[:, 1][scene_labels_proc[:, 1] >= 0])
        if len(unique_labels_proc) > 0:
            label_colors_proc = np.zeros_like(scene_coords_proc)
            palette = generate_colors(len(unique_labels_proc))
            for i, label in enumerate(unique_labels_proc):
                label_colors_proc[scene_labels_proc[:, 1] == label] = palette[i]
            save_pcd(scene_coords_proc, label_colors_proc, output_dir / f"{args.scene_id}_preprocessed_gt.pcd")
        
        # Get preprocessed scene predictions
        scene_scores_proc, scene_masks_proc = infer(model, scene_coords_proc, scene_colors_proc, cfg, device)
        if len(scene_scores_proc) > 0:
            scene_pred_colors_proc = np.zeros_like(scene_coords_proc)
            palette = generate_colors(len(scene_scores_proc))
            for i, mask in enumerate(scene_masks_proc):
                scene_pred_colors_proc[mask] = palette[i]
            save_pcd(scene_coords_proc, scene_pred_colors_proc, output_dir / f"{args.scene_id}_preprocessed_pred.pcd")
        
        print(f"Preprocessed scene: {len(unique_labels_proc)} GT instances, {len(scene_scores_proc)} predictions")
    except Exception as e:
        print(f"Could not load preprocessed scene: {e}")
    
    # Process raw scene
    scene_coords_raw, scene_colors_raw, scene_labels_raw = load_scene_raw(args.scene_id)
    
    # Save raw scene RGB
    save_pcd(scene_coords_raw, scene_colors_raw, output_dir / f"{args.scene_id}_raw_rgb.pcd")
    
    # Save raw scene GT labels if available
    if scene_labels_raw is not None:
        unique_labels_raw = np.unique(scene_labels_raw[scene_labels_raw >= 0])
        if len(unique_labels_raw) > 0:
            label_colors_raw = np.zeros_like(scene_coords_raw)
            palette = generate_colors(len(unique_labels_raw))
            for i, label in enumerate(unique_labels_raw):
                label_colors_raw[scene_labels_raw == label] = palette[i]
            save_pcd(scene_coords_raw, label_colors_raw, output_dir / f"{args.scene_id}_raw_gt.pcd")
        print(f"Raw scene: {len(unique_labels_raw)} GT labels, ", end="")
    else:
        print(f"Raw scene: No GT labels, ", end="")
    
    # Get raw scene predictions
    scene_scores_raw, scene_masks_raw = infer(model, scene_coords_raw, scene_colors_raw, cfg, device)
    if len(scene_scores_raw) > 0:
        scene_pred_colors_raw = np.zeros_like(scene_coords_raw)
        palette = generate_colors(len(scene_scores_raw))
        for i, mask in enumerate(scene_masks_raw):
            scene_pred_colors_raw[mask] = palette[i]
        save_pcd(scene_coords_raw, scene_pred_colors_raw, output_dir / f"{args.scene_id}_raw_pred.pcd")
    
    print(f"{len(scene_scores_raw)} predictions")
    
    print(f"Frame: {len(unique_labels)} GT instances, {len(scores)} predictions")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()