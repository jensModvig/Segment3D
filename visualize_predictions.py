import argparse
import json
import colorsys
import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from hydra.experimental import initialize, compose
from demo_utils import get_model, prepare_data
from datasets.scannetpp_stage1 import ScannetppStage1Dataset
from datasets.scannetpp import SemanticSegmentationDataset

try:
    from thes.paths import scannetpp_raw_dir
except ImportError:
    scannetpp_raw_dir = Path("bigdata/scannetpp")


def generate_colors(n):
    return np.array([colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.8, 0.9) for i in range(n)]) * 255


def save_pcd(coords, colors, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors / 255.0, 0, 1))
    if not o3d.io.write_point_cloud(str(path), pcd):
        raise RuntimeError(f"Failed to save {path}")


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
    
    segments_path = raw_mesh_path.parent / "segments.json"
    segments_anno_path = raw_mesh_path.parent / "segments_anno.json"
    
    if segments_path.exists() and segments_anno_path.exists():
        with open(segments_path, 'r') as f:
            segments_data = json.load(f)
        vertex_to_segment = np.array(segments_data["segIndices"])
        
        with open(segments_anno_path, 'r') as f:
            annotations = json.load(f)
        
        max_segment_id = vertex_to_segment.max() + 1
        segment_to_instance = np.full(max_segment_id, -1, dtype=np.int32)
        
        for instance_id, anno in enumerate(annotations["segGroups"]):
            segment_ids = np.array(anno["segments"])
            segment_to_instance[segment_ids] = instance_id
        
        vertex_to_instance = segment_to_instance[vertex_to_segment]
        labels = np.column_stack([np.zeros_like(vertex_to_instance), vertex_to_instance])
    else:
        labels = np.full((len(coords), 2), -1, dtype=np.int32)
    
    return coords, colors, labels


def infer(model, coords, colors, cfg, device):
    from albumentations import Normalize
    
    normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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


def process_data(coords, colors, labels, model, cfg, device, output_dir, prefix):
    save_pcd(coords, colors, output_dir / f"{prefix}_rgb.pcd")
    
    if labels is not None:
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) > 0:
            label_colors = np.zeros_like(coords)
            palette = generate_colors(len(unique_labels))
            for i, label in enumerate(unique_labels):
                label_colors[labels == label] = palette[i]
            save_pcd(coords, label_colors, output_dir / f"{prefix}_gt.pcd")
    
    scores, masks = infer(model, coords, colors, cfg, device)
    if len(scores) > 0:
        pred_colors = np.zeros_like(coords)
        palette = generate_colors(len(scores))
        for i, mask in enumerate(masks):
            pred_colors[mask] = palette[i]
        save_pcd(coords, pred_colors, output_dir / f"{prefix}_pred.pcd")
    
    return len(unique_labels) if labels is not None else 0, len(scores)


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
    
    # Process frame
    coords, colors, labels = ScannetppStage1Dataset.load_specific_frame(args.scene_id, args.frame_id)
    gt_count, pred_count = process_data(
        coords, colors, labels[:, 1], model, cfg, device, output_dir, 
        f"{args.scene_id}_{args.frame_id}_frame"
    )
    print(f"Frame: {gt_count} GT instances, {pred_count} predictions")
    
    # Process preprocessed scene
    coords, colors, labels = SemanticSegmentationDataset.load_specific_scene(args.scene_id)
    gt_count, pred_count = process_data(
        coords, colors, labels[:, 1], model, cfg, device, output_dir,
        f"{args.scene_id}_preprocessed"
    )
    print(f"Preprocessed scene: {gt_count} GT instances, {pred_count} predictions")
    
    # Process raw scene
    coords, colors, labels = load_scene_raw(args.scene_id)
    gt_count, pred_count = process_data(
        coords, colors, labels[:, 1], model, cfg, device, output_dir,
        f"{args.scene_id}_raw"
    )
    print(f"Raw scene: {gt_count} GT instances, {pred_count} predictions")
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()