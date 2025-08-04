import argparse
import json
import colorsys
import shutil
import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from hydra.experimental import initialize, compose
from demo_utils import get_model, prepare_data
from datasets.scannetpp_stage1 import ScannetppStage1Dataset
from datasets.scannetpp import SemanticSegmentationDataset
from datasets.stage1 import Stage1Dataset as Stage1default
from datasets.stage1_conf import Stage1Dataset as Stage1conf
from datasets.stage1_conf_threshold import Stage1DatasetConf as Stage1conf_threshold 

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
    o3d.io.write_point_cloud(str(path), pcd)


def load_scene_raw(scene_id, frame_id=None):
    mesh_path = scannetpp_raw_dir / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    coords = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) * 255 if len(mesh.vertex_colors) else np.full((len(coords), 3), 255)
    
    segments_path = mesh_path.parent / "segments.json"
    segments_anno_path = mesh_path.parent / "segments_anno.json"
    
    if segments_path.exists() and segments_anno_path.exists():
        with open(segments_path) as f:
            vertex_to_segment = np.array(json.load(f)["segIndices"])
        with open(segments_anno_path) as f:
            annotations = json.load(f)
        
        segment_to_instance = np.full(vertex_to_segment.max() + 1, -1, dtype=np.int32)
        for instance_id, anno in enumerate(annotations["segGroups"]):
            segment_to_instance[np.array(anno["segments"])] = instance_id
        
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


def process_dataset(name, loader, scene_id, frame_id, model, cfg, device, output_dir):
    coords, colors, labels = loader(scene_id, frame_id)
    prefix = f"{scene_id}_{frame_id}_{name}" if frame_id else f"{scene_id}_{name}"
    
    save_pcd(coords, colors, output_dir / f"{prefix}_rgb.pcd")
    
    gt_count = 0
    if labels is not None:
        instance_labels = labels[:, 1] if labels.ndim > 1 else labels
        unique_labels = np.unique(instance_labels[instance_labels >= 0])
        if len(unique_labels) > 0:
            gt_count = len(unique_labels)
            label_colors = np.zeros_like(coords)
            palette = generate_colors(gt_count)
            for i, label in enumerate(unique_labels):
                label_colors[instance_labels == label] = palette[i]
            save_pcd(coords, label_colors, output_dir / f"{prefix}_gt.pcd")
    
    scores, masks = infer(model, coords, colors, cfg, device)
    pred_count = len(scores)
    if pred_count > 0:
        pred_colors = np.zeros_like(coords)
        palette = generate_colors(pred_count)
        for i, mask in enumerate(masks):
            pred_colors[mask] = palette[i]
        save_pcd(coords, pred_colors, output_dir / f"{prefix}_pred.pcd")
    
    print(f"{name}: {gt_count} GT, {pred_count} pred")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--output_dir", default="debug_predictions")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    with initialize(config_path="conf"):
        cfg = compose(config_name="config_base_instance_segmentation.yaml")
    
    cfg.general.checkpoint = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).eval().to(device)
    
    datasets = [
        ("scannetpp_gt_stage1", lambda s, f: ScannetppStage1Dataset.load_specific_frame(s, f)),
        # ("stage1_gt_depth256", lambda s, f: Stage1Dataset.load_specific_frame(
        #     scene_id=s,
        #     frame_id=f,
        #     sam_folder='gt_mask',
        #     color_folder='iphone/rgb',
        #     depth_folder='depth_pro/depth_map_fpx_256x192',
        #     intrinsic_folder='depth_pro/intrinsics_fpx_256x192'
        # )),
        # ("stage1_gt_depth640", lambda s, f: Stage1default.load_specific_frame(
        #     scene_id=s,
        #     frame_id=f,
        #     sam_folder='sam',
        #     color_folder='iphone/rgb',
        #     depth_folder='depth_pro/depth_map_fpx_640x480',
        #     intrinsic_folder='depth_pro/intrinsics_fpx_640x480'
        # )),
        ("stage1_gt_depth640_conf", lambda s, f: Stage1conf.load_specific_frame(
            scene_id=s,
            frame_id=f,
            sam_folder='gt_mask',
            color_folder='iphone/rgb',
            depth_folder='depth_pro/depth_map_fpx_640x480',
            intrinsic_folder='depth_pro/intrinsics_fpx_640x480'
        )),
        ("stage1_gt_depth640_conf_thresh", lambda s, f: Stage1conf_threshold.load_specific_frame(
            scene_id=s,
            frame_id=f,
            sam_folder='gt_mask',
            color_folder='iphone/rgb',
            depth_folder='depth_pro/depth_map_fpx_640x480',
            intrinsic_folder='depth_pro/intrinsics_fpx_640x480'
        )),
        # ("stage1_sam_depth256", lambda s, f: Stage1Dataset.load_specific_frame(
        #     scene_id=s,
        #     frame_id=f,
        #     sam_folder='sam',
        #     color_folder='iphone/rgb',
        #     depth_folder='depth_pro/depth_map_fpx_256x192',
        #     intrinsic_folder='depth_pro/intrinsics_fpx_256x192'
        # )),
        # ("stage1_sam_depth640", lambda s, f: Stage1Dataset.load_specific_frame(
        #     scene_id=s,
        #     frame_id=f,
        #     sam_folder='sam',
        #     color_folder='iphone/rgb',
        #     depth_folder='depth_pro/depth_map_fpx_640x480',
        #     intrinsic_folder='depth_pro/intrinsics_fpx_640x480'
        # )),
        # ("preprocessed", lambda s, f: SemanticSegmentationDataset.load_specific_scene(s)),
        # ("raw", load_scene_raw),
    ]
    
    dataset_names = [name for name, _ in datasets]
    assert len(dataset_names) == len(set(dataset_names)), f"Duplicate dataset names found: {[name for name in dataset_names if dataset_names.count(name) > 1]}"
    
    success_count = sum(process_dataset(name, loader, args.scene_id, args.frame_id, 
                                      model, cfg, device, output_dir) 
                       for name, loader in datasets)
    
    print(f"Processed {success_count}/{len(datasets)} datasets → {output_dir}")


if __name__ == "__main__":
    main()