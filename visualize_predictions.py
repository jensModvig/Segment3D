import argparse
import colorsys
import shutil
import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from hydra.experimental import initialize, compose
from demo_utils import get_model
from datasets.stage1_conf_visualization import Stage1DatasetConf
from datasets.stage1 import Stage1Dataset
from datasets.stage1_conf import Stage1DatasetConf as Stage1DatasetConfBase
import albumentations as A
import MinkowskiEngine as ME


def generate_colors(n):
    return np.array([colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.8, 0.9) for i in range(n)]) * 255


def save_pcd(coords, colors, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors / 255.0, 0, 1))
    o3d.io.write_point_cloud(str(path), pcd)


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
    scores, topk_indices = scores.topk(topk_count, sorted=True)
    
    # Get final binary masks
    final_masks = []
    for idx in topk_indices:
        mask = masks[:, idx][transformer.inverse_map]  # Apply inverse mapping
        binary_mask = (mask.sigmoid().numpy() > 0.5).astype(bool)
        final_masks.append(binary_mask)
    
    return scores.numpy(), final_masks


def infer(model, coords, colors, cfg, device):
    transformer = ModelSpaceTransformer(cfg, device)
    mock_mesh = type('M', (), {'vertices': coords, 'vertex_colors': colors / 255.0})()
    data = transformer.prepare_pointcloud(mock_mesh)
    
    scores, mask_list = get_model_masks(cfg, model, data, transformer)
    
    if len(mask_list) == 0:
        return np.array([]), np.zeros((len(coords), 0), dtype=bool)
    
    masks = np.stack(mask_list).T
    return scores, masks


def load_with_confidence_filtering(scene_id, frame_id):
    coords, colors, labels = Stage1DatasetConfBase.load_specific_frame(
        scene_id, frame_id, depth_folder='depth_pro/depth_map_fpxc_640x480'
    )
    confidence = labels[:, 3]
    confidence_mask = confidence >= 0.5
    return coords[confidence_mask], colors[confidence_mask], labels[confidence_mask]


def process_dataset(name, loader, scene_id, frame_id, model, cfg, device, output_dir):
    if name == "sam_confidence_filtered":
        coords, colors, labels = load_with_confidence_filtering(scene_id, frame_id)
    else:
        coords, colors, labels = loader(scene_id, frame_id)
    
    prefix = f"{scene_id}_{frame_id}_{name}"
    
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
    
    valid_indices = [i for i, score in enumerate(scores) if score > 0.7]
    if valid_indices:
        scores, masks = scores[valid_indices], masks[:, valid_indices]
    else:
        scores, masks = np.array([]), np.array([]).reshape(0, masks.shape[1])
            
    pred_count = len(scores)
    if pred_count > 0:
        pred_colors = np.zeros_like(coords)
        palette = generate_colors(pred_count)
        for i, mask in enumerate(masks.T):
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
    cfg.general.train_on_segments = False
    cfg.model.num_queries = 100
    cfg.general.topk_per_image = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).eval().to(device)
    
    datasets = [
        ("scannetpp_raw", lambda s, f: Stage1Dataset.load_specific_frame(
            scene_id=s, frame_id=f, sam_folder='sam', color_folder='iphone/rgb',
            depth_folder='depth_pro/depth_map_fpx_640x480', intrinsic_folder='depth_pro/intrinsics_fpx_640x480'
        )),
        ("sam_confidence_viz", lambda s, f: Stage1DatasetConf.load_specific_frame(
            scene_id=s, frame_id=f, sam_folder='sam', color_folder='iphone/rgb',
            depth_folder='depth_pro/depth_map_fpxc_640x480', intrinsic_folder='depth_pro/intrinsics_fpxc_640x480'
        ))
    ]
    
    success_count = sum(process_dataset(name, loader, args.scene_id, args.frame_id, 
                                      model, cfg, device, output_dir) 
                       for name, loader in datasets)
    
    print(f"Processed {success_count}/{len(datasets)} datasets → {output_dir}")


if __name__ == "__main__":
    main()