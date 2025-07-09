import argparse
import numpy as np
import open3d as o3d
import colorsys
from pathlib import Path
from datasets.scannet_stage1 import SemanticSegmentationDataset
from datasets.scannetpp_stage1 import ScannetppStage1Dataset


def generate_colors(n_labels):
    return np.array([colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.8, 0.9) for i in range(n_labels)])


def save_sample(coords, colors, labels, scene_frame_id, output_dir, dataset_name, suffix, use_labels=False):
    filename = output_dir / f"{dataset_name}_{scene_frame_id.replace('/', '_')}_{suffix}.pcd"
    
    if use_labels:
        instance_labels = labels[:, 1]
        unique_labels = np.unique(instance_labels)
        palette = generate_colors(len(unique_labels))
        point_colors = np.zeros((len(coords), 3))
        for i, label in enumerate(unique_labels):
            point_colors[instance_labels == label] = palette[i]
    else:
        point_colors = colors / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(str(filename), pcd)


def process_dataset(dataset_name, dataset, output_dir, num_samples):
    for i in range(num_samples):
        raw_sample = dataset.get_raw_sample(i)
        aug_sample = dataset[i]
        
        coords, _, labels, scene_frame_id, raw_color = raw_sample[:5]
        
        save_sample(coords, raw_color, labels, scene_frame_id, output_dir, dataset_name, "raw")
        save_sample(coords, raw_color, labels, scene_frame_id, output_dir, dataset_name, "raw_labels", True)
        
        coords, _, labels, scene_frame_id, raw_color = aug_sample[:5]
        
        save_sample(coords, raw_color, labels, scene_frame_id, output_dir, dataset_name, "augmented")
        save_sample(coords, raw_color, labels, scene_frame_id, output_dir, dataset_name, "augmented_labels", True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2)
    args = parser.parse_args()
    
    output_dir = Path("debug")
    output_dir.mkdir(exist_ok=True)
    
    datasets = [
        ("scannet", SemanticSegmentationDataset(
            mode="train",
            max_frames=10,
            point_per_cut=0,
            data_dir="data/processed"
        )),
        ("scannetpp", ScannetppStage1Dataset(
            mode="train",
            max_frames=10,
            point_per_cut=0,
            scenes_to_exclude="dfac5b38df,00dd871005,c4c04e6d6c"
        ))
    ]
    
    for dataset_name, dataset in datasets:
        process_dataset(dataset_name, dataset, output_dir, args.num_samples)


if __name__ == "__main__":
    main()