import numpy as np
from pathlib import Path
import sys
import json
import re
import os
from tqdm import tqdm
from thes.paths import iterate_scannetpp, scannetpp_raw_dir, scannetpp_processed_dir

pose_cache = {}

def natural_sort(paths):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda path: [convert(c) for c in re.split('(\d+)', os.path.basename(str(path)))]
    return sorted(paths, key=alphanum_key)

def load_pose_and_intrinsic(scene_id, frame_id):
    if scene_id not in pose_cache:
        with open(scannetpp_raw_dir / 'data' / scene_id / 'iphone' / 'pose_intrinsic_imu.json') as f:
            pose_cache[scene_id] = json.load(f)
    data = pose_cache[scene_id][frame_id]
    return np.array(data['aligned_pose']).reshape(4, 4), np.array(data["intrinsic"]).reshape(3, 3)

def sample_frames(max_frames_req, split, skip_scenes=None):
    if skip_scenes is None:
        skip_scenes = set()
    else:
        skip_scenes = set(skip_scenes)
    
    scene_paths = list(iterate_scannetpp(split))
    scene_frames = []
    
    for p, *_ in tqdm(scene_paths, desc=f"Counting frames ({split})"):
        if p.name in skip_scenes:
            continue
        
        frame_paths = list((scannetpp_processed_dir / 'data' / p.name / 'iphone' / 'rgb').glob('*.jpg'))
        if frame_paths:
            frame_paths = natural_sort(frame_paths)
            frame_names = [f.stem for f in frame_paths]
            scene_frames.append((p.name, frame_names))
        else:
            raise FileNotFoundError('Empty scene', p.name)
    
    total_available = sum(len(frame_names) for _, frame_names in scene_frames)
    
    if split == 'val':
        val_max_frames = round(max_frames_req * (21.877/78.123))
        if total_available < val_max_frames:
            print(f"WARNING - Validation requires {val_max_frames} frames but only {total_available} available")
            max_frames = total_available
        else:
            max_frames = val_max_frames
    else:
        max_frames = max_frames_req
    
    if total_available < max_frames:
        raise ValueError(f"Training requires {max_frames} frames but only {total_available} available")
    
    factor = max_frames / total_available
    selected = []
    
    for scene_id, frame_names in scene_frames:
        n_select = round(len(frame_names) * factor)
        indices = np.linspace(0, len(frame_names) - 1, n_select, dtype=int)
        selected.extend((scene_id, frame_names[i]) for i in indices)
    
    output_dir = Path('/work3/s173955/Segment3D/data/processed/scannetpp_info')
    output_dir.mkdir(parents=True, exist_ok=True)
    poses_dir = output_dir / 'poses'
    poses_dir.mkdir(exist_ok=True)
    
    with open(output_dir / f'scannetpp_{max_frames_req}_{split}.txt', 'w') as f:
        f.writelines(f"{scene} {frame}\n" for scene, frame in selected)
    
    for scene_id, frame_id in tqdm(selected, desc=f"Saving pose and intrinsics."):
        scene_poses_dir = poses_dir / scene_id
        scene_poses_dir.mkdir(exist_ok=True)
        intrinsic_pose_path = scene_poses_dir / f"{frame_id}.npz"
        
        if intrinsic_pose_path.exists():
            continue
        
        depth_path = scannetpp_processed_dir / 'data' / scene_id / 'iphone' / 'depth' / f"{frame_id}.png"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth frame missing: {depth_path}")
        
        pose, intrinsic = load_pose_and_intrinsic(scene_id, frame_id)
        np.savez_compressed(intrinsic_pose_path, pose=pose, intrinsics=intrinsic)

def main(max_frames, skip_scenes=None):
    for split in ['train', 'val']:
        sample_frames(max_frames, split, skip_scenes)

if __name__ == '__main__':
    max_frames = int(sys.argv[1])
    skip_scenes = [s.strip() for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
    main(max_frames, skip_scenes)