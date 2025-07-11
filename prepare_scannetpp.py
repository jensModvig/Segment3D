import numpy as np
from pathlib import Path
import sys
import json
from tqdm import tqdm
from thes.paths import iterate_scannetpp, scannetpp_raw_dir, scannetpp_processed_dir

pose_cache = {}

def load_pose_and_intrinsic(scene_id, frame_id):
    if scene_id not in pose_cache:
        with open(scannetpp_raw_dir / 'data' / scene_id / 'iphone' / 'pose_intrinsic_imu.json') as f:
            pose_cache[scene_id] = json.load(f)
    
    data = pose_cache[scene_id][frame_id]
    return np.array(data['aligned_pose']).reshape(4, 4), np.array(data["intrinsic"]).reshape(3, 3)

def sample_frames(max_frames_req, split):
    scene_paths = list(iterate_scannetpp(split))
    scene_frames = []
    
    for p, *_ in tqdm(scene_paths, desc=f"Counting frames ({split})"):
        frame_paths = list((scannetpp_processed_dir / 'data' / p.name / 'iphone' / 'rgb').glob('*.jpg'))
        if frame_paths:
            frame_names = [f.stem for f in frame_paths]
            scene_frames.append((p.name, frame_names))
        else:
            raise FileNotFoundError('Empty scene', p.name)
    
    total_available = sum(len(frame_names) for _, frame_names in scene_frames)
    if total_available < max_frames_req:
        raise ValueError(f"Training requires {max_frames_req} frames but only {total_available} available")
    
    if split == 'val':
        val_max_frames = round(max_frames_req * (21.877/78.123))
        if total_available < val_max_frames:
            raise ValueError(f"Validation requires {val_max_frames} frames but only {total_available} available")
        max_frames = val_max_frames
    else:
        max_frames = max_frames_req
    
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
        depth_path = scannetpp_processed_dir / 'data' / scene_id / 'iphone' / 'depth' / f"{frame_id}.png"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth frame missing: {depth_path}")
        
        pose, intrinsic = load_pose_and_intrinsic(scene_id, frame_id)
        
        scene_poses_dir = poses_dir / scene_id
        scene_poses_dir.mkdir(exist_ok=True)
        np.savez_compressed(scene_poses_dir / f"{frame_id}.npz", pose=pose, intrinsics=intrinsic)

def main(max_frames):
    for split in ['train', 'val']:
        sample_frames(max_frames, split)

if __name__ == '__main__':
    main(int(sys.argv[1]))