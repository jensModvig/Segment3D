import os
import time
import subprocess
import torch

def wait_for_gpu_memory(required_gb_per_gpu=8.0, check_interval=30, max_wait=3600):
    """
    Wait for required GPU memory and reserve it incrementally.
    
    Args:
        required_gb_per_gpu: Required memory per GPU in GB
        check_interval: Time between checks in seconds
        max_wait: Maximum wait time in seconds
        
    Returns:
        tuple: (target_gpus, reserved_tensors) or (None, {}) if timeout
    """
    def get_target_gpus():
        return list(range(torch.cuda.device_count()))
    
    def get_physical_gpu_id(device_id):
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            physical_ids = [int(x.strip()) for x in cuda_devices.split(",") if x.strip().isdigit()]
            if device_id < len(physical_ids):
                return physical_ids[device_id]
        return device_id
    
    def get_gpu_free_memory(device_id):
        physical_id = get_physical_gpu_id(device_id)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', f'--id={physical_id}'],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip()) / 1024
        except:
            return 0.0
    
    def reserve_memory_increment(device_id, gb):
        try:
            tensor_size = int((gb * 1024**3) / 4)
            tensor = torch.randn(tensor_size, dtype=torch.float32, device=f'cuda:{device_id}')
            return tensor
        except:
            return None
    
    target_gpus = get_target_gpus()
    if not target_gpus:
        return None, {}
    
    # Debug: Print GPU mapping
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    for i in target_gpus:
        physical_id = get_physical_gpu_id(i)
        print(f"PyTorch GPU{i} -> Physical GPU{physical_id}")
    
    reserved_tensors = {gpu_id: [] for gpu_id in target_gpus}
    reserved_amounts = {gpu_id: 0.0 for gpu_id in target_gpus}
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        all_ready = True
        
        for gpu_id in target_gpus:
            free_gb = get_gpu_free_memory(gpu_id)
            total_available = free_gb + reserved_amounts[gpu_id]
            
            if total_available < required_gb_per_gpu:
                all_ready = False
        
        if all_ready:
            print(f"All GPUs ready! Starting training...")
            return target_gpus, reserved_tensors
        
        for gpu_id in target_gpus:
            free_gb = get_gpu_free_memory(gpu_id)
            needed_gb = max(0, required_gb_per_gpu - reserved_amounts[gpu_id])
            increment_gb = min(free_gb * 0.9, needed_gb)
            
            if increment_gb >= 0.5:
                tensor = reserve_memory_increment(gpu_id, increment_gb)
                if tensor is not None:
                    reserved_tensors[gpu_id].append(tensor)
                    reserved_amounts[gpu_id] += increment_gb
        
        print(f"GPU Status: ", end="")
        for gpu_id in target_gpus:
            physical_id = get_physical_gpu_id(gpu_id)
            free_gb = get_gpu_free_memory(gpu_id)
            reserved = reserved_amounts[gpu_id]
            missing = max(0, required_gb_per_gpu - (free_gb + reserved))
            print(f"GPU{gpu_id}(Phys{physical_id})[Free:{free_gb:.1f}GB Reserved:{reserved:.1f}GB Missing:{missing:.1f}GB] ", end="")
        print()
        
        time.sleep(check_interval)
    
    for gpu_tensors in reserved_tensors.values():
        del gpu_tensors[:]
    torch.cuda.empty_cache()
    return None, {}

def clear_reserved_memory(reserved_tensors):
    """Clear all reserved GPU memory tensors."""
    for gpu_tensors in reserved_tensors.values():
        del gpu_tensors[:]
    torch.cuda.empty_cache()

class GPUMemoryManager:
    """Context manager for GPU memory reservation and cleanup."""
    
    def __init__(self, required_gb_per_gpu=8.0, check_interval=30, max_wait=3600):
        self.required_gb_per_gpu = required_gb_per_gpu
        self.check_interval = check_interval
        self.max_wait = max_wait
        self.reserved_tensors = {}
        self.ready_gpus = None
    
    def __enter__(self):
        self.ready_gpus, self.reserved_tensors = wait_for_gpu_memory(
            self.required_gb_per_gpu, self.check_interval, self.max_wait
        )
        return self.ready_gpus, self.reserved_tensors
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_reserved_memory(self.reserved_tensors)