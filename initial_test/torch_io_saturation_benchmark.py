import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def log2_tick_formatter(x, pos):
    return f"$2^{{{int(np.log2(x))+10}}}$" if x > 0 else "0"

# Configuration
mnt_path = "/mnt/kvcache"
result_path = "./results"
os.makedirs(mnt_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

tensor_sizes = [2 ** i for i in range(10, 31)]  # 2^10 to 2^30
dtype = np.float16
dtype_size = np.dtype(dtype).itemsize
num_gpus = torch.cuda.device_count()
saturation_duration = 60.0  # seconds per test

def prepare_tensor_file(size_bytes):
    path = os.path.join(mnt_path, f"tensor_{size_bytes}.bin")
    if not os.path.exists(path):
        arr = np.random.randn(size_bytes // dtype_size).astype(dtype)
        with open(path, "wb") as f:
            f.write(arr.tobytes())
    return path

def saturated_transfer(size_bytes, gpu_id, use_pinned):
    path = prepare_tensor_file(size_bytes)
    with open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=dtype).copy()
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    total_bytes = 0
    start = time.time()
    while (time.time() - start) < saturation_duration:
        if use_pinned:
            host_tensor = torch.from_numpy(arr).pin_memory()
            tensor = host_tensor.to(device, non_blocking=True)
        else:
            tensor = torch.from_numpy(arr).to(device)
        torch.cuda.synchronize()
        total_bytes += size_bytes
        del tensor
        torch.cuda.empty_cache()
    elapsed = time.time() - start
    return total_bytes / elapsed / 1e9  # GB/s

# Run benchmark
for gpu_id in range(num_gpus):
    results_def = []
    results_pin = []
    print(f"\nSaturation benchmark on GPU {gpu_id}...")

    for size in tensor_sizes:
        try:
            bw_def = saturated_transfer(size, gpu_id, use_pinned=False)
            bw_pin = saturated_transfer(size, gpu_id, use_pinned=True)
            results_def.append((size / 1024, bw_def))
            results_pin.append((size / 1024, bw_pin))
        except Exception as e:
            print(f"Error on size {size} for GPU {gpu_id}: {e}")
            results_def.append((size / 1024, 0))
            results_pin.append((size / 1024, 0))
            torch.cuda.empty_cache()

    # Plot
    x_def, y_def = zip(*results_def)
    x_pin, y_pin = zip(*results_pin)

    plt.figure(figsize=(12, 7))
    plt.plot(x_def, y_def, label="Default Transfer", marker="o")
    plt.plot(x_pin, y_pin, label="Pinned Transfer", marker="s")
    plt.title(f"Sustained Disk → GPU Bandwidth (cuda:{gpu_id}, {saturation_duration}s saturation)")
    plt.xlabel("Transfer Size (KB)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.xscale("log", base=2)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(log2_tick_formatter))
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(result_path, f"gpu_{gpu_id}_saturation_bandwidth.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved chart to {out_path}")
