import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Configuration
mnt_path = "/mnt/kvcache"
result_path = "./results"
os.makedirs(mnt_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

tensor_sizes = [2 ** i for i in range(10, 31)]  # 2^10 to 2^30 bytes
dtype = np.float16
dtype_size = np.dtype(dtype).itemsize
num_gpus = torch.cuda.device_count()
repeats = 10

def prepare_tensor_file(size_bytes):
    path = os.path.join(mnt_path, f"tensor_{size_bytes}.bin")
    if not os.path.exists(path):
        arr = np.random.randn(size_bytes // dtype_size).astype(dtype)
        with open(path, "wb") as f:
            f.write(arr.tobytes())
    return path

def measure_bandwidth(size_bytes, gpu_id):
    path = prepare_tensor_file(size_bytes)
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    ssd_times, gpu_times, total_times = [], [], []

    for _ in range(repeats):
        # SSD → CPU
        start_ssd = time.time()
        with open(path, "rb") as f:
            raw = f.read()
        arr = np.frombuffer(raw, dtype=dtype).copy()
        end_ssd = time.time()

        # CPU → GPU
        pinned_tensor = torch.from_numpy(arr).pin_memory()
        start_gpu = time.time()
        tensor = pinned_tensor.to(device, non_blocking=True)
        torch.cuda.synchronize()
        end_gpu = time.time()

        ssd_times.append(end_ssd - start_ssd)
        gpu_times.append(end_gpu - start_gpu)
        total_times.append((end_gpu - start_ssd))  # full pipeline

        del tensor
        torch.cuda.empty_cache()

    def bw(times): return size_bytes / np.mean(times) / 1e9  # GB/s

    return {
        "ssd_bw": bw(ssd_times),
        "gpu_bw": bw(gpu_times),
        "total_bw": bw(total_times),
    }

def log2_tick_formatter(x, pos):
    return f"$2^{{{int(np.log2(x)+10)}}}$" if x > 0 else "0"

# Run benchmark and plot
for gpu_id in range(num_gpus):
    sizes_kb = []
    ssd_bws = []
    gpu_bws = []
    total_bws = []

    print(f"\nRunning pinned memory benchmark on GPU {gpu_id}...")

    for size in tensor_sizes:
        try:
            result = measure_bandwidth(size, gpu_id)
            sizes_kb.append(size / 1024)
            ssd_bws.append(result["ssd_bw"])
            gpu_bws.append(result["gpu_bw"])
            total_bws.append(result["total_bw"])
        except Exception as e:
            print(f"Error at size {size}: {e}")
            sizes_kb.append(size / 1024)
            ssd_bws.append(0)
            gpu_bws.append(0)
            total_bws.append(0)
            torch.cuda.empty_cache()

    # Plot combined chart
    plt.figure(figsize=(12, 7))
    plt.plot(sizes_kb, ssd_bws, label="SSD → CPU", marker="o")
    plt.plot(sizes_kb, gpu_bws, label="CPU → GPU", marker="s")
    plt.plot(sizes_kb, total_bws, label="Total SSD → GPU", marker="^")

    plt.xscale("log", base=2)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(log2_tick_formatter))
    plt.xlabel("Transfer Size (KB)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title(f"Combined SSD → GPU Bandwidth (Pinned, cuda:{gpu_id})")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(result_path, f"gpu_{gpu_id}_combined_bandwidth.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

