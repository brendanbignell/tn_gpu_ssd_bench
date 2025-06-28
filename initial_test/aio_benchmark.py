import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# Setup
mnt_path = "/mnt/kvcache"
result_path = "./results"
os.makedirs(mnt_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)
tensor_sizes = [2 ** i for i in range(10, 29)]  # 1KB to 256MB
dtype = np.float16
dtype_size = np.dtype(dtype).itemsize
num_gpus = torch.cuda.device_count()
repeats = 100

def prepare_tensor_file(size_bytes):
    path = os.path.join(mnt_path, f"tensor_{size_bytes}.bin")
    if not os.path.exists(path):
        arr = np.random.randn(size_bytes // dtype_size).astype(dtype)
        with open(path, "wb") as f:
            f.write(arr.tobytes())
    return path

def run_transfer(arr, device, use_pinned):
    try:
        if use_pinned:
            host_tensor = torch.from_numpy(arr).pin_memory()
            start = time.time()
            _ = host_tensor.to(device, non_blocking=True)
        else:
            start = time.time()
            _ = torch.from_numpy(arr).to(device)
        torch.cuda.synchronize()
        return time.time() - start
    except Exception as e:
        print(f"Transfer error: {e}")
        torch.cuda.empty_cache()
        return None

def benchmark_transfer(size_bytes, gpu_id):
    file_path = prepare_tensor_file(size_bytes)
    with open(file_path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=dtype).copy()
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    def run_mode(pinned):
        times = []
        for _ in range(repeats):
            dt = run_transfer(arr, device, use_pinned=pinned)
            if dt is not None:
                times.append(dt)
            torch.cuda.empty_cache()
        times = np.array(times)
        if len(times) == 0:
            return 0, 0, 0, 0
        bw = size_bytes / times / 1e9
        return bw.mean(), bw.min(), bw.max(), bw.std()

    return run_mode(False), run_mode(True)

# Main loop
for gpu_id in range(num_gpus):
    stats_default = []
    stats_pinned = []
    print(f"Benchmarking GPU {gpu_id}...")
    for size in tensor_sizes:
        try:
            (m1, min1, max1, std1), (m2, min2, max2, std2) = benchmark_transfer(size, gpu_id)
            kb = size / 1024
            stats_default.append((kb, m1, min1, max1, std1))
            stats_pinned.append((kb, m2, min2, max2, std2))
        except Exception as e:
            print(f"Error at size {size} on GPU {gpu_id}: {e}")
            kb = size / 1024
            stats_default.append((kb, 0, 0, 0, 0))
            stats_pinned.append((kb, 0, 0, 0, 0))

    # Plot
    for label, stats in [("Default", stats_default), ("Pinned", stats_pinned)]:
        sizes_kb = [s[0] for s in stats]
        means = [s[1] for s in stats]
        mins = [s[2] for s in stats]
        maxs = [s[3] for s in stats]
        stds = [s[4] for s in stats]



    # Combined plot per GPU
    sizes_kb = [s[0] for s in stats_default]
    mean_def = [s[1] for s in stats_default]
    min_def = [s[2] for s in stats_default]
    max_def = [s[3] for s in stats_default]
    std_def = [s[4] for s in stats_default]

    mean_pin = [s[1] for s in stats_pinned]
    min_pin = [s[2] for s in stats_pinned]
    max_pin = [s[3] for s in stats_pinned]
    std_pin = [s[4] for s in stats_pinned]

    plt.figure(figsize=(12, 7))
    
    # Default
    plt.plot(sizes_kb, mean_def, label="Default: Mean", color="blue")
    plt.fill_between(sizes_kb, min_def, max_def, alpha=0.15, color="blue", label="Default: Min–Max")
    plt.errorbar(sizes_kb, mean_def, yerr=std_def, fmt='o', markersize=3, color="blue", label="Default: ±Stdev")

    # Pinned
    plt.plot(sizes_kb, mean_pin, label="Pinned: Mean", color="green")
    plt.fill_between(sizes_kb, min_pin, max_pin, alpha=0.15, color="green", label="Pinned: Min–Max")
    plt.errorbar(sizes_kb, mean_pin, yerr=std_pin, fmt='s', markersize=3, color="green", label="Pinned: ±Stdev")

    plt.title(f"Disk → GPU Transfer Bandwidth with Variance (cuda:{gpu_id})")
    plt.xlabel("Transfer Size (KB)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.xscale("log", base=2)
    plt.grid(True, which="both")
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    chart_path = os.path.join(result_path, f"gpu_{gpu_id}_bandwidth_detailed_combined.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved detailed comparison chart to {chart_path}")
