import os
import time
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import asyncio
import aiofiles

mnt_path = "/mnt/kvcache"
result_path = "./results"
os.makedirs(mnt_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

tensor_sizes = [2 ** i for i in range(10, 31)]  # 2^10 to 2^30
dtype = np.float16
dtype_size = np.dtype(dtype).itemsize
repeats = 5
num_gpus = torch.cuda.device_count()

def prepare_tensor_file(size_bytes):
    path = os.path.join(mnt_path, f"tensor_{size_bytes}.bin")
    if not os.path.exists(path):
        arr = np.random.randn(size_bytes // dtype_size).astype(dtype)
        with open(path, "wb") as f:
            f.write(arr.tobytes())
    return path

async def read_with_aiofiles(path):
    async with aiofiles.open(path, 'rb') as f:
        return await f.read()

def measure_bandwidth(size_bytes, gpu_id, use_pinned, use_aio=False, drop_caches=False):
    path = prepare_tensor_file(size_bytes)
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    ssd_times, gpu_times, total_times = [], [], []

    for _ in range(repeats):
        if drop_caches:
            os.system("sync; echo 3 > /proc/sys/vm/drop_caches")

        start_ssd = time.time()
        if use_aio:
            print("Using aiofiles for async read")
            raw = asyncio.run(read_with_aiofiles(path))
        else:
            print("Using synchronous read")
            with open(path, "rb") as f:
                raw = f.read()
        arr = np.frombuffer(raw, dtype=dtype).copy()
        end_ssd = time.time()

        host_tensor = torch.from_numpy(arr).pin_memory() if use_pinned else torch.from_numpy(arr)
        start_gpu = time.time()
        tensor = host_tensor.to(device, non_blocking=use_pinned)
        torch.cuda.synchronize()
        end_gpu = time.time()

        ssd_times.append(end_ssd - start_ssd)
        gpu_times.append(end_gpu - start_gpu)
        total_times.append(end_gpu - start_ssd)

        del tensor
        torch.cuda.empty_cache()

    def bw(times): return size_bytes / np.mean(times) / 1e9
    return {
        "ssd_bw": bw(ssd_times),
        "gpu_bw": bw(gpu_times),
        "total_bw": bw(total_times)
    }

def run_all(args):
    for gpu_id in range(num_gpus):
        results = {
            "default": [],
            "pinned": []
        }

        for size in tensor_sizes:
            print(f"GPU {gpu_id} Size {size}...")
            try:
                res_def = measure_bandwidth(size, gpu_id, use_pinned=False,
                                            use_aio=args.aio, drop_caches=args.drop_caches)
                res_pin = measure_bandwidth(size, gpu_id, use_pinned=True,
                                            use_aio=args.aio, drop_caches=args.drop_caches)
                results["default"].append((size / 1024, res_def))
                results["pinned"].append((size / 1024, res_pin))
            except Exception as e:
                print(f"Error at {size}: {e}")
                results["default"].append((size / 1024, {"ssd_bw": 0, "gpu_bw": 0, "total_bw": 0}))
                results["pinned"].append((size / 1024, {"ssd_bw": 0, "gpu_bw": 0, "total_bw": 0}))
                torch.cuda.empty_cache()

        # Save CSV
        csv_path = os.path.join(result_path, f"gpu_{gpu_id}_bandwidth.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Size_KB", "Mode", "SSD_BW", "GPU_BW", "Total_BW"])
            for mode in ["default", "pinned"]:
                for size, r in results[mode]:
                    writer.writerow([size, mode, r["ssd_bw"], r["gpu_bw"], r["total_bw"]])
        print(f"Saved CSV: {csv_path}")

        # Plotting
        def log2_tick_formatter(x, pos):
            return f"$2^{{{int(np.log2(x))}}}$" if x > 0 else "0"

        for key in ["ssd_bw", "gpu_bw", "total_bw"]:
            plt.figure(figsize=(12, 7))
            for mode, marker in zip(["default", "pinned"], ["o", "s"]):
                x = [size for size, _ in results[mode]]
                y = [r[key] for _, r in results[mode]]
                plt.plot(x, y, label=mode.capitalize(), marker=marker)
            plt.xscale("log", base=2)
            plt.gca().xaxis.set_major_formatter(FuncFormatter(log2_tick_formatter))
            plt.xlabel("Transfer Size (KB)")
            plt.ylabel("Bandwidth (GB/s)")
            plt.title(f"{key.replace('_', ' ').title()} Comparison (GPU {gpu_id})")
            plt.grid(True, which="both")
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(result_path, f"gpu_{gpu_id}_{key}_comparison.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-caches", action="store_true", help="Drop Linux caches before each read")
    parser.add_argument("--aio", action="store_true", help="Use aiofiles for async disk reads")
    args = parser.parse_args()
    run_all(args)
