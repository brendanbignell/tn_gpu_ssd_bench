import os
import time
import numpy as np
import cupy as cp
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

from kvikio import CuFile

#mnt_path = "/tmp"
mnt_path = "/mnt/kvcache"
result_path = "./results"
os.makedirs(result_path, exist_ok=True)
total_transfer_bytes = 1 * 1024 ** 3  # 1 GB per batch size

np_dtype = np.float16
cp_dtype = cp.float16
dtype_size = np.dtype(np_dtype).itemsize

#num_gpus = cp.cuda.runtime.getDeviceCount()
num_gpus = 1
# Use larger batches to avoid GDS/PyTorch bug (1MB up)
batch_sizes = [2 ** i for i in range(28, 29)]  # 1MB to 1GB
repeats = 1

def log(msg):
    now = time.strftime('%H:%M:%S')
    print(f"[{now}] {msg}", flush=True)

def prepare_tensor_file(path, size_bytes, pbar=None):
    if not os.path.exists(path):
        log(f"Creating tensor file {path} ({size_bytes/1e6:.1f} MB)")
        chunk_size = 8 * 1024 * 1024  # 8 MB
        with open(path, "wb") as f:
            written = 0
            while written < size_bytes:
                remain = size_bytes - written
                this_chunk = min(chunk_size, remain)
                arr = np.random.randn(this_chunk // dtype_size).astype(np_dtype)
                f.write(arr.tobytes())
                written += this_chunk
                if pbar is not None:
                    pbar.update(this_chunk)
        log(f"Finished creating {path}")

def kvikio_multi_gpu_benchmark(filespecs, batch_sizes, total_bytes, repeats):
    results = []
    for gpu_id, (file_path, _) in enumerate(filespecs):
        cp.cuda.Device(gpu_id).use()
        batch_results = []
        for bidx, batch_size in enumerate(batch_sizes):
            if batch_size % dtype_size != 0:
                raise ValueError(f"Batch size {batch_size} is not a multiple of dtype size {dtype_size}")
            n_batches = total_bytes // batch_size
            gds_bandwidths = []
            pipeline_times = []
            for run in range(repeats):
                show_pbar = (run == 0)
                pbar = tqdm(total=n_batches, position=gpu_id, desc=f"Kvikio GDS GPU {gpu_id} {batch_size//1024} KB Run {run+1}", leave=show_pbar, ncols=80, disable=not show_pbar)
                cp.cuda.Stream.null.synchronize()
                pipeline_start = time.time()
                n_elements = batch_size // dtype_size
                gpu_array = cp.empty((n_elements,), dtype=cp_dtype)
                for batch_idx in range(n_batches):
                    offset = batch_idx * batch_size
                    if batch_idx == 0 and run == 0 and bidx == 0:
                        log(f"[DEBUG] GPU {gpu_id} batch_size={batch_size}, n_elements={n_elements}, array_bytes={gpu_array.nbytes}, offset=0")
                    with CuFile(file_path, "rb") as f:
                        f.read(gpu_array, offset, batch_size)
                    cp.cuda.Stream.null.synchronize()
                    if show_pbar:
                        pbar.update(1)
                pipeline_end = time.time()
                if show_pbar:
                    pbar.close()
                pipeline_time = pipeline_end - pipeline_start
                gds_bandwidths.append(total_bytes / pipeline_time / 1e9)
                pipeline_times.append(pipeline_time)
    return results

def save_csv_and_plot_kvikio(gpu_id, batch_results):
    csv_path = os.path.join(result_path, f"gpu_{gpu_id}_kvikio_bandwidth.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "BatchSize_KB",
            "GDS_BW_Mean_GBs", "GDS_BW_Min_GBs", "GDS_BW_Max_GBs", "GDS_BW_Std_GBs",
            "Pipeline_Time_Mean_s", "Pipeline_Time_Min_s", "Pipeline_Time_Max_s", "Pipeline_Time_Std_s",
            "NumBatches"
        ])
        for stat in batch_results:
            writer.writerow([
                stat["batch_size"] // 1024,
                stat["gds_bw_mean"], stat["gds_bw_min"], stat["gds_bw_max"], stat["gds_bw_std"],
                stat["pipeline_time_mean"], stat["pipeline_time_min"], stat["pipeline_time_max"], stat["pipeline_time_std"],
                stat["n_batches"]
            ])
    log(f"Saved CSV for GPU {gpu_id} to {csv_path}")

    batch_sizes_kb = [stat["batch_size"] // 1024 for stat in batch_results]
    gds_mean = np.array([stat["gds_bw_mean"] for stat in batch_results])
    gds_std = np.array([stat["gds_bw_std"] for stat in batch_results])

    plt.figure(figsize=(10, 7))
    plt.errorbar(batch_sizes_kb, gds_mean, yerr=gds_std, label="Kvikio GDS SSD→VRAM", fmt='-o', capsize=5)
    plt.xscale("log", base=2)
    plt.xlabel("Batch Size (KB)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title(f"Kvikio Disk→GPU Bandwidth (GPU {gpu_id})")
    plt.legend()
    plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(result_path, f"gpu_{gpu_id}_kvikio_bandwidth.png")
    plt.savefig(plot_path)
    plt.close()
    log(f"Saved plot for GPU {gpu_id} to {plot_path}")

def main():
    # --- File creation with progress bars ---
    log("Preparing files for each GPU with progress bars")
    filespecs = []
    file_creation_pbars = []
    for gpu_id in range(num_gpus):
        path = os.path.join(mnt_path, f"tensor_max_{gpu_id}.bin")
        size_bytes = batch_sizes[-1]  # 1GB, largest batch size
        filespecs.append((path, size_bytes))
        file_creation_pbars.append(
            tqdm(total=size_bytes, position=gpu_id, desc=f"File {gpu_id}", ncols=80)
        )
    for idx, (path, size_bytes) in enumerate(filespecs):
        prepare_tensor_file(path, size_bytes, pbar=file_creation_pbars[idx])
        file_creation_pbars[idx].close()
    log("File creation complete")

    # --- Run Kvikio benchmarks ---
    log(f"Starting Kvikio GDS benchmark: {num_gpus} GPUs, batch sizes {batch_sizes[0]//1024} KB to {batch_sizes[-1]//1024**2} GB, {total_transfer_bytes//1024**2} MB per size, {repeats} runs")
    t0 = time.time()
    results = kvikio_multi_gpu_benchmark(filespecs, batch_sizes, total_transfer_bytes, repeats)
    elapsed = time.time() - t0

    for gpu_id, batch_results in results:
        save_csv_and_plot_kvikio(gpu_id, batch_results)
        for stat in batch_results:
            log(
                f"[RESULT] GPU {gpu_id} | Batch {stat['batch_size']//1024} KB: "
                f"Kvikio GDS: {stat['gds_bw_mean']:.2f}±{stat['gds_bw_std']:.2f} GB/s, "
                f"{stat['n_batches']} batches, {stat['pipeline_time_mean']:.2f} s"
            )

    log(f"Total elapsed wall time: {elapsed:.1f} s")

if __name__ == "__main__":
    main()
