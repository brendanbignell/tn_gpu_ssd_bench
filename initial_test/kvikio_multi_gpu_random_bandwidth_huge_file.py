import os
import time
import numpy as np
import torch
from multiprocessing import Process, Queue, set_start_method
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

import kvikio
from kvikio import CuFile

mnt_path = "/mnt/kvcache"
result_path = "./results"
os.makedirs(result_path, exist_ok=True)

test_file = "/mnt/kvcache/huge_test_4tb.bin"
file_size = os.path.getsize(test_file)

dtype = np.float16
dtype_size = np.dtype(dtype).itemsize
num_gpus = torch.cuda.device_count()
batch_sizes = [2 ** i for i in range(24, 31)]  # 16MB to 1GB
repeats = 10

def log(msg):
    now = time.strftime('%H:%M:%S')
    print(f"[{now}] {msg}", flush=True)

def get_random_offset(batch_size, file_size):
    align = 1 * 1024 * 1024  # 1 MB alignment
    max_offset = file_size - batch_size
    n_pos = max_offset // align
    idx = np.random.randint(0, n_pos)
    return idx * align

def kvikio_gpu_benchmark(gpu_id, file_path, batch_sizes, total_bytes, repeats, result_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    batch_results = []

    for bidx, batch_size in enumerate(batch_sizes):
        n_batches = total_bytes // batch_size

        # Arrays to collect bandwidth for each repeat
        gds_bandwidths = []
        pipeline_times = []

        for run in range(repeats):
            show_pbar = (run == 0)
            pbar = tqdm(total=n_batches, position=gpu_id, desc=f"Kvikio GDS GPU {gpu_id} {batch_size//1024} KB Run {run+1}", leave=show_pbar, ncols=80, disable=not show_pbar)
            torch.cuda.synchronize()
            pipeline_start = time.time()
            with CuFile(file_path, "rb") as f:
                for batch_idx in range(n_batches):
                    gpu_tensor = torch.empty(batch_size // dtype_size, dtype=torch.float16, device=device)
                    offset = get_random_offset(batch_size, file_size)
                    t0 = time.time()
                    f.read(gpu_tensor, batch_size, offset)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    if show_pbar:
                        pbar.update(1)
                    del gpu_tensor
                    torch.cuda.empty_cache()
            pipeline_end = time.time()
            if show_pbar:
                pbar.close()

            pipeline_time = pipeline_end - pipeline_start
            gds_bandwidths.append(total_bytes / pipeline_time / 1e9)
            pipeline_times.append(pipeline_time)

        arr = np.array(gds_bandwidths)
        mean, minv, maxv, std = arr.mean(), arr.min(), arr.max(), arr.std()
        batch_stats = dict(
            batch_size=batch_size,
            n_batches=n_batches,
            gds_bw_mean=mean, gds_bw_min=minv, gds_bw_max=maxv, gds_bw_std=std,
            pipeline_time_mean=np.mean(pipeline_times), pipeline_time_min=np.min(pipeline_times), pipeline_time_max=np.max(pipeline_times), pipeline_time_std=np.std(pipeline_times),
        )
        batch_results.append(batch_stats)

        log(
            f"[GPU {gpu_id}] Batch {batch_size//1024} KB: "
            f"Kvikio GDS: {mean:.2f}±{std:.2f} GB/s"
        )
    result_q.put((gpu_id, batch_results))

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

    # Plot
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
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Already set

    log(f"Starting Kvikio GDS benchmark: {num_gpus} GPUs, batch sizes {batch_sizes[0]//1024} KB to {batch_sizes[-1]//1024**2} GB, {repeats} runs")
    result_queue = Queue()
    procs = []
    t0 = time.time()
    for gpu_id in range(num_gpus):
        p = Process(
            target=kvikio_gpu_benchmark,
            args=(gpu_id, test_file, batch_sizes, 1 * 1024 ** 3, repeats, result_queue),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    elapsed = time.time() - t0

    results = []
    while not result_queue.empty():
        gpu_id, batch_results = result_queue.get()
        results.append((gpu_id, batch_results))
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
