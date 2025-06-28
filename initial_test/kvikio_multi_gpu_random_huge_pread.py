import os
import time
import numpy as np
import torch
from multiprocessing import Process, Queue, set_start_method
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from kvikio import CuFile

# --- Config ---
test_file = "/mnt/kvcache/huge_test_4tb.bin"
result_path = "./results"
os.makedirs(result_path, exist_ok=True)
num_gpus = torch.cuda.device_count()
batch_sizes = [2**i for i in range(20, 32)]  # 1 MB to 1 GB in powers of 2
repeats = 100
num_parallel_reads = 1  # Adjust down if VRAM limited  # Can use KVIKIO_NTHREADS=4 instead
mini_batch = 1       # Launch mini_batches of preads at a time 
dtype = np.float16
dtype_size = np.dtype(dtype).itemsize

def log(msg):
    now = time.strftime('%H:%M:%S')
    print(f"[{now}] {msg}", flush=True)

def kvikio_pread_bandwidth_worker(gpu_id, file_path, batch_sizes, repeats, num_parallel_reads, file_size, result_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    batch_results = []
    align = 1024 * 1024  # 1MB alignment

    for batch_size in batch_sizes:
        gbs_results = []
        for rep in range(repeats):
            total_bytes = batch_size * num_parallel_reads
            t0 = time.perf_counter()
            with CuFile(file_path, "rb") as f:
                n_full = num_parallel_reads // mini_batch
                n_rem = num_parallel_reads % mini_batch
                for group in range(n_full + (1 if n_rem else 0)):
                    this_batch = mini_batch if group < n_full else n_rem
                    if this_batch == 0:
                        continue
                    futures = []
                    buffers = []
                    offsets = []
                    for i in range(this_batch):
                        buf = torch.empty(batch_size // dtype_size, dtype=torch.float16, device=device)
                        max_offset = file_size - batch_size
                        offset = np.random.randint(0, max_offset // align) * align
                        #offset = 0  # Force cache reuse for testing ################################################
                        
                        fut = f.pread(buf, batch_size, offset)
                        futures.append((fut, buf))
                    for fut, buf in futures:
                        fut.get()
                        del buf
                    torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            gbps = total_bytes / elapsed / 1e9
            gbs_results.append(gbps)
        arr = np.array(gbs_results)
        mean, minv, maxv, std = arr.mean(), arr.min(), arr.max(), arr.std()
        batch_results.append(dict(
            batch_size=batch_size,
            gds_bw_mean=mean,
            gds_bw_min=minv,
            gds_bw_max=maxv,
            gds_bw_std=std,
            num_parallel_reads=num_parallel_reads
        ))
        log(f"[GPU {gpu_id}] Batch {batch_size//1024} KB: Kvikio GDS pread: {mean:.2f}±{std:.2f} GB/s [{minv:.2f}-{maxv:.2f}]")
    result_q.put((gpu_id, batch_results))

def save_csv_and_plot(gpu_id, batch_results):
    csv_path = os.path.join(result_path, f"gpu_{gpu_id}_kvikio_pread_bandwidth.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "BatchSize_KB", "GDS_BW_Mean_GBs", "GDS_BW_Min_GBs", "GDS_BW_Max_GBs", "GDS_BW_Std_GBs", "NumParallelReads"
        ])
        for stat in batch_results:
            writer.writerow([
                stat["batch_size"] // 1024,
                stat["gds_bw_mean"], stat["gds_bw_min"], stat["gds_bw_max"], stat["gds_bw_std"], stat["num_parallel_reads"]
            ])
    log(f"Saved CSV for GPU {gpu_id} to {csv_path}")

    batch_sizes_kb = [stat["batch_size"] // 1024 for stat in batch_results]
    gds_mean = np.array([stat["gds_bw_mean"] for stat in batch_results])
    gds_std = np.array([stat["gds_bw_std"] for stat in batch_results])

    plt.figure(figsize=(10, 7))
    plt.errorbar(batch_sizes_kb, gds_mean, yerr=gds_std, label=f"GPU {gpu_id} - Kvikio GDS pread", fmt='-o', capsize=5)
    plt.xscale("log", base=2)
    plt.xlabel("Batch Size (KB)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title(f"Kvikio Random Disk→GPU Bandwidth (GPU {gpu_id}, {stat['num_parallel_reads']} parallel preads)")
    plt.legend()
    plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(result_path, f"gpu_{gpu_id}_kvikio_pread_bandwidth.png")
    plt.savefig(plot_path)
    plt.close()
    log(f"Saved plot for GPU {gpu_id} to {plot_path}")

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    file_size = os.path.getsize(test_file)
    log(f"*** NOTE: Using large file {test_file} ({file_size/1e9:.1f} GB). Random, 1MB-aligned offsets. ***")
    log(f"Testing {num_gpus} GPUs in parallel, {num_parallel_reads} outstanding random preads per batch (mini-batch={mini_batch})...")

    result_queue = Queue()
    procs = []
    t0 = time.time()
    for gpu_id in range(num_gpus):
        p = Process(
            target=kvikio_pread_bandwidth_worker,
            args=(gpu_id, test_file, batch_sizes, repeats, num_parallel_reads, file_size, result_queue),
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
        save_csv_and_plot(gpu_id, batch_results)
        for stat in batch_results:
            log(f"[RESULT] GPU {gpu_id} | Batch {stat['batch_size']//1024} KB: "
                f"Kvikio GDS pread: {stat['gds_bw_mean']:.2f}±{stat['gds_bw_std']:.2f} GB/s "
                f"({stat['num_parallel_reads']} outstanding)")

    log(f"Total elapsed wall time: {elapsed:.1f} s")

if __name__ == "__main__":
    main()
