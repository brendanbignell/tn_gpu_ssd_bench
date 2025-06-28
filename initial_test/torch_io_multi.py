import os
import time
import numpy as np
import torch
import threading
import queue
from multiprocessing import Process, Queue, set_start_method
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

mnt_path = "/mnt/kvcache"
result_path = "./results"
os.makedirs(result_path, exist_ok=True)
total_transfer_bytes = 1 * 1024 ** 3  # 1 GB per batch size
dtype = np.float16
dtype_size = np.dtype(dtype).itemsize
num_gpus = torch.cuda.device_count()
batch_sizes = [2 ** i for i in range(16, 31)]  # 64KB to 1GB
repeats = 10  # Number of runs per batch size

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
                arr = np.random.randn(this_chunk // dtype_size).astype(dtype)
                f.write(arr.tobytes())
                written += this_chunk
                if pbar is not None:
                    pbar.update(this_chunk)
        log(f"Finished creating {path}")

def pipelined_gpu_benchmark(gpu_id, file_path, batch_sizes, total_bytes, repeats, result_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    batch_results = []

    for bidx, batch_size in enumerate(batch_sizes):
        n_batches = total_bytes // batch_size

        # Arrays to collect bandwidth for each repeat
        ssd_ram_bandwidths = []
        ram_vram_bandwidths = []
        pipeline_bandwidths = []
        pipeline_times = []

        for run in range(repeats):
            q_depth = 2
            arr_queue = queue.Queue(maxsize=q_depth)
            stop_token = object()
            # Only show the progress bar on first run to avoid clutter
            show_pbar = (run == 0)
            pbar = tqdm(total=n_batches, position=gpu_id, desc=f"GPU {gpu_id} {batch_size//1024} KB Run {run+1}", leave=show_pbar, ncols=80, disable=not show_pbar)

            ssd_ram_times = []
            ram_vram_times = []
            pipeline_start = None
            pipeline_end = None

            def reader():
                nonlocal pipeline_start
                with open(file_path, "rb") as f:
                    for _ in range(n_batches):
                        ssd_start = time.time()
                        raw = f.read(batch_size)
                        ssd_end = time.time()
                        arr = np.frombuffer(raw, dtype=dtype)
                        arr_queue.put((arr, ssd_end-ssd_start))
                    arr_queue.put((stop_token, 0))

            t = threading.Thread(target=reader)
            t.start()
            torch.cuda.synchronize()
            pipeline_start = time.time()
            for _ in range(n_batches):
                arr, ssd_ram_time = arr_queue.get()
                if arr is stop_token:
                    break
                ssd_ram_times.append(ssd_ram_time)
                # RAM->VRAM
                ram_start = time.time()
                host_tensor = torch.from_numpy(arr).pin_memory()
                gpu_tensor = host_tensor.to(device, non_blocking=True)
                torch.cuda.synchronize()
                ram_end = time.time()
                ram_vram_times.append(ram_end - ram_start)
                del gpu_tensor
                torch.cuda.empty_cache()
                if show_pbar:
                    pbar.update(1)
            t.join()
            pipeline_end = time.time()
            if show_pbar:
                pbar.close()

            ssd_ram_times = np.array(ssd_ram_times)
            ram_vram_times = np.array(ram_vram_times)
            pipeline_time = pipeline_end - pipeline_start

            # Per-run mean bandwidth for this batch size
            ssd_ram_bandwidths.append(batch_size / ssd_ram_times.mean() / 1e9)
            ram_vram_bandwidths.append(batch_size / ram_vram_times.mean() / 1e9)
            pipeline_bandwidths.append(total_bytes / pipeline_time / 1e9)
            pipeline_times.append(pipeline_time)

        # Save stats for all repeats
        def stats(arr):
            arr = np.array(arr)
            return arr.mean(), arr.min(), arr.max(), arr.std()

        ssd_mean, ssd_min, ssd_max, ssd_std = stats(ssd_ram_bandwidths)
        ram_mean, ram_min, ram_max, ram_std = stats(ram_vram_bandwidths)
        pipe_mean, pipe_min, pipe_max, pipe_std = stats(pipeline_bandwidths)
        time_mean, time_min, time_max, time_std = stats(pipeline_times)

        batch_stats = dict(
            batch_size=batch_size,
            n_batches=n_batches,
            # SSD→RAM
            ssd_ram_mean=ssd_mean, ssd_ram_min=ssd_min, ssd_ram_max=ssd_max, ssd_ram_std=ssd_std,
            # RAM→VRAM
            ram_vram_mean=ram_mean, ram_vram_min=ram_min, ram_vram_max=ram_max, ram_vram_std=ram_std,
            # Pipeline
            pipeline_bw_mean=pipe_mean, pipeline_bw_min=pipe_min, pipeline_bw_max=pipe_max, pipeline_bw_std=pipe_std,
            pipeline_time_mean=time_mean, pipeline_time_min=time_min, pipeline_time_max=time_max, pipeline_time_std=time_std
        )
        batch_results.append(batch_stats)

        log(
            f"[GPU {gpu_id}] Batch {batch_size//1024} KB: "
            f"SSD→RAM: {ssd_mean:.2f}±{ssd_std:.2f} GB/s | "
            f"RAM→VRAM: {ram_mean:.2f}±{ram_std:.2f} GB/s | "
            f"Pipeline: {pipe_mean:.2f}±{pipe_std:.2f} GB/s"
        )
    result_q.put((gpu_id, batch_results))

def save_csv_and_plot(gpu_id, batch_results):
    csv_path = os.path.join(result_path, f"gpu_{gpu_id}_bandwidth.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "BatchSize_KB",
            "SSD_RAM_Mean_GBs", "SSD_RAM_Min_GBs", "SSD_RAM_Max_GBs", "SSD_RAM_Std_GBs",
            "RAM_VRAM_Mean_GBs", "RAM_VRAM_Min_GBs", "RAM_VRAM_Max_GBs", "RAM_VRAM_Std_GBs",
            "Pipeline_BW_Mean_GBs", "Pipeline_BW_Min_GBs", "Pipeline_BW_Max_GBs", "Pipeline_BW_Std_GBs",
            "Pipeline_Time_Mean_s", "Pipeline_Time_Min_s", "Pipeline_Time_Max_s", "Pipeline_Time_Std_s",
            "NumBatches"
        ])
        for stat in batch_results:
            writer.writerow([
                stat["batch_size"] // 1024,
                stat["ssd_ram_mean"], stat["ssd_ram_min"], stat["ssd_ram_max"], stat["ssd_ram_std"],
                stat["ram_vram_mean"], stat["ram_vram_min"], stat["ram_vram_max"], stat["ram_vram_std"],
                stat["pipeline_bw_mean"], stat["pipeline_bw_min"], stat["pipeline_bw_max"], stat["pipeline_bw_std"],
                stat["pipeline_time_mean"], stat["pipeline_time_min"], stat["pipeline_time_max"], stat["pipeline_time_std"],
                stat["n_batches"]
            ])
    log(f"Saved CSV for GPU {gpu_id} to {csv_path}")

    # Plot
    batch_sizes_kb = [stat["batch_size"] // 1024 for stat in batch_results]
    ssd_mean = np.array([stat["ssd_ram_mean"] for stat in batch_results])
    ssd_std = np.array([stat["ssd_ram_std"] for stat in batch_results])
    ram_mean = np.array([stat["ram_vram_mean"] for stat in batch_results])
    ram_std = np.array([stat["ram_vram_std"] for stat in batch_results])
    pipe_mean = np.array([stat["pipeline_bw_mean"] for stat in batch_results])
    pipe_std = np.array([stat["pipeline_bw_std"] for stat in batch_results])

    plt.figure(figsize=(10, 7))
    plt.errorbar(batch_sizes_kb, ssd_mean, yerr=ssd_std, label="SSD→RAM", fmt='-o', capsize=5)
    plt.errorbar(batch_sizes_kb, ram_mean, yerr=ram_std, label="RAM→VRAM", fmt='-s', capsize=5)
    plt.errorbar(batch_sizes_kb, pipe_mean, yerr=pipe_std, label="Pipeline (Overall)", fmt='-^', capsize=5)
    plt.xscale("log", base=2)
    plt.xlabel("Batch Size (KB)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title(f"Disk→GPU Bandwidth Breakdown (GPU {gpu_id})")
    plt.legend()
    plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(result_path, f"gpu_{gpu_id}_bandwidth.png")
    plt.savefig(plot_path)
    plt.close()
    log(f"Saved plot for GPU {gpu_id} to {plot_path}")

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Already set

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

    # --- Run benchmarks ---
    log(f"Starting benchmark: {num_gpus} GPUs, batch sizes {batch_sizes[0]//1024} KB to {batch_sizes[-1]//1024**2} GB, {total_transfer_bytes//1024**2} MB total per size, {repeats} runs per batch size")
    result_queue = Queue()
    procs = []
    t0 = time.time()
    for gpu_id in range(num_gpus):
        p = Process(
            target=pipelined_gpu_benchmark,
            args=(gpu_id, filespecs[gpu_id][0], batch_sizes, total_transfer_bytes, repeats, result_queue),
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
            log(
                f"[RESULT] GPU {gpu_id} | Batch {stat['batch_size']//1024} KB: "
                f"SSD→RAM: {stat['ssd_ram_mean']:.2f}±{stat['ssd_ram_std']:.2f} GB/s, "
                f"RAM→VRAM: {stat['ram_vram_mean']:.2f}±{stat['ram_vram_std']:.2f} GB/s, "
                f"Pipeline: {stat['pipeline_bw_mean']:.2f}±{stat['pipeline_bw_std']:.2f} GB/s, "
                f"{stat['n_batches']} batches, {stat['pipeline_time_mean']:.2f} s"
            )

    log(f"Total elapsed wall time: {elapsed:.1f} s")

if __name__ == "__main__":
    main()
