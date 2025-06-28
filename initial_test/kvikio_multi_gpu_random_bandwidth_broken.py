import os
import time
import random
import csv
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, set_start_method

from kvikio import CuFile

mnt_path = "/mnt/kvcache"
os.makedirs(mnt_path, exist_ok=True)
file_path = os.path.join(mnt_path, "huge_test_1tb.bin")

np_dtype = np.float16
cp_dtype = cp.float16
dtype_size = np.dtype(np_dtype).itemsize

#total_test_bytes = 4 * 1024 ** 4   # 4TB
total_test_bytes = 10 * 1024 ** 3   # 10GB

batch_sizes = [2**i for i in range(16, 17)][::-1]  # 1GB down to 64KB
num_runs = 10

##############################
# PARALLEL FILE CREATION UTILS
##############################

def make_chunk_for_write(args):
    chunk_idx, chunk_bytes, total_bytes, filepath = args
    seed = (chunk_idx * 179) % 2**16  # Now always valid uint16
    arr = (np.arange(chunk_bytes // 2, dtype=np.uint16) ^ seed).astype(np.uint16)
    data = arr.tobytes()
    chunk_offset = chunk_idx * chunk_bytes
    with open(filepath, "r+b", buffering=0) as f:
        f.seek(chunk_offset)
        f.write(data)
    return chunk_idx

def fast_pattern_file_write(filepath, total_bytes, chunk_bytes=1024*1024*1024, n_workers=32):
    """
    Fill a file with 'pseudo-random' but fast-to-generate data in parallel chunks.
    Each chunk is an integer sequence XOR'd with the chunk index for minimal compressibility/caching.
    """
    import concurrent.futures

    num_chunks = (total_bytes + chunk_bytes - 1) // chunk_bytes
    print(f"Creating file {filepath} ({total_bytes//1024**3} GB) in {num_chunks} chunks ({chunk_bytes//1024**2} MB each, {n_workers} workers)...")

    # Preallocate file (for parallel r+b access)
    with open(filepath, "wb") as f:
        f.truncate(total_bytes)

    chunk_args = [
        (i, min(chunk_bytes, total_bytes - i*chunk_bytes), total_bytes, filepath)
        for i in range(num_chunks)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        for i, _ in enumerate(pool.map(make_chunk_for_write, chunk_args), 1):
            if i % max(1, num_chunks // 20) == 0:
                print(f"  ...{min(i*chunk_bytes, total_bytes)//1024**3} GB written")

    print("Fast pattern file creation complete.")

###########################
# MAIN BENCHMARK FUNCTIONS
###########################

def prepare_huge_file():
    if not os.path.exists(file_path) or os.path.getsize(file_path) < total_test_bytes:
        fast_pattern_file_write(
            file_path, 
            total_test_bytes, 
            chunk_bytes=1024*1024*1024,   # 1GB
            n_workers=max(4, os.cpu_count()//2)
        )

def kvikio_bandwidth_worker(gpu_id, q):
    import cupy as cp
    from kvikio import CuFile
    import time
    import random

    np_dtype = np.float16
    cp_dtype = cp.float16
    dtype_size = np.dtype(np_dtype).itemsize

    cp.cuda.Device(gpu_id).use()
    results = []

    max_offset = total_test_bytes

    for batch_size in batch_sizes:
        run_bandwidths = []
        for run in range(num_runs):
            # Only allow offsets where offset + batch_size <= max_offset
            assert batch_size % dtype_size == 0, f"batch_size {batch_size} not a multiple of dtype size {dtype_size}"
            n_elements = batch_size // dtype_size
            max_valid_offset = max_offset - batch_size
            # The "+1" ensures we allow the last valid offset
            #offset = random.randrange(0, (max_valid_offset // batch_size) + 1) * batch_size
            offset = run * batch_size
            assert offset + batch_size <= max_offset, f"Read would go beyond file: offset {offset}, batch_size {batch_size}, file {max_offset}"
            cp.cuda.Stream.null.synchronize()
            gpu_array = cp.empty(n_elements, dtype=cp_dtype)
            assert gpu_array.nbytes == batch_size, \
                f"Allocated buffer {gpu_array.nbytes} does not match batch_size {batch_size} (batch_size={batch_size}, dtype={cp_dtype})"
            t0 = time.perf_counter()
            try:
                with CuFile(file_path, "rb") as f:
                    f.read(gpu_array, offset, batch_size)
            except Exception as e:
                print(f"ERROR: batch_size={batch_size}, nbytes={gpu_array.nbytes}, offset={offset}, run={run}, gpu={gpu_id}")
                raise
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            bw = batch_size / elapsed / 1e9
            run_bandwidths.append(bw)
            # For raw CSV: gpu, batch_size_kb, run, offset_MB, bw_GBs
            results.append((gpu_id, batch_size//1024, run, offset//1024//1024, bw))
    q.put(results)

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    prepare_huge_file()

    print("\n*** NOTE: This benchmark uses a huge test file with random offsets to minimize caching. For stricter results, drop OS caches before running. ***\n")

    #num_gpus = cp.cuda.runtime.getDeviceCount()
    num_gpus = 1
    print(f"Testing {num_gpus} GPUs in parallel...")

    q = Queue()
    procs = []
    for gpu_id in range(num_gpus):
        p = Process(target=kvikio_bandwidth_worker, args=(gpu_id, q))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Collect all results and organize
    all_results = []
    while not q.empty():
        all_results.extend(q.get())

    # Save all raw results to CSV
    csv_file = "kvikio_multi_gpu_random_bandwidth.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_id", "batch_size_kb", "run_idx", "offset_mb", "bandwidth_gbs"])
        for row in all_results:
            writer.writerow(row)
    print(f"Raw results saved to {csv_file}")

    # Compute stats for plotting
    # Format: stats[(gpu_id, batch_size_kb)] = [all bandwidths]
    stats = {}
    for gpu_id, batch_size_kb, run_idx, offset_mb, bw in all_results:
        stats.setdefault((gpu_id, batch_size_kb), []).append(bw)

    plt.figure(figsize=(13,6))
    for gpu_id in range(num_gpus):
        means, stds, sizes_kb = [], [], []
        for batch_size in batch_sizes:
            batch_size_kb = batch_size // 1024
            arr = np.array(stats.get((gpu_id, batch_size_kb), []))
            if len(arr) == 0: continue
            means.append(arr.mean())
            stds.append(arr.std())
            sizes_kb.append(batch_size_kb)
            print(f"GPU {gpu_id} {batch_size_kb}KB: mean={arr.mean():.2f} GB/s, std={arr.std():.2f}, min={arr.min():.2f}, max={arr.max():.2f}")
        plt.errorbar(sizes_kb, means, yerr=stds, marker='o', label=f"GPU {gpu_id}", capsize=4)

    plt.xscale("log", base=2)
    plt.xlabel("Batch Size (KB)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("Kvikio SSD → VRAM Bandwidth vs Batch Size (Random Offsets, All GPUs)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_file = "kvikio_multi_gpu_random_bandwidth.png"
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

if __name__ == "__main__":
    main()
