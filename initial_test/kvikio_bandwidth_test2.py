import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, set_start_method
import time
import csv

from kvikio import CuFile

#mnt_path = "/mnt/kvcache"
mnt_path = "/tmp"
os.makedirs(mnt_path, exist_ok=True)
file_path = os.path.join(mnt_path, "tensor_1gb.bin")

np_dtype = np.float16
cp_dtype = cp.float16
dtype_size = np.dtype(np_dtype).itemsize

batch_sizes = [2**i for i in range(16, 31)][::-1]  # 1GB to 64KB
num_runs = 100

def prepare_tensor_file():
    # Create a 1GB file if it doesn't exist
    buffer_bytes = batch_sizes[0]
    if not os.path.exists(file_path) or os.path.getsize(file_path) < buffer_bytes:
        print("Creating test file...")
        chunk_size = 8 * 1024 * 1024  # 8 MB
        with open(file_path, "wb") as f:
            written = 0
            while written < buffer_bytes:
                remain = buffer_bytes - written
                this_chunk = min(chunk_size, remain)
                arr = np.random.randn(this_chunk // dtype_size).astype(np_dtype)
                f.write(arr.tobytes())
                written += this_chunk
        print("Done.")

def kvikio_bandwidth_worker(gpu_id, q):
    import cupy as cp
    from kvikio import CuFile
    import time

    np_dtype = np.float16
    cp_dtype = cp.float16
    dtype_size = np.dtype(np_dtype).itemsize

    cp.cuda.Device(gpu_id).use()
    result = {}

    for batch_size in batch_sizes:
        n_elements = batch_size // dtype_size
        bandwidths = []
        for run in range(num_runs):
            cp.cuda.Stream.null.synchronize()
            gpu_array = cp.empty((n_elements,), dtype=cp_dtype)
            t0 = time.perf_counter()
            with CuFile(file_path, "rb") as f:
                f.read(gpu_array, 0, batch_size)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            bw = batch_size / elapsed / 1e9
            bandwidths.append(bw)
        result[batch_size] = bandwidths
    q.put((gpu_id, result))

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    prepare_tensor_file()

    print("\n*** NOTE: Disk reads may be OS-cached unless run on a large/unmounted dataset, or OS caches are manually dropped before each run. For strict benchmarking, run `sudo sh -c \"echo 3 > /proc/sys/vm/drop_caches\"` before each test batch. ***\n")

    num_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"Testing {num_gpus} GPUs in parallel...")

    q = Queue()
    procs = []
    for gpu_id in range(num_gpus):
        p = Process(target=kvikio_bandwidth_worker, args=(gpu_id, q))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    results = {}
    while not q.empty():
        gpu_id, bandwidths = q.get()
        results[gpu_id] = bandwidths

    # Print stats and plot
    plt.figure(figsize=(12,6))
    csv_rows = []
    for gpu_id in sorted(results):
        means = []
        stds = []
        sizes_kb = []
        for batch_size in batch_sizes:
            bw = np.array(results[gpu_id][batch_size])
            sizes_kb.append(batch_size // 1024)
            means.append(bw.mean())
            stds.append(bw.std())
            # For CSV
            for i, b in enumerate(bw):
                csv_rows.append([gpu_id, batch_size // 1024, i, b])
            print(f"\nGPU {gpu_id} batch {batch_size//1024}KB: mean {bw.mean():.2f} GB/s, std {bw.std():.2f}, min {bw.min():.2f}, max {bw.max():.2f}")
        plt.errorbar(sizes_kb, means, yerr=stds, marker='o', label=f"GPU {gpu_id}", capsize=5)

    plt.xscale("log", base=2)
    plt.xlabel("Batch Size (KB)")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("Kvikio SSD → VRAM Bandwidth vs Batch Size (All GPUs)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_file = "kvikio_multi_gpu_bandwidth_vs_batch.png"
    plt.savefig(plot_file)
    print(f"\nPlot saved to {plot_file}")

    # Save CSV
    csv_file = "kvikio_multi_gpu_bandwidth_vs_batch.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_id", "batch_size_kb", "run_idx", "bandwidth_gbs"])
        writer.writerows(csv_rows)
    print(f"Raw results saved to {csv_file}")

if __name__ == "__main__":
    main()
