import os
import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, set_start_method

from kvikio import CuFile

mnt_path = "/mnt/kvcache"
os.makedirs(mnt_path, exist_ok=True)
file_path = os.path.join(mnt_path, "tensor_1gb.bin")

np_dtype = np.float16
cp_dtype = cp.float16
dtype_size = np.dtype(np_dtype).itemsize

buffer_bytes = 1 * 1024 ** 3  # 1 GB
n_elements = buffer_bytes // dtype_size
num_runs = 10

def prepare_tensor_file():
    if not os.path.exists(file_path):
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

    cp.cuda.Device(gpu_id).use()
    gpu_array = cp.empty((n_elements,), dtype=cp_dtype)
    bandwidths = []
    for run in range(num_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.time()
        with CuFile(file_path, "rb") as f:
            f.read(gpu_array, 0, buffer_bytes)
        cp.cuda.Stream.null.synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        bw = buffer_bytes / elapsed / 1e9
        bandwidths.append(bw)
    q.put((gpu_id, bandwidths))

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    prepare_tensor_file()

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
    plt.figure(figsize=(8,4))
    for gpu_id in sorted(results):
        bw = np.array(results[gpu_id])
        print(f"\nGPU {gpu_id}:")
        for i, x in enumerate(bw):
            print(f"  Run {i+1:2d}: {x:.2f} GB/s")
        print(f"  Mean: {bw.mean():.2f} GB/s  Min: {bw.min():.2f}  Max: {bw.max():.2f}  Std: {bw.std():.2f}")
        plt.plot(np.arange(1, num_runs+1), bw, marker='o', label=f"GPU {gpu_id}")

    plt.title("Kvikio SSD → VRAM 1GB Bandwidth per Run (All GPUs)")
    plt.xlabel("Run")
    plt.ylabel("Bandwidth (GB/s)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("kvikio_multi_gpu_1gb_bandwidth.png")
    plt.show()

if __name__ == "__main__":
    main()
