import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Configuration
mnt_path = "/mnt/kvcache"
os.makedirs(mnt_path, exist_ok=True)
tensor_sizes = [2 ** i for i in range(10, 24)] 
dtype = np.float16
num_gpus = torch.cuda.device_count()

# Utility to generate file if it doesn't exist
def prepare_tensor_file(size_bytes):
    count = size_bytes // np.dtype(dtype).itemsize
    data = np.random.randn(count).astype(dtype)
    path = os.path.join(mnt_path, f"tensor_{size_bytes}.bin")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data.tobytes())
    return path

# Function to test bandwidth for a given GPU and tensor size
def transfer_test(args):
    size_bytes, gpu_id = args
    path = prepare_tensor_file(size_bytes)

    torch.cuda.set_device(gpu_id)
    start = time.time()

    with open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=dtype)
    tensor = torch.from_numpy(arr).to(f"cuda:{gpu_id}")

    end = time.time()
    duration = end - start
    bandwidth_gbps = size_bytes / duration / 1e9
    return (gpu_id, size_bytes, bandwidth_gbps)

# Run benchmarks
tasks = [(size, gpu_id) for size in tensor_sizes for gpu_id in range(num_gpus)]
with Pool(min(len(tasks), cpu_count())) as pool:
    results = pool.map(transfer_test, tasks)

# Organize results
bw_data = {f"cuda:{i}": [] for i in range(num_gpus)}
for gpu_id, size, bw in results:
    bw_data[f"cuda:{gpu_id}"].append((size / 1024, bw))  # Convert size to KB

# Plotting
plt.figure(figsize=(12, 6))
for gpu, data in bw_data.items():
    sizes_kb, bws = zip(*sorted(data))
    plt.plot(sizes_kb, bws, label=gpu)

plt.xscale("log", base=2)
plt.xlabel("Transfer Size (KB)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("NVME → GPU Transfer Bandwidth")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()
plt.savefig(f"torch_io_benchmark_2.png")