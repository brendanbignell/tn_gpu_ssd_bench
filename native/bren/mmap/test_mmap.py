import torch
import os
import mmap
import time
import random
import concurrent.futures

# Total size to read: 5 GB
size = 2 * 1024 * 1024 * 1024
block_size = 1 * 1024 * 1024  # 1 MB
num_workers = 64
per_worker_size = size // num_workers

# Set unique random seed
random.seed(os.getpid() + int(time.time()))

# Absolute path to the file on SSD
filepath = "/mnt/kvcache/huge_test_4tb.bin"
file_size = os.path.getsize(filepath)

# Compute random 1MB-aligned offset
max_offset = file_size - size
offset = random.randrange(0, max_offset // block_size) * block_size

# Allocate the final GPU tensor
gpu_tensor = torch.empty(size, dtype=torch.uint8, device='cuda:1')

# Worker function to read and transfer a slice
def read_and_transfer(worker_id):
    worker_offset = offset + worker_id * per_worker_size
    with open(filepath, "rb") as f:
        mmapped = mmap.mmap(f.fileno(), length=per_worker_size, offset=worker_offset, access=mmap.ACCESS_READ)
        mmapped.madvise(mmap.MADV_RANDOM)

        tensor = torch.frombuffer(mmapped, dtype=torch.uint8, count=per_worker_size)

        stream = torch.cuda.Stream(device='cuda:1')
        with torch.cuda.stream(stream):
            gpu_tensor[worker_id * per_worker_size:(worker_id + 1) * per_worker_size].copy_(tensor, non_blocking=True)
        stream.synchronize()
        mmapped.close()

# Start timing
start = time.perf_counter()

# Launch parallel workers
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(read_and_transfer, range(num_workers))

torch.cuda.synchronize()
end = time.perf_counter()

elapsed_time = end - start
bandwidth_gb_s = (size / elapsed_time) / (1024**3)

print(f"Elapsed time: {elapsed_time:.4f}s")
print(f"Transfer bandwidth: {bandwidth_gb_s:.2f} GB/s")