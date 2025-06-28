import time
import torch
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt

num_workers = 1
batch_sizes = list(range(1, 512, num_workers))
write_times = []
filename = "/mnt/kvcache/tensor_fp16.dat"
for B in batch_sizes:
    tensor = torch.full((B, 32, 1024), 0, dtype=torch.float16, device="cpu").pin_memory()
    np.memmap(filename, dtype=np.float16, mode='w+', shape=(B, 32, 1024)).flush()
    
    def write_chunk(start_idx, chunk):
        mmap = np.memmap(filename, dtype=np.float16, mode="r+", shape=(B, 32, 1024))
        mmap[start_idx:start_idx + chunk.shape[0]] = chunk.numpy()
        mmap.flush()
    
    chunk_size = B // num_workers
    start_t = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_workers - 1 else B
            new_tensor = tensor[start:end]
            # print(f"Chunk shape - {new_tensor.shape}")
            futures.append(executor.submit(write_chunk, start, tensor[start:end]))
        concurrent.futures.wait(futures)
    end_t = time.time()

    print(f"[B-{B}] Time - {(end_t - start_t) * 1e3}ms")
    
    write_times.append((end_t - start_t))

element_size = np.dtype(np.float16).itemsize  # 2 bytes
bandwidth_write = []
bandwidth_read = []

for i, B in enumerate(batch_sizes):
    num_elements = B * 32 * 1024
    total_bytes = num_elements * element_size
    total_gb = total_bytes / (1024 ** 3)

    bw_w = total_gb / write_times[i]
    # bw_r = total_gb / read_times[i]

    bandwidth_write.append(bw_w)
    # bandwidth_read.append(bw_r)

# print(f"Bw - {bandwidth_write}")
# --- Plotting ---
plt.figure(figsize=(10, 6))
# plt.plot(batch_sizes, write_times, label='GPU → CPU + Write', marker='o')
# plt.plot(batch_sizes, read_times, label='Load + CPU → GPU', marker='o')
# plt.plot(batch_sizes, round_trip_times, label='Total Round Trip', marker='o')
# plt.xlabel('Batch Size (B)')
# plt.ylabel('Time (seconds)')
# plt.title('GPU↔CPU Transfer and Write/Read Benchmark')
plt.plot(batch_sizes, bandwidth_write, label='GPU → CPU + Write', marker='o')
# plt.plot(batch_sizes, bandwidth_read, label='Load + CPU → GPU', marker='o')
plt.xlabel('Batch Size (B)')
plt.ylabel('Bandwidth (GB/s)')
plt.title('I/O and Transfer Bandwidth vs Batch Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"ssd_speeds.png")