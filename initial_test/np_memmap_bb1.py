import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os

shape_fixed = (32, 1024)
dtype = torch.float16
device = 'cuda'

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
write_times = []
read_times = []
round_trip_times = []

for B in batch_sizes:
    shape = (B, *shape_fixed)
    tensor_gpu = torch.full(shape, 32, dtype=dtype, device=device)
    filename = f'/mnt/kvcache/payload_{B}.dat'
    np_shape = (B, *shape_fixed)
    np_dtype = np.float16

    tensor_cpu = tensor_gpu.cpu().pin_memory()
    arr_np = tensor_cpu.numpy()
    start_write = time.perf_counter()
    fp = np.memmap(filename, dtype=np_dtype, mode='w+', shape=np_shape)
    fp[:] = arr_np[:]
    end_write = time.perf_counter()
    del fp

    write_time = end_write - start_write
    write_times.append(write_time)

    start_read = time.perf_counter()
    fp = np.memmap(filename, dtype=np_dtype, mode='r', shape=np_shape)
    arr_read = np.array(fp, copy=False)
    tensor_back = torch.from_numpy(arr_read).pin_memory().to(device, non_blocking=True)
    end_read = time.perf_counter()


    read_time = end_read - start_read
    read_times.append(read_time)
    round_trip_times.append(write_time + read_time)

    os.remove(filename)


element_size = np.dtype(np.float16).itemsize
bandwidth_write = []
bandwidth_read = []

for i, B in enumerate(batch_sizes):
    num_elements = B * 32 * 1024
    total_bytes = num_elements * element_size
    total_gb = total_bytes / (1024 ** 3)

    bw_w = total_gb / write_times[i]
    bw_r = total_gb / read_times[i]

    bandwidth_write.append(bw_w)
    bandwidth_read.append(bw_r)


plt.figure(figsize=(10, 6))
# plt.plot(batch_sizes, write_times, label='GPU → CPU + Write', marker='o')
# plt.plot(batch_sizes, read_times, label='Load + CPU → GPU', marker='o')
# plt.plot(batch_sizes, round_trip_times, label='Total Round Trip', marker='o')
# plt.xlabel('Batch Size (B)')
# plt.ylabel('Time (seconds)')
# plt.title('GPU↔CPU Transfer and Write/Read Benchmark')
plt.plot(batch_sizes, bandwidth_write, label='GPU → CPU + Write', marker='o')
plt.plot(batch_sizes, bandwidth_read, label='Load + CPU → GPU', marker='o')
plt.xlabel('Batch Size (B)')
plt.ylabel('Bandwidth (GB/s)')
plt.title('I/O and Transfer Bandwidth vs Batch Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"ssd_speeds_memmap_bb1.png")