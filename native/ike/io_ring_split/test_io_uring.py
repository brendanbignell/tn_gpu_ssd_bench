import ctypes
import torch
import os
import time
import matplotlib.pyplot as plt

lib = ctypes.CDLL('./libread_io_uring.so')

lib.io_uring_cuda_init.argtypes = [ctypes.c_char_p]
lib.io_uring_cuda_init.restype = ctypes.c_int

lib.io_uring_cuda_run.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
lib.io_uring_cuda_run.restype = ctypes.c_int

lib.io_uring_cuda_close.argtypes = []
lib.io_uring_cuda_close.restype = None

filepath = os.path.abspath("/mnt/kvcache/huge_test_4tb.bin").encode("utf-8")

# Initialize once
start_init = time.perf_counter()
ret = lib.io_uring_cuda_init(filepath)
end_init = time.perf_counter()
if ret != 0:
    raise RuntimeError(f"init failed, code {ret}")
print(f"Init time: {end_init - start_init:.4f}s\n")

size_mb = 1
max_size_mb = 1024  # 4GB

sizes = []
throughputs = []

while size_mb <= max_size_mb:
    total_size = size_mb * 1024 * 1024
    #print(f"Testing with size: {size_mb} MB")

    gpu_tensor = torch.empty(total_size, dtype=torch.uint8, device='cuda:1')

    start = time.perf_counter()
    ret = lib.io_uring_cuda_run(ctypes.c_void_p(gpu_tensor.data_ptr()), total_size)
    if ret != 0:
        raise RuntimeError(f"run failed at size {size_mb}MB, code {ret}")
    end = time.perf_counter()

    elapsed = end - start
    throughput = (total_size / (1024 * 1024)) / elapsed  # MB/s

    #print(f"Elapsed time: {elapsed:.4f}s")
    #print(f"Throughput: {throughput:.2f} MB/s\n")
    print(f"Testing size: {size_mb} MB | Elapsed time: {elapsed:.4f}s | Throughput: {throughput:.2f} MB/s")

    sizes.append(size_mb)
    throughputs.append(throughput)

    size_mb *= 2

# Close resources
start_close = time.perf_counter()
lib.io_uring_cuda_close()
end_close = time.perf_counter()
print(f"Close time: {end_close - start_close:.4f}s\n")

# Plot throughput vs file size
plt.figure(figsize=(8, 5))
plt.plot(sizes, throughputs, marker='o')
plt.xlabel('File Size (MB)')
plt.ylabel('Throughput (MB/s)')
plt.title('SSD to GPU Transfer Throughput')
plt.grid(True)
plt.savefig('throughput_vs_file_size.png')
plt.show()