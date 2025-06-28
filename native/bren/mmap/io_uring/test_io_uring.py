import ctypes
import os
import time
import torch
import time

# Load the shared library
lib = ctypes.CDLL("./libread_io_uring.so")
lib.read_to_gpu_io_uring.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
lib.read_to_gpu_io_uring.restype = None

# Settings
filepath = b"/mnt/kvcache/huge_test_4tb.bin"
total_size = 1 * 1024 * 1024 * 1024  # 5 GB
device = "cuda:1"

# Allocate target tensor
gpu_tensor = torch.empty(total_size, dtype=torch.uint8, device='cuda:1')

# Run and time
start = time.perf_counter()
lib.read_to_gpu_io_uring(filepath, ctypes.c_void_p(gpu_tensor.data_ptr()), total_size)
torch.cuda.synchronize()
end = time.perf_counter()

elapsed = end - start
bw = total_size / elapsed / (1024 ** 3)

print(f"Elapsed time: {elapsed:.4f} s")
print(f"Transfer bandwidth: {bw:.2f} GB/s")
