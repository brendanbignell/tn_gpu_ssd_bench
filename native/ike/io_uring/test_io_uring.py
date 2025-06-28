import ctypes
import torch
import os
import time

# Load the shared library
lib = ctypes.CDLL('./libread_io_uring.so')
lib.read_to_gpu_io_uring.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
lib.read_to_gpu_io_uring.restype = None

# File size to read: 1 GB
size = 5 * 1024 * 1024 * 1024

# Allocate a GPU tensor
gpu_tensor = torch.empty(size, dtype=torch.uint8, device='cuda:1')

# Absolute path to the file on SSD
filepath = os.path.abspath("/mnt/kvcache/huge_test_4tb.bin").encode("utf-8")
# Alternatively, specify directly:
# filepath = b"/mnt/ssd/largefile.bin"

# Call the C function and measure elapsed time
start = time.perf_counter()
lib.read_to_gpu_io_uring(filepath, ctypes.c_void_p(gpu_tensor.data_ptr()), size)
torch.cuda.synchronize()
end = time.perf_counter()

elapsed = end - start
bw = size / elapsed / (1024 ** 3)

# Print results
#print("Sum of tensor:", gpu_tensor.sum().item())
print(f"Elapsed time: {elapsed:.4f}s")
print(f"Bandwidth: {bw:.2f} GB/s")

