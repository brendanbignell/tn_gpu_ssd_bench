import multiprocessing
import torch
import time
import ctypes

def run_reader(filepath, size, device):
    lib = ctypes.CDLL("./libread_io_uring.so")
    lib.read_to_gpu_io_uring.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.read_to_gpu_io_uring.restype = None

    tensor = torch.empty(size, dtype=torch.uint8, device=device)
    start = time.perf_counter()
    lib.read_to_gpu_io_uring(filepath.encode(), ctypes.c_void_p(tensor.data_ptr()), size)
    #torch.cuda.synchronize()  # Don't think I need this as stream sync in C code
    end = time.perf_counter()
    print(f"{filepath}: {size / (end - start) / (1024 ** 3):.2f} GB/s")

if __name__ == "__main__":
    size = 1 * 1024 * 1024 * 1024   # 1 GB per process
    files = [
        "/mnt/kvcache/huge_test_4tb.bin",
        "/mnt/kvcache/huge_test_4tb.bin",
        "/mnt/kvcache/huge_test_4tb.bin",
        "/mnt/kvcache/huge_test_4tb.bin",
        "/mnt/kvcache/huge_test_4tb.bin",

    ]
    ctx = multiprocessing.get_context("spawn")
    start = time.perf_counter()
    procs = [ctx.Process(target=run_reader, args=(f, size, "cuda:1")) for f in files]
    for p in procs: p.start()
    for p in procs: p.join()
    end = time.perf_counter()
    print(f"Total time: {end - start:.4f} s")
    print(f"Total bandwidth: {len(files) * size / (end - start) / (1024 ** 3):.2f} GB/s")
