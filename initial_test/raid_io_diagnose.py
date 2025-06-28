# RAID0 NVMe array reports 70 GB/s peak, but current results show:
# - Sequential Read: ~0.4 - 1.3 GB/s even with 4 threads
# - Python test at 100% CPU but low disk utilization

# Recommendations to diagnose SSD bottleneck:

import os
import psutil
import threading
import time
import mmap
import numpy as np
from concurrent.futures import ThreadPoolExecutor

mnt_path = "/mnt/kvcache"
file_sizes = [2**i for i in range(24, 31)]  # 16MB to 1GB for faster test
threads = 4
results = []

def generate_file(path, size):
    with open(path, "wb") as f:
        f.write(os.urandom(size))

def read_file(path):
    with open(path, "rb") as f:
        return f.read()

def read_mmap(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mm[:]
        mm.close()

def measure_io(method, path):
    start = time.time()
    if method == "read":
        read_file(path)
    elif method == "mmap":
        read_mmap(path)
    return time.time() - start

# Generate test files
for size in file_sizes:
    p = os.path.join(mnt_path, f"test_{size}.bin")
    if not os.path.exists(p):
        generate_file(p, size)

# Benchmark
for method in ["read", "mmap"]:
    for size in file_sizes:
        paths = [os.path.join(mnt_path, f"test_{size}.bin") for _ in range(threads)]
        for p in paths:
            if not os.path.exists(p):
                generate_file(p, size)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            start = time.time()
            list(executor.map(lambda p: measure_io(method, p), paths))
            end = time.time()

        total_MB = (size * threads) / 1024 / 1024
        bandwidth = total_MB / (end - start)
        results.append((method, size, threads, round(bandwidth, 2)))

# Print summary
print("Method\tSize_MB\tThreads\tRead_BW_MBps")
for method, size, t, bw in results:
    print(f"{method}\t{size//1024//1024}\t{t}\t{bw}")