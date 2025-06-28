import os
import threading
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import csv
import mmap

BLOCK_SIZES_MB = [2 ** exp for exp in range(0, 11)]  # 1MB to 1GB
BLOCK_SIZES = [size * 1024 * 1024 for size in BLOCK_SIZES_MB]
THREAD_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 256, 1024]
FILE_PATH = "/mnt/kvcache/test_read_block"
REPEATS = 5
METHODS = ["read", "mmap"]

def generate_file(size_bytes):
    with open(FILE_PATH, "wb") as f:
        f.write(os.urandom(size_bytes))

def threaded_read(method: str, size: int, threads: int):
    def read_worker(offset, length):
        if method == "read":
            with open(FILE_PATH, "rb") as f:
                f.seek(offset)
                _ = f.read(length)
        elif method == "mmap":
            with open(FILE_PATH, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                _ = mm[offset:offset + length]
                mm.close()

    def monitor_io():
        proc = psutil.Process()
        start_cpu = proc.cpu_times()
        start_io = proc.io_counters()
        return start_cpu, start_io

    def compute_stats(start_cpu, start_io):
        proc = psutil.Process()
        end_cpu = proc.cpu_times()
        end_io = proc.io_counters()
        cpu_user = end_cpu.user - start_cpu.user
        cpu_sys = end_cpu.system - start_cpu.system
        io_read = end_io.read_bytes - start_io.read_bytes
        return cpu_user, cpu_sys, io_read

    per_thread = size // threads
    barrier = threading.Barrier(threads + 1)
    threads_list = []

    def worker(offset, length):
        barrier.wait()
        read_worker(offset, length)

    for i in range(threads):
        offset = i * per_thread
        t = threading.Thread(target=worker, args=(offset, per_thread))
        t.start()
        threads_list.append(t)

    start_cpu, start_io = monitor_io()
    start = time.perf_counter()
    barrier.wait()
    for t in threads_list:
        t.join()
    end = time.perf_counter()
    cpu_user, cpu_sys, io_read = compute_stats(start_cpu, start_io)

    total_time = end - start
    bandwidth = size / total_time / (1024 ** 2)
    return bandwidth, cpu_user, cpu_sys, io_read

# Results data structure
results = {
    method: {
        threads: {
            bs: {"bws": [], "cpu_user": [], "cpu_sys": [], "io_read": []}
            for bs in BLOCK_SIZES_MB
        } for threads in THREAD_OPTIONS
    } for method in METHODS
}

# CSV output
with open("thread_scaling_results_full.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "BlockSize_MB", "Threads", "Bandwidth_MBps", "StdDev", "CPU_User", "CPU_Sys", "IO_Read_MB"])

    for method in METHODS:
        for threads in THREAD_OPTIONS:
            for size, size_mb in zip(BLOCK_SIZES, BLOCK_SIZES_MB):
                print(f"Running {method} for block size {size_mb} MB with {threads} threads...")
                generate_file(size)
                bws, cpu_us, cpu_ss, io_rs = [], [], [], []
                for _ in range(REPEATS):
                    bw, cpu_u, cpu_s, io_r = threaded_read(method, size, threads)
                    bws.append(bw)
                    cpu_us.append(cpu_u)
                    cpu_ss.append(cpu_s)
                    io_rs.append(io_r / 1024 / 1024)  # to MB
                mean_bw = np.mean(bws)
                std_bw = np.std(bws)
                mean_cpu_u = np.mean(cpu_us)
                mean_cpu_s = np.mean(cpu_ss)
                mean_io_r = np.mean(io_rs)
                results[method][threads][size_mb]["bws"] = mean_bw
                results[method][threads][size_mb]["std"] = std_bw
                writer.writerow([method, size_mb, threads, mean_bw, std_bw, mean_cpu_u, mean_cpu_s, mean_io_r])

# Plotting
plt.figure(figsize=(14, 8))
for method in METHODS:
    for threads in THREAD_OPTIONS:
        bws = [results[method][threads][bs]["bws"] for bs in BLOCK_SIZES_MB]
        stds = [results[method][threads][bs]["std"] for bs in BLOCK_SIZES_MB]
        label = f"{method}-{threads} threads"
        plt.plot(BLOCK_SIZES_MB, bws, marker="o", label=label)
        plt.fill_between(BLOCK_SIZES_MB, np.array(bws) - np.array(stds), np.array(bws) + np.array(stds), alpha=0.2)

plt.xlabel("Block Size (MB)")
plt.ylabel("Bandwidth (MB/s)")
plt.title("RAID0 Read Bandwidth: Method & Thread Scaling")
plt.xscale("log", base=2)
plt.xticks(BLOCK_SIZES_MB, labels=[str(x) for x in BLOCK_SIZES_MB])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("read_bandwidth_thread_method_comparison.png")
print("Saved plot as read_bandwidth_thread_method_comparison.png")