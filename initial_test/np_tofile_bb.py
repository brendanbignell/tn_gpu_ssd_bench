#!/usr/bin/env python3

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def flush_caches():
    """Flush system caches"""
    try:
        os.system('sync')
    except:
        pass

def test_optimized_io():
    """Optimized version of your original script"""
    
    # Use larger batch sizes for better RAID performance
    batches = list(range(10, 500, 20))  # ~1.3MB to ~64MB files
    bw_writes = []
    bw_reads = []
    
    test_file = "/mnt/kvcache/test_write.bin"
    buffer_size = 64 * 1024 * 1024  # 64MB buffer
    
    print(f"Testing file sizes from {batches[0] * 32 * 1024 * 2 / 1e6:.1f}MB to {batches[-1] * 32 * 1024 * 2 / 1e6:.1f}MB")
    
    for B in tqdm(batches):
        data = np.random.randn(B, 32, 1024).astype(np.float16)
        
        # Pre-allocate file space
        with open(test_file, 'wb') as f:
            f.seek(data.nbytes - 1)
            f.write(b'\0')
        
        # WRITE TEST with large buffer
        start = time.time()
        with open(test_file, 'wb', buffering=buffer_size) as f:
            data.tofile(f)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        end = time.time()
        
        dur = end - start
        bw = data.nbytes / dur / 1e9
        bw_writes.append(bw)
        
        flush_caches()
        
        # READ TEST with large buffer
        start = time.time()
        with open(test_file, 'rb', buffering=buffer_size) as f:
            data_read = np.fromfile(f, dtype=np.float16)
            data_read = data_read.reshape(B, 32, 1024)
        end = time.time()
        
        dur = end - start
        bw = data_read.nbytes / dur / 1e9
        bw_reads.append(bw)
    
    return batches, bw_writes, bw_reads

def test_chunked_io():
    """Test with chunked I/O"""
    
    batches = list(range(10, 500, 20))
    bw_writes = []
    bw_reads = []
    
    test_file = "/mnt/kvcache/test_write_chunked.bin"
    chunk_size = 64 * 1024 * 1024  # 64MB chunks
    
    print("Testing chunked I/O...")
    
    for B in tqdm(batches):
        data = np.random.randn(B, 32, 1024).astype(np.float16)
        data_bytes = data.tobytes()
        
        # CHUNKED WRITE
        start = time.time()
        with open(test_file, 'wb', buffering=0) as f:  # Unbuffered
            for i in range(0, len(data_bytes), chunk_size):
                chunk = data_bytes[i:i+chunk_size]
                f.write(chunk)
            os.fsync(f.fileno())
        end = time.time()
        
        dur = end - start
        bw = data.nbytes / dur / 1e9
        bw_writes.append(bw)
        
        flush_caches()
        
        # CHUNKED READ
        start = time.time()
        with open(test_file, 'rb', buffering=0) as f:
            chunks = []
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
            data_bytes = b''.join(chunks)
            data_read = np.frombuffer(data_bytes, dtype=np.float16).reshape(B, 32, 1024)
        end = time.time()
        
        dur = end - start
        bw = data_read.nbytes / dur / 1e9
        bw_reads.append(bw)
    
    return batches, bw_writes, bw_reads

def main():
    print("=== Storage Benchmark Test ===")
    
    # Test 1: Optimized version of your original approach
    print("\n1. Testing optimized numpy approach...")
    batches1, writes1, reads1 = test_optimized_io()
    
    # Test 2: Chunked I/O
    print("\n2. Testing chunked I/O approach...")
    batches2, writes2, reads2 = test_chunked_io()
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    # Write speeds
    plt.subplot(1, 2, 1)
    plt.plot(batches1, writes1, label='Optimized numpy write', marker='o', markersize=3)
    plt.plot(batches2, writes2, label='Chunked write', marker='s', markersize=3)
    plt.xlabel('Batch size (B)')
    plt.ylabel('Write Bandwidth (GB/s)')
    plt.title('Write Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    # Read speeds
    plt.subplot(1, 2, 2)
    plt.plot(batches1, reads1, label='Optimized numpy read', marker='x', markersize=3)
    plt.plot(batches2, reads2, label='Chunked read', marker='+', markersize=3)
    plt.xlabel('Batch size (B)')
    plt.ylabel('Read Bandwidth (GB/s)')
    plt.title('Read Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("optimized_ssd_benchmark.png", dpi=150)
    plt.show()
    
    # Print results
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Optimized numpy - Max Write: {max(writes1):6.2f} GB/s, Max Read: {max(reads1):6.2f} GB/s")
    print(f"Chunked I/O     - Max Write: {max(writes2):6.2f} GB/s, Max Read: {max(reads2):6.2f} GB/s")
    
    # Cleanup
    try:
        os.remove("/mnt/kvcache/test_write.bin")
        os.remove("/mnt/kvcache/test_write_chunked.bin")
    except:
        pass
    
    print("\nIf performance is still low, check:")
    print("1. RAID stripe size: cat /proc/mdstat")
    print("2. Queue depth: cat /sys/block/nvme*/queue/nr_requests") 
    print("3. CPU governor: cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    print("4. Monitor during test: iostat -x 1")

if __name__ == "__main__":
    main()