#!/usr/bin/env python3

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess

def flush_caches():
    """Flush system caches"""
    try:
        os.system('sync')
        # Try to drop caches if possible
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], 
                      check=False, capture_output=True)
    except:
        pass

def test_parallel_io(num_threads=4):
    """Test with parallel I/O to fully utilize RAID0"""
    
    print(f"=== PARALLEL I/O TEST ({num_threads} threads) ===")
    
    # Test with large files that should stress the RAID
    file_sizes_gb = [0.5, 1.0, 2.0, 4.0, 8.0]  # 0.5GB to 8GB per thread
    
    results = {'sizes': [], 'writes': [], 'reads': []}
    
    def write_worker(thread_id, data, chunk_size=256*1024*1024):
        """Worker function for parallel writes"""
        filename = f"/mnt/kvcache/parallel_test_{thread_id}.bin"
        
        with open(filename, 'wb', buffering=0) as f:
            data_bytes = data.tobytes()
            for i in range(0, len(data_bytes), chunk_size):
                chunk = data_bytes[i:i+chunk_size]
                f.write(chunk)
            os.fsync(f.fileno())
        
        return len(data_bytes)
    
    def read_worker(thread_id, expected_shape, chunk_size=256*1024*1024):
        """Worker function for parallel reads"""
        filename = f"/mnt/kvcache/parallel_test_{thread_id}.bin"
        
        with open(filename, 'rb', buffering=0) as f:
            chunks = []
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
            
            data_bytes = b''.join(chunks)
            data = np.frombuffer(data_bytes, dtype=np.float16).reshape(expected_shape)
            # Force memory access
            _ = data.sum()
        
        return len(data_bytes)
    
    for size_gb in file_sizes_gb:
        print(f"\nTesting {size_gb}GB per thread ({size_gb * num_threads}GB total)...")
        
        # Calculate array dimensions for target size
        target_bytes = int(size_gb * 1e9)
        elements_per_thread = target_bytes // 2  # float16 = 2 bytes
        # Shape: (batches, 32, 1024) where batches * 32 * 1024 = elements_per_thread
        batches = elements_per_thread // (32 * 1024)
        
        if batches < 1:
            continue
            
        actual_size_gb = (batches * 32 * 1024 * 2) / 1e9
        total_size_gb = actual_size_gb * num_threads
        
        print(f"  Actual: {actual_size_gb:.2f}GB per thread, {total_size_gb:.2f}GB total")
        
        # Generate data for each thread
        thread_data = []
        for i in range(num_threads):
            data = np.random.randn(batches, 32, 1024).astype(np.float16)
            thread_data.append(data)
        
        # PARALLEL WRITE TEST
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i, data in enumerate(thread_data):
                future = executor.submit(write_worker, i, data)
                futures.append(future)
            
            total_bytes_written = sum(future.result() for future in futures)
        
        write_time = time.time() - start_time
        write_bw = total_bytes_written / write_time / 1e9
        
        flush_caches()
        time.sleep(1)
        
        # PARALLEL READ TEST
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i, data in enumerate(thread_data):
                future = executor.submit(read_worker, i, data.shape)
                futures.append(future)
            
            total_bytes_read = sum(future.result() for future in futures)
        
        read_time = time.time() - start_time
        read_bw = total_bytes_read / read_time / 1e9
        
        results['sizes'].append(total_size_gb)
        results['writes'].append(write_bw)
        results['reads'].append(read_bw)
        
        print(f"  Write: {write_bw:.2f} GB/s, Read: {read_bw:.2f} GB/s")
        
        # Cleanup
        for i in range(num_threads):
            try:
                os.remove(f"/mnt/kvcache/parallel_test_{i}.bin")
            except:
                pass
    
    return results

def test_optimized_sequential():
    """Test optimized sequential I/O with very large files"""
    
    print("=== OPTIMIZED SEQUENTIAL I/O TEST ===")
    
    # Test with files optimized for your 1024K chunk size
    # Use chunk sizes that are multiples of RAID chunk size
    chunk_sizes_mb = [256, 512, 1024]  # 256MB, 512MB, 1GB chunks
    file_size_gb = 4  # 4GB files
    
    results = {}
    
    target_bytes = int(file_size_gb * 1e9)
    elements = target_bytes // 2  # float16 = 2 bytes
    batches = elements // (32 * 1024)
    
    for chunk_mb in chunk_sizes_mb:
        print(f"\nTesting {chunk_mb}MB chunks...")
        
        chunk_size = chunk_mb * 1024 * 1024
        data = np.random.randn(batches, 32, 1024).astype(np.float16)
        
        filename = "/mnt/kvcache/sequential_test.bin"
        
        # WRITE TEST
        start_time = time.time()
        with open(filename, 'wb', buffering=0) as f:
            data_bytes = data.tobytes()
            for i in range(0, len(data_bytes), chunk_size):
                chunk = data_bytes[i:i+chunk_size]
                f.write(chunk)
            os.fsync(f.fileno())
        
        write_time = time.time() - start_time
        write_bw = data.nbytes / write_time / 1e9
        
        flush_caches()
        time.sleep(1)
        
        # READ TEST
        start_time = time.time()
        with open(filename, 'rb', buffering=0) as f:
            chunks = []
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
            
            data_bytes = b''.join(chunks)
            data_read = np.frombuffer(data_bytes, dtype=np.float16)
            _ = data_read.sum()  # Force memory access
        
        read_time = time.time() - start_time
        read_bw = len(data_bytes) / read_time / 1e9
        
        results[f"{chunk_mb}MB"] = {'write': write_bw, 'read': read_bw}
        
        print(f"  Write: {write_bw:.2f} GB/s, Read: {read_bw:.2f} GB/s")
        
        # Cleanup
        try:
            os.remove(filename)
        except:
            pass
    
    return results

def benchmark_individual_drives():
    """Benchmark individual drives in your RAID to identify bottlenecks"""
    
    print("=== INDIVIDUAL DRIVE BENCHMARKS ===")
    
    # Test the different drive types in your RAID
    drives_to_test = [
        ("/dev/nvme0n1", "SK Hynix PC811 2TB"),
        ("/dev/nvme13n1", "WD PC SN820 4TB"),
    ]
    
    for drive, description in drives_to_test:
        print(f"\nTesting {drive} ({description})...")
        
        try:
            # Use dd for raw drive performance
            cmd = f"sudo dd if={drive} of=/dev/null bs=1M count=1024 iflag=direct 2>&1"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            # Parse dd output for speed
            output = result.stderr
            if "copied" in output and "GB/s" in output:
                speed_line = [line for line in output.split('\n') if 'GB/s' in line][-1]
                print(f"  Raw read speed: {speed_line.split()[-2]} {speed_line.split()[-1]}")
            elif "copied" in output and "MB/s" in output:
                speed_line = [line for line in output.split('\n') if 'MB/s' in line][-1]
                speed_mb = float(speed_line.split()[-2])
                print(f"  Raw read speed: {speed_mb:.0f} MB/s ({speed_mb/1000:.2f} GB/s)")
            else:
                print("  Could not parse speed from dd output")
                
        except Exception as e:
            print(f"  Error testing {drive}: {e}")

def main():
    print("=== COMPREHENSIVE RAID PERFORMANCE TEST ===")
    print("============================================")
    
    # Check if fixes were applied
    cpu_gov = open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()
    print(f"CPU Governor: {cpu_gov}")
    
    if cpu_gov != 'performance':
        print("⚠️  WARNING: CPU governor is not set to 'performance'!")
        print("   Run the performance fixes script first for best results.")
    
    # Test 1: Individual drive benchmarks
    benchmark_individual_drives()
    
    # Test 2: Optimized sequential I/O
    seq_results = test_optimized_sequential()
    
    # Test 3: Parallel I/O (2 threads, then 4 threads)
    parallel_2 = test_parallel_io(num_threads=2)
    parallel_4 = test_parallel_io(num_threads=4)
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print("\nSequential I/O (4GB files):")
    for chunk_size, perf in seq_results.items():
        print(f"  {chunk_size:8} chunks - Write: {perf['write']:6.2f} GB/s, Read: {perf['read']:6.2f} GB/s")
    
    print(f"\nParallel I/O (2 threads):")
    if parallel_2['writes']:
        max_write_2 = max(parallel_2['writes'])
        max_read_2 = max(parallel_2['reads'])
        print(f"  Max Write: {max_write_2:.2f} GB/s, Max Read: {max_read_2:.2f} GB/s")
    
    print(f"\nParallel I/O (4 threads):")
    if parallel_4['writes']:
        max_write_4 = max(parallel_4['writes'])
        max_read_4 = max(parallel_4['reads'])
        print(f"  Max Write: {max_write_4:.2f} GB/s, Max Read: {max_read_4:.2f} GB/s")
    
    # Expected vs actual
    print(f"\nExpected performance for 11-drive RAID0:")
    print(f"  Theoretical maximum: 40-70+ GB/s")
    print(f"  With mixed drives: 20-30 GB/s (limited by slowest drives)")
    
    best_write = max(
        max(seq_results[k]['write'] for k in seq_results),
        max(parallel_2['writes']) if parallel_2['writes'] else 0,
        max(parallel_4['writes']) if parallel_4['writes'] else 0
    )
    
    best_read = max(
        max(seq_results[k]['read'] for k in seq_results),
        max(parallel_2['reads']) if parallel_2['reads'] else 0,
        max(parallel_4['reads']) if parallel_4['reads'] else 0
    )
    
    print(f"\nYour best results:")
    print(f"  Write: {best_write:.2f} GB/s")
    print(f"  Read: {best_read:.2f} GB/s")
    
    if best_write < 5 or best_read < 5:
        print(f"\n⚠️  Performance is still much lower than expected!")
        print(f"   Possible issues:")
        print(f"   1. Mixed drive types limiting array performance")
        print(f"   2. PCIe lane limitations or NUMA issues")
        print(f"   3. Individual drives underperforming")
        print(f"   4. Need to rebuild RAID with matched drives")

if __name__ == "__main__":
    main()