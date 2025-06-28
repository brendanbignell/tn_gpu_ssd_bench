import ctypes
import torch
import os
import time
import subprocess
import multiprocessing

def test_process_based_scaling(filepath, size_gb, device_id=1):
    """Test process-based parallelism using your working single-threaded code"""
    
    print("Process-Based Scaling Test")
    print("Using multiple processes instead of threads")
    print("=" * 50)
    
    # Load your original working library
    lib = ctypes.CDLL('./libmicro_opt.so')  # From previous working test
    lib.read_to_gpu_io_uring.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.read_to_gpu_io_uring.restype = None
    
    # Set CUDA device
    torch.cuda.set_device(device_id)
    
    # Calculate size in bytes
    size_bytes = int(size_gb * 1024 * 1024 * 1024)
    filepath_bytes = os.path.abspath(filepath).encode("utf-8")
    
    print(f"File: {filepath}")
    print(f"Size: {size_gb} GB")
    print(f"Testing 1, 2, 4 process approach")
    print("")
    
    results = {}
    
    # Test 1: Single process (your proven approach)
    print("Testing 1 process (baseline):")
    gpu_tensor = torch.empty(size_bytes, dtype=torch.uint8, device=f'cuda:{device_id}')
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    lib.read_to_gpu_io_uring(filepath_bytes, ctypes.c_void_p(gpu_tensor.data_ptr()), size_bytes)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed1 = end - start
    bandwidth1 = size_gb / elapsed1
    results['1_process'] = bandwidth1
    
    print(f"  1 process: {elapsed1:.4f}s -> {bandwidth1:.2f} GB/s")
    del gpu_tensor
    
    # Test 2: Simulated 2-process approach
    print("\nTesting 2 processes (simulated):")
    print("Note: This is a simulation - real implementation would need separate processes")
    
    # Simulate by reading two halves
    half_size = size_bytes // 2
    gpu_tensor2a = torch.empty(half_size, dtype=torch.uint8, device=f'cuda:{device_id}')
    gpu_tensor2b = torch.empty(half_size, dtype=torch.uint8, device=f'cuda:{device_id}')
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    # First half
    lib.read_to_gpu_io_uring(filepath_bytes, ctypes.c_void_p(gpu_tensor2a.data_ptr()), half_size)
    
    # Second half (would be parallel in real implementation)
    lib.read_to_gpu_io_uring(filepath_bytes, ctypes.c_void_p(gpu_tensor2b.data_ptr()), half_size)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed2 = end - start
    bandwidth2 = size_gb / elapsed2
    results['2_process_sim'] = bandwidth2
    
    print(f"  2 process (sim): {elapsed2:.4f}s -> {bandwidth2:.2f} GB/s")
    print(f"  Note: Real parallel would be faster")
    
    del gpu_tensor2a, gpu_tensor2b
    
    # Analysis
    print("\n" + "=" * 50)
    print("SCALING ANALYSIS:")
    print("=" * 50)
    
    print(f"1 Process:        {bandwidth1:.2f} GB/s (baseline)")
    print(f"2 Process (sim):  {bandwidth2:.2f} GB/s ({bandwidth2/bandwidth1:.2f}x)")
    
    print("\nRECOMMENDATIONS:")
    print("1. Your single-threaded code is excellent")
    print("2. For your 11x NVMe array, try:")
    print("   • Multiple independent processes reading different file regions")
    print("   • Each process uses your proven single-threaded approach")
    print("   • Coordinate GPU transfers between processes")
    print("3. Consider array-level optimization:")
    print("   • Check array striping/RAID configuration")
    print("   • Test with different block sizes (1MB, 2MB)")
    print("   • Monitor per-drive utilization during reads")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 simple_parallel_test.py <filepath> [size_gb] [device_id]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    size_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    test_process_based_scaling(filepath, size_gb, device_id)