# array_scaling_test.py
import ctypes
import torch
import os
import time
import threading

def test_array_scaling(filepath, size_gb, device_id=1):
    """Test different levels of parallelism to find optimal for 11x NVMe array"""
    
    # Load the shared library
    lib = ctypes.CDLL('./libarray_aware.so')
    
    # Set up function signatures
    functions = {
        'original_single': lib.read_to_gpu_io_uring,
        'multi_simple_4': lib.multi_reader_simple,
        'array_aware_8': lib.array_aware_parallel_read,
    }
    
    for func in functions.values():
        func.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
        func.restype = None
    
    # Set CUDA device
    torch.cuda.set_device(device_id)
    
    # Calculate size in bytes
    size_bytes = int(size_gb * 1024 * 1024 * 1024)
    
    # Prepare filepath
    filepath_bytes = os.path.abspath(filepath).encode("utf-8")
    
    print(f"Array Scaling Test for 11x NVMe Array")
    print(f"File: {filepath}")
    print(f"Size: {size_gb} GB")
    print(f"Hardware: 64 cores, 512GB RAM, 11x NVMe PCIe4, 1MB chunks")
    print("=" * 70)
    
    results = {}
    
    for version_name, func in functions.items():
        print(f"\nTesting {version_name.upper().replace('_', ' ')}:")
        print("-" * 50)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Allocate fresh tensor
        gpu_tensor = torch.empty(size_bytes, dtype=torch.uint8, device=f'cuda:{device_id}')
        torch.cuda.synchronize()
        
        # Run the test
        print("Starting transfer...")
        start = time.perf_counter()
        func(filepath_bytes, ctypes.c_void_p(gpu_tensor.data_ptr()), size_bytes)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = end - start
        bandwidth = size_gb / elapsed
        results[version_name] = bandwidth
        
        print(f"Python timing: {elapsed:.4f}s -> {bandwidth:.2f} GB/s")
        print(f"Estimated array utilization: {bandwidth/30*100:.1f}% (assuming 30 GB/s array capacity)")
        
        # Clear tensor
        del gpu_tensor
    
    # Summary
    print("\n" + "=" * 70)
    print("ARRAY SCALING RESULTS:")
    print("=" * 70)
    
    baseline = results['original_single']
    
    print(f"{'Version':<25} {'Bandwidth':<12} {'Array Util':<12} {'Speedup'}")
    print("-" * 70)
    
    for version_name, bandwidth in results.items():
        speedup = bandwidth / baseline
        array_util = bandwidth / 30 * 100  # Assuming 30 GB/s array capacity
        
        if version_name == 'original_single':
            print(f"{version_name.replace('_', ' ').title():<25} {bandwidth:<8.2f} GB/s {array_util:<8.1f}%     {'(baseline)'}")
        else:
            status = "🚀" if speedup > 2 else "📈" if speedup > 1.2 else "📊"
            print(f"{version_name.replace('_', ' ').title():<25} {bandwidth:<8.2f} GB/s {array_util:<8.1f}%     {speedup:<6.2f}x {status}")
    
    # Analysis
    best_version = max(results.keys(), key=lambda x: results[x])
    best_bandwidth = results[best_version]
    
    print("-" * 70)
    if best_version == 'original_single':
        print("ANALYSIS: Single-threaded is still best")
        print("Your array might be optimized for single large sequential reads")
    else:
        speedup = best_bandwidth / baseline
        print(f"WINNER: {best_version.replace('_', ' ').title()}")
        print(f"Best bandwidth: {best_bandwidth:.2f} GB/s ({speedup:.1f}x speedup)")
        print(f"Array utilization: {best_bandwidth/30*100:.1f}%")
        
        if best_bandwidth > 15:
            print("🎉 Excellent! You're saturating significant array bandwidth!")
        elif best_bandwidth > 10:
            print("📈 Good improvement! Still room for more array utilization.")
        else:
            print("📊 Modest improvement. Array might prefer different access patterns.")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if best_bandwidth < 15:
        print("• Try more parallel readers (your array can likely handle 8-16)")
        print("• Consider process-based parallelism instead of threads")
        print("• Test different read sizes (1MB, 2MB to match array chunk size)")
        print("• Check if array has preferred alignment/stride patterns")
    else:
        print("• Excellent performance! Consider testing even more parallelism")
        print("• You might be approaching your array's sequential read limit")
    
    print(f"• Current single-thread efficiency: {baseline/30*100:.1f}% of estimated array capacity")
    print(f"• Theoretical max improvement: {30/baseline:.1f}x (if you can fully saturate array)")
    
    return results

def benchmark_thread_scaling(filepath, size_gb, device_id=1):
    """Test different numbers of threads to find optimal scaling"""
    
    print(f"Thread Scaling Analysis")
    print(f"Testing 1, 2, 4, 8 parallel readers to find sweet spot")
    print("=" * 60)
    
    # This would require building multiple versions with different thread counts
    # For now, just show the concept
    
    thread_counts = [1, 2, 4, 8]
    results = {}
    
    # In practice, you'd have different compiled versions or parameters
    print("Thread scaling test would go here...")
    print("Suggestion: Compile versions with 2, 4, 8, 16 parallel readers")
    print("and test each to find the optimal parallelism for your array")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 array_scaling_test.py <filepath> [size_gb] [device_id]")
        print("Example: python3 array_scaling_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1")
        print("")
        print("This tests parallel approaches optimized for your 11x NVMe array")
        print("Goal: Saturate more of your array's ~30+ GB/s potential bandwidth")
        sys.exit(1)
    
    filepath = sys.argv[1]
    size_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    
    if size_gb > 4.0:
        print(f"Warning: {size_gb} GB might be too large for RTX A2000 (5.67GB total)")
        print("Consider using 3.0 GB or less")
    
    print("Testing array-aware optimizations for high-end storage...")
    test_array_scaling(filepath, size_gb, device_id)