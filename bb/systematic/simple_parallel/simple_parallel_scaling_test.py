import ctypes
import torch
import os
import time

def test_simple_parallel_scaling(filepath, size_gb, device_id=1):
    """Test simple parallel approaches using OpenMP"""
    
    # Load the shared library
    lib = ctypes.CDLL('./libsimple_parallel.so')
    
    # Set up function signatures
    functions = {
        'original_single': lib.read_to_gpu_io_uring,
        'dual_reader': lib.dual_reader_simple,
        'quad_reader': lib.quad_reader_simple,
        'array_aligned_1mb': lib.array_aligned_1mb,
        'multi_fd_simple': lib.multi_fd_simple,
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
    
    print(f"Simple Parallel Scaling Test for 11x NVMe Array")
    print(f"File: {filepath}")
    print(f"Size: {size_gb} GB")
    print(f"Hardware: 64 cores, 512GB RAM, 11x NVMe PCIe4, 1MB chunks")
    print("Goal: Scale from 7 GB/s to 15+ GB/s using parallelism")
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
        
        # Estimate array utilization (assuming 30 GB/s theoretical max)
        array_util = bandwidth / 30 * 100
        print(f"Estimated array utilization: {array_util:.1f}%")
        
        # Clear tensor
        del gpu_tensor
    
    # Summary
    print("\n" + "=" * 70)
    print("PARALLEL SCALING RESULTS:")
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
            if speedup >= 2.0:
                status = "🚀 EXCELLENT"
            elif speedup >= 1.5:
                status = "📈 GOOD"
            elif speedup >= 1.1:
                status = "📊 MODEST"
            else:
                status = "📉 NO GAIN"
            
            print(f"{version_name.replace('_', ' ').title():<25} {bandwidth:<8.2f} GB/s {array_util:<8.1f}%     {speedup:<6.2f}x {status}")
    
    # Analysis
    best_version = max(results.keys(), key=lambda x: results[x])
    best_bandwidth = results[best_version]
    
    print("-" * 70)
    print("ANALYSIS:")
    
    if best_version == 'original_single':
        print("• Your single-threaded code is still the best")
        print("• Your array might be optimized for large sequential reads")
        print("• Consider testing larger block sizes or different array settings")
    else:
        speedup = best_bandwidth / baseline
        print(f"• WINNER: {best_version.replace('_', ' ').title()}")
        print(f"• Best bandwidth: {best_bandwidth:.2f} GB/s ({speedup:.1f}x speedup)")
        print(f"• Array utilization improved to: {best_bandwidth/30*100:.1f}%")
        
        if speedup >= 2.0:
            print("• 🚀 Excellent scaling! Your array loves parallel access")
            print("• Consider testing even more readers (8, 16)")
        elif speedup >= 1.5:
            print("• 📈 Good scaling! Parallel approach is working")
            print("• Try tuning: more readers, different block sizes")
        else:
            print("• 📊 Modest scaling. Array might prefer different patterns")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if best_bandwidth < 12:
        print("• Current approach: Array not fully utilized")
        print("• Try: Process-based parallelism instead of threads")
        print("• Try: More parallel readers (8, 16, 32)")
        print("• Check: Array RAID/striping configuration")
        print("• Test: Different I/O patterns (random vs sequential)")
    elif best_bandwidth < 20:
        print("• Good progress! Getting substantial array utilization")
        print("• Try: Scale up successful approach (more of what works)")
        print("• Test: Optimize GPU transfer coordination")
    else:
        print("• Excellent! You're approaching array bandwidth limits")
        print("• Focus: GPU transfer optimization")
        print("• Consider: Multiple GPUs if available")
    
    print(f"• Theoretical improvement ceiling: {30/baseline:.1f}x (if array is 30 GB/s)")
    print(f"• Your single-thread efficiency: {baseline/30*100:.1f}% of estimated array capacity")
    
    return results

def quick_parallel_test(filepath, size_gb, device_id=1):
    """Quick test with smaller size"""
    print("Quick parallel scaling test...")
    return test_simple_parallel_scaling(filepath, size_gb, device_id)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 simple_parallel_scaling_test.py <filepath> [size_gb] [device_id]")
        print("Example: python3 simple_parallel_scaling_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1")
        print("")
        print("Tests simple parallel approaches for 11x NVMe array:")
        print("• Single reader (your excellent baseline)")
        print("• Dual reader (2 parallel file readers)")  
        print("• Quad reader (4 parallel file readers)")
        print("• 1MB aligned blocks (matching array chunk size)")
        print("• Multi-FD simple (2 file descriptors, sequential)")
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
    
    test_simple_parallel_scaling(filepath, size_gb, device_id)