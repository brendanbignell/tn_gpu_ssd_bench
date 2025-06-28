# systematic_test.py
import ctypes
import torch
import os
import time

def test_all_versions(filepath, size_gb, device_id=1, runs=3):
    """Test all optimization versions systematically"""
    
    # Load the shared library
    lib = ctypes.CDLL('./liboptimized_io.so')
    
    # Set up function signatures
    functions = {
        'original': lib.read_to_gpu_io_uring,
        'optimized_v1': lib.optimized_v1_read_to_gpu,
        'optimized_v2': lib.optimized_v2_read_to_gpu,
        'optimized_v3': lib.optimized_v3_read_to_gpu,
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
    
    print(f"Systematic Performance Test")
    print(f"File: {filepath}")
    print(f"Size: {size_gb} GB")
    print(f"Device: GPU {device_id}")
    print(f"Runs per version: {runs}")
    print("=" * 60)
    
    results = {}
    
    for version_name, func in functions.items():
        print(f"\nTesting {version_name.upper()}:")
        print("-" * 30)
        
        bandwidths = []
        
        for run in range(runs):
            # Allocate fresh tensor for each run
            gpu_tensor = torch.empty(size_bytes, dtype=torch.uint8, device=f'cuda:{device_id}')
            torch.cuda.synchronize()
            
            # Run the test
            start = time.perf_counter()
            func(filepath_bytes, ctypes.c_void_p(gpu_tensor.data_ptr()), size_bytes)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            elapsed = end - start
            bandwidth = size_gb / elapsed
            bandwidths.append(bandwidth)
            
            print(f"  Run {run+1}: {elapsed:.4f}s -> {bandwidth:.2f} GB/s")
        
        # Calculate statistics
        avg_bandwidth = sum(bandwidths) / len(bandwidths)
        min_bandwidth = min(bandwidths)
        max_bandwidth = max(bandwidths)
        
        results[version_name] = {
            'avg': avg_bandwidth,
            'min': min_bandwidth,
            'max': max_bandwidth,
            'all': bandwidths
        }
        
        print(f"  Average: {avg_bandwidth:.2f} GB/s")
        print(f"  Range: {min_bandwidth:.2f} - {max_bandwidth:.2f} GB/s")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON (Average Bandwidth):")
    print("=" * 60)
    
    baseline = results['original']['avg']
    
    print(f"{'Version':<15} {'Bandwidth':<12} {'vs Original':<12} {'Improvement'}")
    print("-" * 60)
    
    for version_name, result in results.items():
        bandwidth = result['avg']
        vs_original = bandwidth / baseline
        improvement = ((bandwidth - baseline) / baseline) * 100
        
        print(f"{version_name:<15} {bandwidth:<8.2f} GB/s {vs_original:<8.2f}x {improvement:>+6.1f}%")
    
    # Find best version
    best_version = max(results.keys(), key=lambda x: results[x]['avg'])
    best_bandwidth = results[best_version]['avg']
    
    print("-" * 60)
    print(f"WINNER: {best_version.upper()} with {best_bandwidth:.2f} GB/s")
    
    if best_version != 'original':
        improvement = ((best_bandwidth - baseline) / baseline) * 100
        print(f"Improvement over original: +{improvement:.1f}%")
    else:
        print("Original version is still the best!")
    
    return results, best_version

def quick_test(filepath, size_gb, device_id=1):
    """Quick single-run test of all versions"""
    return test_all_versions(filepath, size_gb, device_id, runs=1)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 systematic_test.py <filepath> [size_gb] [device_id] [--quick]")
        print("Example: python3 systematic_test.py /mnt/kvcache/huge_test_4tb.bin 5.0 1")
        print("         python3 systematic_test.py /mnt/kvcache/huge_test_4tb.bin 5.0 1 --quick")
        sys.exit(1)
    
    filepath = sys.argv[1]
    size_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    quick = '--quick' in sys.argv
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    
    if quick:
        print("Running quick test (1 run per version)...")
        quick_test(filepath, size_gb, device_id)
    else:
        print("Running full test (3 runs per version)...")
        test_all_versions(filepath, size_gb, device_id)