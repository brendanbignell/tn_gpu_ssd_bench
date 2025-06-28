import ctypes
import torch
import os
import time

def test_micro_optimizations(filepath, size_gb, device_id=1, runs=5):
    """Test micro-optimizations against your excellent baseline"""
    
    # Load the shared library
    lib = ctypes.CDLL('./libmicro_opt.so')
    
    # Set up function signatures
    functions = {
        'original': lib.read_to_gpu_io_uring,
        'priority_stream': lib.micro_opt_priority_stream,
        'batch_submit': lib.micro_opt_batch_submit,
        '6mb_blocks': lib.micro_opt_6mb_blocks,
        'nonblocking': lib.micro_opt_nonblocking,
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
    
    print(f"Micro-Optimization Test")
    print(f"File: {filepath}")
    print(f"Size: {size_gb} GB")
    print(f"Device: GPU {device_id}")
    print(f"Runs per version: {runs}")
    print("=" * 60)
    print("Testing very small tweaks to your already excellent code...")
    print("=" * 60)
    
    results = {}
    
    for version_name, func in functions.items():
        print(f"\nTesting {version_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        bandwidths = []
        
        for run in range(runs):
            # Clear GPU memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            
            # Clear tensor
            del gpu_tensor
        
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
    print("MICRO-OPTIMIZATION RESULTS:")
    print("=" * 60)
    
    baseline = results['original']['avg']
    
    print(f"{'Version':<20} {'Bandwidth':<12} {'vs Original':<12} {'Improvement'}")
    print("-" * 60)
    
    for version_name, result in results.items():
        bandwidth = result['avg']
        vs_original = bandwidth / baseline
        improvement = ((bandwidth - baseline) / baseline) * 100
        
        if version_name == 'original':
            print(f"{version_name.replace('_', ' ').title():<20} {bandwidth:<8.2f} GB/s {'(baseline)'}")
        else:
            status = "🎉" if improvement > 1 else "📊" if improvement > -1 else "📉"
            print(f"{version_name.replace('_', ' ').title():<20} {bandwidth:<8.2f} GB/s {vs_original:<6.2f}x {improvement:>+5.1f}% {status}")
    
    # Find best version
    best_version = max(results.keys(), key=lambda x: results[x]['avg'])
    best_bandwidth = results[best_version]['avg']
    
    print("-" * 60)
    if best_version == 'original':
        print("RESULT: Your original code is still unbeaten! 🏆")
        print("This confirms your code is exceptionally well optimized.")
    else:
        improvement = ((best_bandwidth - baseline) / baseline) * 100
        print(f"WINNER: {best_version.replace('_', ' ').title()} 🎉")
        print(f"Best bandwidth: {best_bandwidth:.2f} GB/s (+{improvement:.1f}% improvement)")
        print("Even a small improvement over your excellent baseline is significant!")
    
    # Show any improvements
    improvements = [(k, v['avg']) for k, v in results.items() 
                   if k != 'original' and v['avg'] > baseline]
    
    if improvements:
        print(f"\nMicro-optimizations that helped:")
        for name, bw in sorted(improvements, key=lambda x: x[1], reverse=True):
            improvement = ((bw - baseline) / baseline) * 100
            print(f"  • {name.replace('_', ' ').title()}: +{improvement:.1f}%")
    else:
        print(f"\nNo micro-optimizations beat your original (which is excellent!)")
    
    return results

def quick_micro_test(filepath, size_gb, device_id=1):
    """Quick single-run test of micro-optimizations"""
    return test_micro_optimizations(filepath, size_gb, device_id, runs=1)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 micro_test.py <filepath> [size_gb] [device_id] [--quick]")
        print("Example: python3 micro_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1")
        print("         python3 micro_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1 --quick")
        print("")
        print("Note: Use 3GB or less for RTX A2000 (5.67GB total GPU memory)")
        sys.exit(1)
    
    filepath = sys.argv[1]
    size_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    quick = '--quick' in sys.argv
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    
    if size_gb > 4.0:
        print(f"Warning: {size_gb} GB might be too large for RTX A2000 (5.67GB total)")
        print("Consider using 3.0 GB or less")
    
    if quick:
        print("Running quick micro-optimization test (1 run per version)...")
        quick_micro_test(filepath, size_gb, device_id)
    else:
        print("Running full micro-optimization test (5 runs per version)...")
        test_micro_optimizations(filepath, size_gb, device_id)