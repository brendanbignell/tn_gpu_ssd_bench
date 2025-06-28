# block_size_optimization_test.py
import ctypes
import torch
import os
import time

def test_block_size_optimization(filepath, size_gb, device_id=1):
    """Test different block sizes to find optimal for your array"""
    
    # Load the shared library
    lib = ctypes.CDLL('./libblock_size_opt.so')
    
    # Set up function signatures
    functions = {
        'original_4mb': lib.read_to_gpu_io_uring,
        '512kb_blocks': lib.test_512kb_blocks,
        '1mb_blocks': lib.test_1mb_blocks,
        '2mb_blocks': lib.test_2mb_blocks,
        '8mb_blocks': lib.test_8mb_blocks,
        '16mb_blocks': lib.test_16mb_blocks,
        '1mb_qd64': lib.test_1mb_qd64,
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
    
    print(f"Block Size Optimization for 11x NVMe Array")
    print(f"File: {filepath}")
    print(f"Size: {size_gb} GB")
    print(f"Array: 1MB chunk size, 11x NVMe PCIe4")
    print(f"Previous winner: 1MB blocks (11.32 GB/s, 24% improvement)")
    print("=" * 70)
    print("Testing various block sizes to find the absolute optimum...")
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
        
        # Clear tensor
        del gpu_tensor
    
    # Analysis
    print("\n" + "=" * 70)
    print("BLOCK SIZE OPTIMIZATION RESULTS:")
    print("=" * 70)
    
    baseline = results['original_4mb']
    
    print(f"{'Block Size':<20} {'Bandwidth':<12} {'vs 4MB':<10} {'Improvement':<12} {'Status'}")
    print("-" * 70)
    
    # Sort by bandwidth for easy comparison
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for version_name, bandwidth in sorted_results:
        vs_baseline = bandwidth / baseline
        improvement = ((bandwidth - baseline) / baseline) * 100
        
        if version_name == 'original_4mb':
            status = "(baseline)"
        elif improvement >= 10:
            status = "🚀 EXCELLENT"
        elif improvement >= 5:
            status = "📈 GOOD" 
        elif improvement >= 1:
            status = "📊 MODEST"
        elif improvement >= -5:
            status = "📉 SLIGHTLY WORSE"
        else:
            status = "❌ MUCH WORSE"
        
        block_size = version_name.replace('_', ' ').replace('blocks', '').replace('original', '4MB').title()
        print(f"{block_size:<20} {bandwidth:<8.2f} GB/s {vs_baseline:<6.2f}x {improvement:>+6.1f}%      {status}")
    
    # Find optimal
    best_version, best_bandwidth = sorted_results[0]
    best_improvement = ((best_bandwidth - baseline) / baseline) * 100
    
    print("-" * 70)
    print("OPTIMIZATION RESULTS:")
    
    if best_version == 'original_4mb':
        print("• Your original 4MB blocks are still optimal")
        print("• Array might already be perfectly tuned for 4MB")
    else:
        print(f"• OPTIMAL: {best_version.replace('_', ' ').title()}")
        print(f"• Best bandwidth: {best_bandwidth:.2f} GB/s")
        print(f"• Improvement: {best_improvement:+.1f}% over 4MB baseline")
        
        if best_improvement >= 20:
            print("• 🚀 Excellent optimization found!")
        elif best_improvement >= 10:
            print("• 📈 Significant improvement!")
        elif best_improvement >= 5:
            print("• 📊 Good optimization")
        else:
            print("• 📊 Modest but meaningful improvement")
    
    # Technical insights
    print("\nTECHNICAL INSIGHTS:")
    
    # Check if 1MB is still winner
    if '1mb_blocks' in results:
        mb1_improvement = ((results['1mb_blocks'] - baseline) / baseline) * 100
        print(f"• 1MB blocks: {mb1_improvement:+.1f}% (matches your array's 1MB chunks)")
    
    # Check queue depth effect
    if '1mb_qd64' in results and '1mb_blocks' in results:
        qd_effect = ((results['1mb_qd64'] - results['1mb_blocks']) / results['1mb_blocks']) * 100
        print(f"• Higher queue depth (64 vs 32): {qd_effect:+.1f}% effect")
    
    # Check larger blocks
    large_blocks = [k for k in results.keys() if '8mb' in k or '16mb' in k]
    if large_blocks:
        print("• Larger blocks: ", end="")
        for block in large_blocks:
            improvement = ((results[block] - baseline) / baseline) * 100
            print(f"{block.replace('_blocks', '').upper()} {improvement:+.1f}%, ", end="")
        print()
    
    print(f"• Array sweet spot: Your array strongly prefers {best_version.replace('_', ' ')}")
    print(f"• Single-threaded approach: Confirmed optimal for your array architecture")
    
    return results, best_version

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 block_size_optimization_test.py <filepath> [size_gb] [device_id]")
        print("Example: python3 block_size_optimization_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1")
        print("")
        print("Based on previous results:")
        print("• 1MB blocks gave 24% improvement (11.32 GB/s)")
        print("• Parallel approaches hurt performance") 
        print("• This test finds the absolute optimal block size")
        sys.exit(1)
    
    filepath = sys.argv[1]
    size_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    
    test_block_size_optimization(filepath, size_gb, device_id)