# ultra_fast_test.py
import ctypes
import torch
import os
import time
import argparse

def benchmark_transfer(filepath, size_gb, device_id=0, config=None):
    """Benchmark the ultra-fast I/O to GPU transfer"""
    
    # Load the shared library
    lib = ctypes.CDLL('./libultra_fast_io.so')
    
    # Set up function signatures
    lib.ultra_fast_read_to_gpu.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.ultra_fast_read_to_gpu.restype = None
    
    lib.configurable_read_to_gpu.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.configurable_read_to_gpu.restype = None
    
    # Set CUDA device
    torch.cuda.set_device(device_id)
    
    # Calculate size in bytes
    size_bytes = int(size_gb * 1024 * 1024 * 1024)
    
    # Allocate GPU tensor
    print(f"Allocating {size_gb} GB on GPU {device_id}...")
    gpu_tensor = torch.empty(size_bytes, dtype=torch.uint8, device=f'cuda:{device_id}')
    
    # Prepare filepath
    filepath_bytes = os.path.abspath(filepath).encode("utf-8")
    
    print(f"Starting transfer from {filepath}")
    print(f"File size: {size_gb} GB")
    
    # Warm up GPU
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    if config is None:
        # Use default optimized settings
        lib.ultra_fast_read_to_gpu(filepath_bytes, ctypes.c_void_p(gpu_tensor.data_ptr()), size_bytes)
    else:
        # Use custom configuration
        queue_depth, block_size_mb, num_streams, num_rings = config
        lib.configurable_read_to_gpu(
            filepath_bytes, 
            ctypes.c_void_p(gpu_tensor.data_ptr()), 
            size_bytes,
            queue_depth, block_size_mb, num_streams, num_rings
        )
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = end - start
    bandwidth_gb_s = size_gb / elapsed
    
    print(f"\nPython measurement:")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print(f"Bandwidth: {bandwidth_gb_s:.2f} GB/s")
    print(f"GPU tensor sum (verification): {gpu_tensor.sum().item()}")
    
    return elapsed, bandwidth_gb_s

def run_parameter_sweep(filepath, size_gb, device_id=0):
    """Run parameter sweep to find optimal settings"""
    
    print("Running parameter sweep to find optimal settings...")
    
    # Parameter ranges to test
    configs = [
        # (queue_depth, block_size_mb, num_streams, num_rings)
        (64, 4, 4, 1),    # Conservative baseline
        (96, 8, 6, 2),    # Default optimized
        (128, 8, 8, 2),   # High parallelism
        (96, 16, 6, 2),   # Larger blocks
        (96, 8, 4, 4),    # More rings
        (128, 16, 8, 2),  # Maximum everything
        (64, 32, 4, 1),   # Very large blocks
    ]
    
    best_bandwidth = 0
    best_config = None
    
    for i, config in enumerate(configs):
        queue_depth, block_size_mb, num_streams, num_rings = config
        print(f"\nTest {i+1}/{len(configs)}: QD={queue_depth}, Block={block_size_mb}MB, "
              f"Streams={num_streams}, Rings={num_rings}")
        
        try:
            elapsed, bandwidth = benchmark_transfer(filepath, size_gb, device_id, config)
            
            if bandwidth > best_bandwidth:
                best_bandwidth = bandwidth
                best_config = config
                
        except Exception as e:
            print(f"Configuration failed: {e}")
            continue
    
    print(f"\nBest configuration:")
    print(f"Queue Depth: {best_config[0]}")
    print(f"Block Size: {best_config[1]} MB")
    print(f"Streams: {best_config[2]}")
    print(f"Rings: {best_config[3]}")
    print(f"Best Bandwidth: {best_bandwidth:.2f} GB/s")
    
    return best_config, best_bandwidth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ultra-fast I/O to GPU benchmark')
    parser.add_argument('filepath', help='Path to the file to read')
    parser.add_argument('--size', type=float, default=5.0, help='Size to read in GB (default: 5.0)')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID (default: 0)')
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--config', nargs=4, type=int, metavar=('QD', 'BLOCK_MB', 'STREAMS', 'RINGS'),
                       help='Custom configuration: queue_depth block_size_mb num_streams num_rings')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File {args.filepath} not found")
        exit(1)
    
    print("Ultra-Fast I/O to GPU Benchmark")
    print("=" * 40)
    
    if args.sweep:
        run_parameter_sweep(args.filepath, args.size, args.device)
    else:
        config = tuple(args.config) if args.config else None
        benchmark_transfer(args.filepath, args.size, args.device, config)

