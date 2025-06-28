# simple_test.py
import ctypes
import torch
import os
import time

def test_both_versions(filepath, size_gb, device_id=1):
    """Test both original and enhanced versions"""
    
    # Load the shared library
    lib = ctypes.CDLL('./libenhanced_io.so')
    
    # Set up function signatures
    lib.read_to_gpu_io_uring.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.read_to_gpu_io_uring.restype = None
    
    lib.enhanced_read_to_gpu.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.enhanced_read_to_gpu.restype = None
    
    # Set CUDA device
    torch.cuda.set_device(device_id)
    
    # Calculate size in bytes
    size_bytes = int(size_gb * 1024 * 1024 * 1024)
    
    # Prepare filepath
    filepath_bytes = os.path.abspath(filepath).encode("utf-8")
    
    print(f"Testing {size_gb} GB transfer on GPU {device_id}")
    print(f"File: {filepath}")
    print("=" * 50)
    
    # Test original version
    print("\n1. Testing ORIGINAL version:")
    gpu_tensor1 = torch.empty(size_bytes, dtype=torch.uint8, device=f'cuda:{device_id}')
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    lib.read_to_gpu_io_uring(filepath_bytes, ctypes.c_void_p(gpu_tensor1.data_ptr()), size_bytes)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed1 = end - start
    bandwidth1 = size_gb / elapsed1
    print(f"Python timing: {elapsed1:.4f} sec -> {bandwidth1:.2f} GB/s")
    
    # Test enhanced version
    print("\n2. Testing ENHANCED version:")
    gpu_tensor2 = torch.empty(size_bytes, dtype=torch.uint8, device=f'cuda:{device_id}')
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    lib.enhanced_read_to_gpu(filepath_bytes, ctypes.c_void_p(gpu_tensor2.data_ptr()), size_bytes)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed2 = end - start
    bandwidth2 = size_gb / elapsed2
    print(f"Python timing: {elapsed2:.4f} sec -> {bandwidth2:.2f} GB/s")
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON:")
    print(f"Original:  {bandwidth1:.2f} GB/s")
    print(f"Enhanced:  {bandwidth2:.2f} GB/s")
    print(f"Speedup:   {bandwidth2/bandwidth1:.2f}x")
    print(f"Data verification: {torch.equal(gpu_tensor1, gpu_tensor2)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 simple_test.py <filepath> [size_gb] [device_id]")
        print("Example: python3 simple_test.py /mnt/kvcache/huge_test_4tb.bin 5.0 1")
        sys.exit(1)
    
    filepath = sys.argv[1]
    size_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    
    test_both_versions(filepath, size_gb, device_id)