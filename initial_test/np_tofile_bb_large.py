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
        # If you have sudo access, uncomment this:
        # os.system('sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"')
    except:
        pass

def test_large_files():
    """Test with much larger files to stress the RAID array"""
    
    # Test much larger files: 100MB to 4GB
    # Format: (batch_size, description)
    test_sizes = [
        (1600, "~100MB"),    # 1600 * 32 * 1024 * 2 bytes = ~104MB
        (3200, "~200MB"),    # ~209MB
        (6400, "~400MB"),    # ~419MB
        (12800, "~800MB"),   # ~838MB
        (25600, "~1.6GB"),   # ~1.67GB
        (51200, "~3.2GB"),   # ~3.35GB
    ]
    
    results = {}
    test_file = "/mnt/kvcache/large_test.bin"
    
    print("=== LARGE FILE STORAGE TEST ===")
    print("Testing files that should stress your RAID array...")
    
    for method_name, chunk_mb, description in [
        ("chunked_64mb", 64, "64MB chunks"),
        ("chunked_256mb", 256, "256MB chunks"),
    ]:
        
        print(f"\n{description}:")
        print("-" * 40)
        
        writes = []
        reads = []
        actual_sizes = []
        
        chunk_size = chunk_mb * 1024 * 1024
        
        for batch_size, size_desc in test_sizes:
            print(f"Testing {size_desc}...", end=" ", flush=True)
            
            try:
                # Generate data
                data = np.random.randn(batch_size, 32, 1024).astype(np.float16)
                actual_size_mb = data.nbytes / 1e6
                actual_sizes.append(actual_size_mb)
                
                # WRITE TEST
                start = time.time()
                with open(test_file, 'wb', buffering=0) as f:
                    data_bytes = data.tobytes()
                    for i in range(0, len(data_bytes), chunk_size):
                        chunk = data_bytes[i:i+chunk_size]
                        f.write(chunk)
                    os.fsync(f.fileno())
                end = time.time()
                
                write_time = end - start
                write_bw = data.nbytes / write_time / 1e9
                writes.append(write_bw)
                
                flush_caches()
                time.sleep(0.5)  # Brief pause
                
                # READ TEST
                start = time.time()
                with open(test_file, 'rb', buffering=0) as f:
                    chunks = []
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    data_bytes = b''.join(chunks)
                    data_read = np.frombuffer(data_bytes, dtype=np.float16)
                    # Force actual memory access
                    _ = data_read.sum()
                end = time.time()
                
                read_time = end - start
                read_bw = len(data_bytes) / read_time / 1e9
                reads.append(read_bw)
                
                print(f"Write: {write_bw:.1f} GB/s, Read: {read_bw:.1f} GB/s")
                
                # Clean up for next test
                try:
                    os.remove(test_file)
                except:
                    pass
                
            except Exception as e:
                print(f"Error: {e}")
                writes.append(0)
                reads.append(0)
                actual_sizes.append(batch_size * 32 * 1024 * 2 / 1e6)
        
        results[method_name] = {
            'sizes': actual_sizes,
            'writes': writes, 
            'reads': reads,
            'description': description
        }
    
    return results

def plot_results(results):
    """Plot the large file test results"""
    
    plt.figure(figsize=(15, 10))
    
    # Write performance
    plt.subplot(2, 1, 1)
    for method, data in results.items():
        plt.plot(data['sizes'], data['writes'], 
                label=f"{data['description']} - Write", 
                marker='o', linewidth=2, markersize=6)
    
    plt.xlabel('File Size (MB)')
    plt.ylabel('Write Bandwidth (GB/s)')
    plt.title('Large File Write Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Read performance  
    plt.subplot(2, 1, 2)
    for method, data in results.items():
        plt.plot(data['sizes'], data['reads'], 
                label=f"{data['description']} - Read", 
                marker='s', linewidth=2, markersize=6)
    
    plt.xlabel('File Size (MB)')
    plt.ylabel('Read Bandwidth (GB/s)')
    plt.title('Large File Read Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig("large_file_raid_test.png", dpi=200, bbox_inches='tight')
    plt.show()

def main():
    print("This test will create files up to 3.2GB to properly stress your RAID array.")
    print("Make sure you have enough free space on /mnt/kvcache\n")
    
    # Check available space
    statvfs = os.statvfs('/mnt/kvcache')
    free_gb = (statvfs.f_frsize * statvfs.f_bavail) / 1e9
    print(f"Available space: {free_gb:.1f} GB")
    
    if free_gb < 5:
        print("WARNING: Less than 5GB free space. Test may fail.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Run tests
    results = test_large_files()
    
    # Print summary
    print("\n" + "="*60)
    print("LARGE FILE PERFORMANCE SUMMARY")
    print("="*60)
    
    for method, data in results.items():
        max_write = max(data['writes']) if data['writes'] else 0
        max_read = max(data['reads']) if data['reads'] else 0
        print(f"{data['description']:15} - Max Write: {max_write:6.2f} GB/s, Max Read: {max_read:6.2f} GB/s")
    
    # Plot results
    plot_results(results)
    
    print(f"\nExpected performance for 11x PCIe4 NVMe RAID0:")
    print(f"- Sequential Write: 40-60+ GB/s")
    print(f"- Sequential Read:  50-70+ GB/s")
    
    if max(max(data['writes']) for data in results.values()) < 10:
        print(f"\n⚠️  Performance is MUCH lower than expected!")
        print(f"   Run the diagnostic commands to check RAID configuration.")

if __name__ == "__main__":
    main()