#!/usr/bin/env python3
"""
KV Cache Time-Based Transfer Benchmark - GDS Optimized
Tests how much data can be transferred from NVME storage to GPU devices
within specific time windows using PyTorch + Kvikio.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process, Queue, set_start_method
import psutil
import gc
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict
import json

# Only import non-kvikio libraries at module level
try:
    import torch
    import pynvml
except ImportError as e:
    print(f"Missing required libraries: {e}")
    print("Install with: pip install torch pynvml")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitor CPU and RAM usage during transfers."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.cpu_usage = []
        self.ram_usage = []
        self.timestamps = []
        from threading import Thread
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring system resources."""
        from threading import Thread
        self.monitoring = True
        self.cpu_usage = []
        self.ram_usage = []
        self.timestamps = []
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return average usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if self.cpu_usage and self.ram_usage:
            return {
                'avg_cpu_percent': np.mean(self.cpu_usage),
                'max_cpu_percent': np.max(self.cpu_usage),
                'avg_ram_percent': np.mean(self.ram_usage),
                'max_ram_percent': np.max(self.ram_usage),
                'avg_ram_gb': np.mean([r * psutil.virtual_memory().total / (100 * 1024**3) for r in self.ram_usage])
            }
        return {
            'avg_cpu_percent': 0, 'max_cpu_percent': 0,
            'avg_ram_percent': 0, 'max_ram_percent': 0, 'avg_ram_gb': 0
        }
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                self.cpu_usage.append(cpu_percent)
                self.ram_usage.append(ram_percent)
                self.timestamps.append(time.time())
                
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                break

def run_single_config_benchmark(test_file_path: str, time_windows: List[float], 
                               compat_mode: str, thread_count: int,
                               num_gpus: int, file_size: int, chunk_size: int,
                               result_queue: Queue):
    """
    Run benchmark for a single (compat_mode, thread_count) configuration in a separate process.
    This ensures both KVIKIO_COMPAT_MODE and KVIKIO_NTHREADS are set before Kvikio import.
    """
    # CRITICAL: Set environment variables BEFORE importing kvikio
    os.environ['KVIKIO_COMPAT_MODE'] = compat_mode
    os.environ['KVIKIO_NTHREADS'] = str(thread_count)
    logger.info(f"=== CONFIG: COMPAT_MODE={compat_mode}, NTHREADS={thread_count} ===")
    logger.info(f"Set KVIKIO_COMPAT_MODE={os.environ.get('KVIKIO_COMPAT_MODE', 'NOT_SET')}")
    logger.info(f"Set KVIKIO_NTHREADS={os.environ.get('KVIKIO_NTHREADS', 'NOT_SET')}")
    
    # Now import kvikio after setting environment variables
    try:
        import kvikio
        logger.info(f"Successfully imported kvikio with COMPAT_MODE={compat_mode}, NTHREADS={thread_count}")
    except ImportError as e:
        logger.error(f"Import failed for config {compat_mode}/{thread_count}: {e}")
        result_queue.put((compat_mode, thread_count, {}, {}, {}, {}))
        return
    
    # Initialize results storage for this configuration
    all_gpu_results = {}
    all_cpu_results = {}
    all_ram_results = {}
    all_gds_status = {}
    
    # Create a separate process for each GPU with this configuration
    gpu_queue = Queue()
    gpu_processes = []
    
    for gpu_id in range(num_gpus):
        p = Process(
            target=gpu_worker_process_single_config,
            args=(gpu_id, test_file_path, time_windows, 
                  compat_mode, thread_count, file_size, chunk_size, gpu_queue)
        )
        p.start()
        gpu_processes.append(p)
    
    # Wait for all GPU processes to complete
    for p in gpu_processes:
        p.join()
    
    # Collect results from GPU workers
    while not gpu_queue.empty():
        gpu_id, gpu_results, cpu_results, ram_results, gds_enabled = gpu_queue.get()
        all_gpu_results[gpu_id] = gpu_results
        all_cpu_results[gpu_id] = cpu_results
        all_ram_results[gpu_id] = ram_results
        all_gds_status[gpu_id] = gds_enabled
        
        logger.info(f"CONFIG {compat_mode}/{thread_count}T: Collected GPU {gpu_id} results, GDS={'ENABLED' if gds_enabled else 'DISABLED'}")
    
    # Send combined results back to main process
    result_queue.put((compat_mode, thread_count, all_gpu_results, all_cpu_results, all_ram_results, all_gds_status))
    logger.info(f"=== CONFIG {compat_mode}/{thread_count}T COMPLETED ===")

def gpu_worker_process_single_config(gpu_id: int, test_file_path: str, time_windows: List[float], 
                                    compat_mode: str, thread_count: int, file_size: int, 
                                    chunk_size: int, result_queue: Queue):
    """
    Worker process for testing a single GPU with a specific (COMPAT_MODE, NTHREADS) configuration.
    Tests how much data can be transferred within each time window.
    """
    # CRITICAL: Import kvikio in this process AFTER environment variables are set
    try:
        import kvikio
        logger.info(f"GPU {gpu_id} worker: imported kvikio with COMPAT_MODE={os.environ.get('KVIKIO_COMPAT_MODE', 'NOT_SET')}, NTHREADS={os.environ.get('KVIKIO_NTHREADS', 'NOT_SET')}")
    except ImportError as e:
        logger.error(f"GPU {gpu_id} worker: kvikio import failed: {e}")
        result_queue.put((gpu_id, {}, {}, {}, False))
        return
    
    # Isolate this GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Initialize PyTorch CUDA context
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    
    logger.info(f"GPU {gpu_id} worker started for COMPAT_MODE={compat_mode}, NTHREADS={thread_count}")
    
    # Results storage
    gpu_results = []
    cpu_usage_results = {}
    ram_usage_results = {}
    
    # Test GDS capability for this configuration
    gds_enabled = False
    try:
        # Multiple attempts with different buffer sizes
        for test_size in [1024, 4096, 16384]:  # 1KB, 4KB, 16KB
            test_buffer = torch.empty(test_size // 4, dtype=torch.float32, device=device)
            try:
                with kvikio.CuFile(test_file_path, "rb") as f:
                    future = f.pread(test_buffer, test_size, 0)
                    bytes_read = future.get()
                    if bytes_read == test_size:
                        gds_enabled = True
                        logger.info(f"SUCCESS: GPU {gpu_id} GDS ENABLED (COMPAT_MODE={compat_mode}, NTHREADS={thread_count}, test_size={test_size})")
                        break
                    else:
                        logger.warning(f"GPU {gpu_id}: GDS test incomplete read {bytes_read}/{test_size}")
            except Exception as test_error:
                logger.debug(f"GPU {gpu_id}: GDS test failed for size {test_size}: {test_error}")
                continue
            finally:
                del test_buffer
                
        if not gds_enabled:
            logger.warning(f"FAILED: GPU {gpu_id} GDS test failed for all sizes (COMPAT_MODE={compat_mode}, NTHREADS={thread_count})")
            
    except Exception as e:
        logger.error(f"GPU {gpu_id}: GDS test exception (COMPAT_MODE={compat_mode}, NTHREADS={thread_count}): {e}")
    
    # Allocate GPU buffer for transfers (using chunk_size)
    try:
        gpu_buffer = torch.empty(chunk_size // 4, dtype=torch.float32, device=device)
    except Exception as e:
        logger.error(f"GPU {gpu_id}: Failed to allocate buffer of size {chunk_size}: {e}")
        result_queue.put((gpu_id, [], {}, {}, gds_enabled))
        return
    
    # Test each time window
    for time_window in time_windows:
        time_str = _format_time(time_window)
        
        try:
            # Start system monitoring
            monitor = SystemMonitor(sampling_interval=0.05)
            monitor.start_monitoring()
            
            # Performance measurement - Run 10 iterations and collect all measurements
            bytes_transferred_list = []
            num_iterations = 10
            align = 1024 * 1024  # 1MB alignment
            
            # Warmup - quick transfers to prime the system
            for warmup_iter in range(2):
                try:
                    start_time = time.perf_counter()
                    total_bytes = 0
                    offset = 0
                    
                    while (time.perf_counter() - start_time) < min(time_window, 0.01):  # Limit warmup time
                        if gds_enabled:
                            with kvikio.CuFile(test_file_path, "rb") as f:
                                future = f.pread(gpu_buffer, chunk_size, offset)
                                bytes_read = future.get()
                                total_bytes += bytes_read
                        else:
                            # Host fallback
                            with open(test_file_path, "rb") as f:
                                f.seek(offset)
                                data = f.read(chunk_size)
                                bytes_read = len(data)
                                if bytes_read > 0:
                                    host_array = np.frombuffer(data, dtype=np.uint8)
                                    host_tensor = torch.from_numpy(host_array).float()
                                    copy_size = min(len(host_tensor), len(gpu_buffer))
                                    gpu_buffer[:copy_size] = host_tensor[:copy_size].to(device)
                                    total_bytes += bytes_read
                        
                        torch.cuda.synchronize()
                        offset = (offset + chunk_size) % (file_size - chunk_size)
                        
                        if bytes_read < chunk_size:
                            break
                            
                except Exception:
                    continue
            
            # Actual measurements - measure bytes transferred in each time window
            for i in range(num_iterations):
                try:
                    # Random starting offset with 1MB alignment
                    max_offset = file_size - chunk_size
                    start_offset = np.random.randint(0, max_offset // align) * align
                    current_offset = start_offset
                    
                    start_time = time.perf_counter()
                    total_bytes = 0
                    chunks_read = 0
                    
                    # Read as much data as possible within the time window
                    while True:
                        current_time = time.perf_counter()
                        elapsed = current_time - start_time
                        
                        if elapsed >= time_window:
                            break
                        
                        try:
                            if gds_enabled:
                                # TRUE GDS PATH
                                with kvikio.CuFile(test_file_path, "rb") as f:
                                    future = f.pread(gpu_buffer, chunk_size, current_offset)
                                    bytes_read = future.get()
                            else:
                                # Host-mediated fallback
                                with open(test_file_path, "rb") as f:
                                    f.seek(current_offset)
                                    data = f.read(chunk_size)
                                    bytes_read = len(data)
                                    if bytes_read > 0:
                                        host_array = np.frombuffer(data, dtype=np.uint8)
                                        host_tensor = torch.from_numpy(host_array).float()
                                        copy_size = min(len(host_tensor), len(gpu_buffer))
                                        gpu_buffer[:copy_size] = host_tensor[:copy_size].to(device)
                            
                            torch.cuda.synchronize()
                            
                            if bytes_read > 0:
                                total_bytes += bytes_read
                                chunks_read += 1
                                # Move to next chunk location
                                current_offset = (current_offset + chunk_size) % (file_size - chunk_size)
                            else:
                                break
                                
                        except Exception as e:
                            logger.debug(f"GPU {gpu_id}: Read error during iteration {i+1}: {e}")
                            break
                    
                    # Record total bytes transferred in this time window
                    if total_bytes > 0:
                        bytes_transferred_list.append(total_bytes)
                        logger.debug(f"GPU {gpu_id}: Window {time_str}, iter {i+1}: {total_bytes/(1024**3):.3f} GB in {elapsed:.6f}s ({chunks_read} chunks)")
                    
                except Exception as e:
                    logger.error(f"GPU {gpu_id}: Transfer iteration {i+1} failed: {e}")
                    continue
            
            # Stop monitoring
            system_stats = monitor.stop_monitoring()
            
            # Calculate statistics
            if bytes_transferred_list:
                bytes_array = np.array(bytes_transferred_list)
                bandwidth_array = bytes_array / (time_window * 1024**3)  # Convert to GB/s
                
                transfer_stats = {
                    'bytes_measurements': bytes_transferred_list,
                    'bandwidth_measurements': bandwidth_array.tolist(),
                    'bytes_mean': float(np.mean(bytes_array)),
                    'bytes_std': float(np.std(bytes_array)),
                    'bytes_min': float(np.min(bytes_array)),
                    'bytes_max': float(np.max(bytes_array)),
                    'bandwidth_mean': float(np.mean(bandwidth_array)),
                    'bandwidth_std': float(np.std(bandwidth_array)),
                    'bandwidth_min': float(np.min(bandwidth_array)),
                    'bandwidth_max': float(np.max(bandwidth_array)),
                    'count': len(bytes_transferred_list),
                    'time_window': time_window
                }
                
                mode_str = "GDS" if gds_enabled else "Host"
                logger.info(f"GPU {gpu_id}: {time_str}, {thread_count}T, {compat_mode}: "
                          f"{transfer_stats['bytes_mean']/(1024**3):.3f}±{transfer_stats['bytes_std']/(1024**3):.3f} GB "
                          f"({transfer_stats['bandwidth_mean']:.2f}±{transfer_stats['bandwidth_std']:.2f} GB/s) "
                          f"[{mode_str}] ({transfer_stats['count']}/{num_iterations} successful)")
                gpu_results.append(transfer_stats)
            else:
                logger.error(f"GPU {gpu_id}: No successful transfers for {time_str}")
                gpu_results.append({
                    'bytes_measurements': [],
                    'bandwidth_measurements': [],
                    'bytes_mean': 0.0,
                    'bytes_std': 0.0,
                    'bytes_min': 0.0,
                    'bytes_max': 0.0,
                    'bandwidth_mean': 0.0,
                    'bandwidth_std': 0.0,
                    'bandwidth_min': 0.0,
                    'bandwidth_max': 0.0,
                    'count': 0,
                    'time_window': time_window
                })
            
            # Store system stats
            cpu_usage_results[time_window] = {
                'avg': system_stats.get('avg_cpu_percent', 0),
                'max': system_stats.get('max_cpu_percent', 0)
            }
            ram_usage_results[time_window] = {
                'avg_percent': system_stats.get('avg_ram_percent', 0),
                'max_percent': system_stats.get('max_ram_percent', 0),
                'avg_gb': system_stats.get('avg_ram_gb', 0)
            }
            
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Test failed for {time_str}: {e}")
            gpu_results.append({
                'bytes_measurements': [],
                'bandwidth_measurements': [],
                'bytes_mean': 0.0,
                'bytes_std': 0.0,
                'bytes_min': 0.0,
                'bytes_max': 0.0,
                'bandwidth_mean': 0.0,
                'bandwidth_std': 0.0,
                'bandwidth_min': 0.0,
                'bandwidth_max': 0.0,
                'count': 0,
                'time_window': time_window
            })
            
            cpu_usage_results[time_window] = {'avg': 0, 'max': 0}
            ram_usage_results[time_window] = {
                'avg_percent': 0, 'max_percent': 0, 'avg_gb': 0
            }
    
    # Cleanup
    del gpu_buffer
    torch.cuda.empty_cache()
    
    result_queue.put((gpu_id, gpu_results, cpu_usage_results, ram_usage_results, gds_enabled))
    logger.info(f"GPU {gpu_id}: Worker process completed for COMPAT_MODE={compat_mode}, NTHREADS={thread_count}")

def _format_time(time_seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if time_seconds >= 1:
        return f"{time_seconds:.1f}s"
    elif time_seconds >= 0.001:
        return f"{time_seconds * 1000:.1f}ms"
    else:
        return f"{time_seconds * 1000000:.0f}μs"

def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f}GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f}KB"
    else:
        return f"{size_bytes}B"

class KVCacheTimeBandwidthTester:
    def __init__(self, test_file_path: str = "/mnt/kvcache/huge_test_4tb.bin", 
                 chunk_size: int = 256 * 1024 * 1024):  # 256MB chunks by default
        """
        Initialize the time-based bandwidth tester with GDS optimization.
        
        Args:
            test_file_path: Path to test file
            chunk_size: Size of each read chunk in bytes (default: 256MB)
        """
        self.test_file_path = test_file_path
        self.chunk_size = chunk_size
        
        # Initialize NVML for GPU detection
        pynvml.nvmlInit()
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        
        # Time windows to test (0.1ms to 1s)
        self.time_windows = [
            0.0001,   # 0.1ms
            0.0005,   # 0.5ms
            0.001,    # 1ms
            0.005,    # 5ms
            0.01,     # 10ms
            0.05,     # 50ms
            0.1,      # 100ms
            0.5,      # 500ms
            1.0       # 1s
        ]
        
        # Thread counts to test
        self.thread_counts = [1, 2, 4, 8, 16, 32, 64]
        
        # KVIKIO compatibility modes to test
        self.compat_modes = ['OFF', 'ON']
        
        # Verify test file exists
        if not os.path.exists(self.test_file_path):
            raise FileNotFoundError(f"Test file not found: {self.test_file_path}")
        
        self.file_size = os.path.getsize(self.test_file_path)
        logger.info(f"Test file size: {self.file_size / (1024**4):.2f} TB")
        logger.info(f"Chunk size: {self.chunk_size / (1024**2):.2f} MB")
        
        # Results structure
        self.results = {
            'time_windows': self.time_windows,
            'thread_counts': self.thread_counts,
            'compat_modes': self.compat_modes,
            'chunk_size': self.chunk_size,
            'gpu_results': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),  # [compat_mode][thread_count][gpu_id]
            'aggregate_results': defaultdict(lambda: defaultdict(list)),  # [compat_mode][thread_count]
            'cpu_usage': defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),  # [compat_mode][thread_count][time_window]
            'ram_usage': defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),  # [compat_mode][thread_count][time_window]
            'system_info': self._get_system_info(),
            'gds_status': {}  # [gpu_id][compat_mode]
        }
        
        # Log GPU info
        logger.info(f"Detected {self.num_gpus} GPUs")
        for i in range(self.num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            logger.info(f"GPU {i}: {name}")

    def _get_system_info(self) -> Dict:
        """Get system information for documentation."""
        return {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'num_gpus': self.num_gpus
        }

    def run_benchmark(self) -> Dict:
        """
        Run the complete time-based bandwidth benchmark using multiprocessing.
        Tests each (KVIKIO_COMPAT_MODE, KVIKIO_NTHREADS) combination sequentially.
        """
        logger.info("Starting KV Cache time-based bandwidth benchmark")
        logger.info(f"Time windows: {[_format_time(t) for t in self.time_windows]}")
        logger.info(f"Thread counts: {self.thread_counts}")
        logger.info(f"KVIKIO compatibility modes: {self.compat_modes}")
        
        # Use multiprocessing for parallel testing
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        total_configs = len(self.compat_modes) * len(self.thread_counts)
        logger.info(f"Testing {total_configs} configurations sequentially...")
        
        start_time = time.time()
        config_num = 0
        
        # Run configurations sequentially to avoid GPU memory conflicts
        for compat_mode in self.compat_modes:
            for thread_count in self.thread_counts:
                config_num += 1
                logger.info(f"=== RUNNING CONFIG {config_num}/{total_configs}: COMPAT_MODE={compat_mode}, NTHREADS={thread_count} ===")
                
                result_queue = Queue()
                
                # Start single configuration process
                p = Process(
                    target=run_single_config_benchmark,
                    args=(self.test_file_path, self.time_windows, 
                          compat_mode, thread_count, self.num_gpus, 
                          self.file_size, self.chunk_size, result_queue)
                )
                p.start()
                
                # Wait for this configuration to complete before starting next one
                p.join()
                
                # Collect results from this configuration
                if not result_queue.empty():
                    compat_mode_result, thread_count_result, all_gpu_results, all_cpu_results, all_ram_results, all_gds_status = result_queue.get()
                    
                    logger.info(f"Processing results from COMPAT_MODE={compat_mode_result}, NTHREADS={thread_count_result}")
                    
                    # Store individual GPU results
                    for gpu_id in all_gpu_results:
                        gpu_data = all_gpu_results[gpu_id]
                        if compat_mode_result not in self.results['gpu_results']:
                            self.results['gpu_results'][compat_mode_result] = defaultdict(dict)
                        if thread_count_result not in self.results['gpu_results'][compat_mode_result]:
                            self.results['gpu_results'][compat_mode_result][thread_count_result] = {}
                        
                        self.results['gpu_results'][compat_mode_result][thread_count_result][gpu_id] = gpu_data
                        
                        measurements = len(gpu_data)
                        if measurements > 0:
                            max_bytes = max(item.get('bytes_mean', 0) for item in gpu_data)
                            logger.info(f"  Stored GPU {gpu_id} {compat_mode_result}/{thread_count_result}T: "
                                       f"{measurements} time windows, max={max_bytes/(1024**3):.3f} GB")
                    
                    # Store system usage results  
                    for gpu_id in all_cpu_results:
                        cpu_data = all_cpu_results[gpu_id]
                        if compat_mode_result not in self.results['cpu_usage']:
                            self.results['cpu_usage'][compat_mode_result] = defaultdict(dict)
                        self.results['cpu_usage'][compat_mode_result][thread_count_result].update(cpu_data)
                    
                    for gpu_id in all_ram_results:
                        ram_data = all_ram_results[gpu_id]
                        if compat_mode_result not in self.results['ram_usage']:
                            self.results['ram_usage'][compat_mode_result] = defaultdict(dict)
                        self.results['ram_usage'][compat_mode_result][thread_count_result].update(ram_data)
                    
                    # Store GDS status
                    for gpu_id in all_gds_status:
                        if gpu_id not in self.results['gds_status']:
                            self.results['gds_status'][gpu_id] = {}
                        self.results['gds_status'][gpu_id][compat_mode_result] = all_gds_status[gpu_id]
                        logger.info(f"GPU {gpu_id} COMPAT_MODE={compat_mode_result} GDS: {'ENABLED' if all_gds_status[gpu_id] else 'DISABLED'}")
                
                # Log progress
                elapsed_time = time.time() - start_time
                configs_remaining = total_configs - config_num
                if config_num > 0:
                    avg_time_per_config = elapsed_time / config_num
                    estimated_remaining = avg_time_per_config * configs_remaining
                    logger.info(f"Progress: {config_num}/{total_configs} configs completed. "
                               f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {estimated_remaining:.1f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.1f} seconds")
        
        # Calculate aggregate statistics for each configuration
        for compat_mode in self.compat_modes:
            if compat_mode not in self.results['aggregate_results']:
                self.results['aggregate_results'][compat_mode] = defaultdict(list)
                
            for thread_count in self.thread_counts:
                aggregate_stats = []
                for i in range(len(self.time_windows)):
                    # Collect all measurements for this time window across all GPUs
                    all_bytes = []
                    all_bandwidths = []
                    for gpu_id in range(self.num_gpus):
                        if (compat_mode in self.results['gpu_results'] and
                            thread_count in self.results['gpu_results'][compat_mode] and
                            gpu_id in self.results['gpu_results'][compat_mode][thread_count] and 
                            i < len(self.results['gpu_results'][compat_mode][thread_count][gpu_id])):
                            
                            gpu_stats = self.results['gpu_results'][compat_mode][thread_count][gpu_id][i]
                            if 'bytes_measurements' in gpu_stats:
                                # Sum bytes across GPUs for aggregate
                                all_bytes.extend(gpu_stats['bytes_measurements'])
                                all_bandwidths.extend(gpu_stats['bandwidth_measurements'])
                    
                    # Calculate aggregate statistics
                    if all_bytes:
                        # For aggregate, we sum the bytes from all GPUs
                        total_bytes_per_iter = []
                        num_measurements = len(all_bytes) // self.num_gpus
                        for j in range(num_measurements):
                            iter_sum = sum(all_bytes[j::num_measurements])
                            total_bytes_per_iter.append(iter_sum)
                        
                        if total_bytes_per_iter:
                            bytes_array = np.array(total_bytes_per_iter)
                            time_window = self.time_windows[i]
                            bandwidth_array = bytes_array / (time_window * 1024**3)
                            
                            aggregate_stat = {
                                'bytes_mean': float(np.mean(bytes_array)),
                                'bytes_std': float(np.std(bytes_array)),
                                'bytes_min': float(np.min(bytes_array)),
                                'bytes_max': float(np.max(bytes_array)),
                                'bandwidth_mean': float(np.mean(bandwidth_array)),
                                'bandwidth_std': float(np.std(bandwidth_array)),
                                'bandwidth_min': float(np.min(bandwidth_array)),
                                'bandwidth_max': float(np.max(bandwidth_array)),
                                'count': len(total_bytes_per_iter),
                                'time_window': time_window
                            }
                        else:
                            aggregate_stat = {
                                'bytes_mean': 0.0, 'bytes_std': 0.0, 'bytes_min': 0.0, 'bytes_max': 0.0,
                                'bandwidth_mean': 0.0, 'bandwidth_std': 0.0, 
                                'bandwidth_min': 0.0, 'bandwidth_max': 0.0,
                                'count': 0, 'time_window': self.time_windows[i]
                            }
                        
                        aggregate_stats.append(aggregate_stat)
                    else:
                        aggregate_stats.append({
                            'bytes_mean': 0.0, 'bytes_std': 0.0, 'bytes_min': 0.0, 'bytes_max': 0.0,
                            'bandwidth_mean': 0.0, 'bandwidth_std': 0.0, 
                            'bandwidth_min': 0.0, 'bandwidth_max': 0.0,
                            'count': 0, 'time_window': self.time_windows[i]
                        })
                
                self.results['aggregate_results'][compat_mode][thread_count] = aggregate_stats
        
        logger.info("Benchmark data collection completed")
        return self.results

    def visualize_results(self, save_path: str = "kv_cache_time_bandwidth_results.png"):
        """
        Create comprehensive visualization of time-based benchmark results.
        """
        # Convert time windows to milliseconds for plotting
        time_windows_ms = [t * 1000 for t in self.time_windows]
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(24, 20))
        
        # Set style
        sns.set_style("whitegrid")
        colors_compat = {'OFF': 'blue', 'ON': 'red'}
        colors_threads = sns.color_palette("husl", len(self.thread_counts))
        
        # 1. Data Transferred vs Time Window - COMPAT_MODE Comparison
        ax1 = plt.subplot(4, 3, 1)
        for compat_mode in self.compat_modes:
            best_bytes = []
            best_stds = []
            for i, time_window in enumerate(self.time_windows):
                max_bytes = 0
                best_std = 0
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_results'][compat_mode][thread_count] and 
                        i < len(self.results['aggregate_results'][compat_mode][thread_count])):
                        
                        item = self.results['aggregate_results'][compat_mode][thread_count][i]
                        if item['bytes_mean'] > max_bytes:
                            max_bytes = item['bytes_mean']
                            best_std = item['bytes_std']
                
                best_bytes.append(max_bytes / (1024**3))  # Convert to GB
                best_stds.append(best_std / (1024**3))
            
            # Plot with error bars
            ax1.errorbar(time_windows_ms, best_bytes, yerr=best_stds, 
                        label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode], 
                        linewidth=2, markersize=6, marker='o', capsize=3)
        
        ax1.set_title('Peak Data Transfer by Time Window', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Window (ms)')
        ax1.set_ylabel('Data Transferred (GB)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Effective Bandwidth vs Time Window
        ax2 = plt.subplot(4, 3, 2)
        for compat_mode in self.compat_modes:
            best_bw = []
            best_stds = []
            for i, time_window in enumerate(self.time_windows):
                max_bw = 0
                best_std = 0
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_results'][compat_mode][thread_count] and 
                        i < len(self.results['aggregate_results'][compat_mode][thread_count])):
                        
                        item = self.results['aggregate_results'][compat_mode][thread_count][i]
                        if item['bandwidth_mean'] > max_bw:
                            max_bw = item['bandwidth_mean']
                            best_std = item['bandwidth_std']
                
                best_bw.append(max_bw)
                best_stds.append(best_std)
            
            ax2.errorbar(time_windows_ms, best_bw, yerr=best_stds,
                        label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode],
                        linewidth=2, markersize=6, marker='o', capsize=3)
        
        ax2.set_title('Effective Bandwidth by Time Window', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Window (ms)')
        ax2.set_ylabel('Bandwidth (GB/s)')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Individual GPU Performance for Best COMPAT_MODE (1s window)
        ax3 = plt.subplot(4, 3, 3)
        # Find best compat mode based on 1s window performance
        best_compat_mode = 'OFF'
        max_bytes_1s = 0
        target_window_idx = len(self.time_windows) - 1  # 1s window
        
        for compat_mode in self.compat_modes:
            for thread_count in self.thread_counts:
                if (self.results['aggregate_results'][compat_mode][thread_count] and
                    target_window_idx < len(self.results['aggregate_results'][compat_mode][thread_count])):
                    
                    bytes_val = self.results['aggregate_results'][compat_mode][thread_count][target_window_idx]['bytes_mean']
                    if bytes_val > max_bytes_1s:
                        max_bytes_1s = bytes_val
                        best_compat_mode = compat_mode
        
        # Find best thread count for best compat mode
        best_thread_count = 1
        max_bytes = 0
        for thread_count in self.thread_counts:
            if (self.results['aggregate_results'][best_compat_mode][thread_count] and
                target_window_idx < len(self.results['aggregate_results'][best_compat_mode][thread_count])):
                
                bytes_val = self.results['aggregate_results'][best_compat_mode][thread_count][target_window_idx]['bytes_mean']
                if bytes_val > max_bytes:
                    max_bytes = bytes_val
                    best_thread_count = thread_count
        
        for gpu_id in range(self.num_gpus):
            if gpu_id in self.results['gpu_results'][best_compat_mode][best_thread_count]:
                gpu_data = self.results['gpu_results'][best_compat_mode][best_thread_count][gpu_id]
                bytes_means = [item['bytes_mean'] / (1024**3) for item in gpu_data]
                bytes_stds = [item['bytes_std'] / (1024**3) for item in gpu_data]
                
                if any(m > 0 for m in bytes_means):
                    ax3.errorbar(time_windows_ms, bytes_means, yerr=bytes_stds,
                               label=f'GPU {gpu_id}', linewidth=2, markersize=4, 
                               marker='o', capsize=2)
        
        ax3.set_title(f'Individual GPU Performance\n(COMPAT_MODE={best_compat_mode}, {best_thread_count} threads)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Window (ms)')
        ax3.set_ylabel('Data Transferred (GB)')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        if ax3.get_legend_handles_labels()[0]:
            ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Thread Scaling for 100ms window
        ax4 = plt.subplot(4, 3, 4)
        target_window = 0.1  # 100ms
        target_idx = self.time_windows.index(target_window) if target_window in self.time_windows else 6
        
        for compat_mode in self.compat_modes:
            thread_bytes = []
            thread_stds = []
            for thread_count in self.thread_counts:
                if (self.results['aggregate_results'][compat_mode][thread_count] and 
                    target_idx < len(self.results['aggregate_results'][compat_mode][thread_count])):
                    
                    item = self.results['aggregate_results'][compat_mode][thread_count][target_idx]
                    thread_bytes.append(item['bytes_mean'] / (1024**3))
                    thread_stds.append(item['bytes_std'] / (1024**3))
                else:
                    thread_bytes.append(0)
                    thread_stds.append(0)
            
            ax4.errorbar(self.thread_counts, thread_bytes, yerr=thread_stds,
                        label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode],
                        linewidth=2, markersize=6, marker='o', capsize=3)
        
        ax4.set_title(f'Threading Performance (100ms window)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Thread Count')
        ax4.set_ylabel('Data Transferred (GB)')
        ax4.set_xscale('log', base=2)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. CPU Usage by Time Window
        ax5 = plt.subplot(4, 3, 5)
        for compat_mode in self.compat_modes:
            cpu_by_window = []
            for time_window in self.time_windows:
                cpu_vals = []
                for thread_count in self.thread_counts:
                    if time_window in self.results['cpu_usage'][compat_mode][thread_count]:
                        cpu_vals.append(self.results['cpu_usage'][compat_mode][thread_count][time_window]['avg'])
                cpu_by_window.append(np.mean(cpu_vals) if cpu_vals else 0)
            
            ax5.plot(time_windows_ms, cpu_by_window, 'o-', 
                    label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode],
                    linewidth=2, markersize=4)
        
        ax5.set_title('Average CPU Usage by Time Window', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time Window (ms)')
        ax5.set_ylabel('CPU Usage (%)')
        ax5.set_xscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Data Transfer Efficiency (GB per % CPU)
        ax6 = plt.subplot(4, 3, 6)
        for compat_mode in self.compat_modes:
            efficiency = []
            for i, time_window in enumerate(self.time_windows):
                max_bytes = 0
                cpu_for_max = 0
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_results'][compat_mode][thread_count] and 
                        i < len(self.results['aggregate_results'][compat_mode][thread_count]) and
                        time_window in self.results['cpu_usage'][compat_mode][thread_count]):
                        
                        bytes_mean = self.results['aggregate_results'][compat_mode][thread_count][i]['bytes_mean']
                        if bytes_mean > max_bytes:
                            max_bytes = bytes_mean
                            cpu_for_max = self.results['cpu_usage'][compat_mode][thread_count][time_window]['avg']
                
                if cpu_for_max > 0:
                    efficiency.append((max_bytes / (1024**3)) / cpu_for_max)
                else:
                    efficiency.append(0)
            
            ax6.plot(time_windows_ms, efficiency, 'o-',
                    label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode],
                    linewidth=2, markersize=4)
        
        ax6.set_title('Transfer Efficiency (GB per % CPU)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time Window (ms)')
        ax6.set_ylabel('GB / CPU%')
        ax6.set_xscale('log')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Statistical Distribution of Measurements
        ax7 = plt.subplot(4, 3, 7)
        for compat_mode in self.compat_modes:
            all_cvs = []
            all_windows = []
            for i, time_window in enumerate(self.time_windows):
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_results'][compat_mode][thread_count] and 
                        i < len(self.results['aggregate_results'][compat_mode][thread_count])):
                        
                        item = self.results['aggregate_results'][compat_mode][thread_count][i]
                        if item['bytes_mean'] > 0:
                            cv = (item['bytes_std'] / item['bytes_mean']) * 100
                            all_cvs.append(cv)
                            all_windows.append(time_window * 1000)
            
            if all_cvs:
                ax7.scatter(all_windows, all_cvs, label=f'COMPAT_MODE={compat_mode}', 
                           color=colors_compat[compat_mode], alpha=0.6, s=30)
        
        ax7.set_title('Measurement Variability', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Time Window (ms)')
        ax7.set_ylabel('Coefficient of Variation (%)')
        ax7.set_xscale('log')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Best Configuration Heatmap
        ax8 = plt.subplot(4, 3, 8)
        best_config_matrix = np.zeros((len(self.compat_modes), len(self.thread_counts)))
        for i, compat_mode in enumerate(self.compat_modes):
            for j, thread_count in enumerate(self.thread_counts):
                if self.results['aggregate_results'][compat_mode][thread_count]:
                    # Use 1s window performance for comparison
                    if target_window_idx < len(self.results['aggregate_results'][compat_mode][thread_count]):
                        bytes_val = self.results['aggregate_results'][compat_mode][thread_count][target_window_idx]['bytes_mean']
                        best_config_matrix[i, j] = bytes_val / (1024**3)  # Convert to GB
        
        im8 = ax8.imshow(best_config_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax8.set_title('Data Transfer in 1s by Configuration', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Thread Count')
        ax8.set_ylabel('COMPAT_MODE')
        ax8.set_xticks(range(len(self.thread_counts)))
        ax8.set_xticklabels(self.thread_counts)
        ax8.set_yticks(range(len(self.compat_modes)))
        ax8.set_yticklabels(self.compat_modes)
        plt.colorbar(im8, ax=ax8, shrink=0.6, label='Data (GB)')
        
        # 9. Performance Summary Table
        ax9 = plt.subplot(4, 3, 9)
        ax9.axis('off')
        
        # Create summary table
        summary_data = []
        for compat_mode in self.compat_modes:
            max_gb_100ms = 0
            max_gb_1s = 0
            max_bw = 0
            
            # Find max for 100ms window
            if 0.1 in self.time_windows:
                idx_100ms = self.time_windows.index(0.1)
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_results'][compat_mode][thread_count] and
                        idx_100ms < len(self.results['aggregate_results'][compat_mode][thread_count])):
                        gb = self.results['aggregate_results'][compat_mode][thread_count][idx_100ms]['bytes_mean'] / (1024**3)
                        max_gb_100ms = max(max_gb_100ms, gb)
            
            # Find max for 1s window
            if 1.0 in self.time_windows:
                idx_1s = self.time_windows.index(1.0)
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_results'][compat_mode][thread_count] and
                        idx_1s < len(self.results['aggregate_results'][compat_mode][thread_count])):
                        gb = self.results['aggregate_results'][compat_mode][thread_count][idx_1s]['bytes_mean'] / (1024**3)
                        bw = self.results['aggregate_results'][compat_mode][thread_count][idx_1s]['bandwidth_mean']
                        max_gb_1s = max(max_gb_1s, gb)
                        max_bw = max(max_bw, bw)
            
            gds_count = sum(1 for gpu_id in range(self.num_gpus) 
                           if self.results['gds_status'].get(gpu_id, {}).get(compat_mode, False))
            
            summary_data.append([
                f'{compat_mode}',
                f'{max_gb_100ms:.2f}',
                f'{max_gb_1s:.2f}',
                f'{max_bw:.2f}',
                f'{gds_count}/{self.num_gpus}'
            ])
        
        table = ax9.table(cellText=summary_data,
                         colLabels=['Mode', '100ms (GB)', '1s (GB)', 'BW (GB/s)', 'GDS GPUs'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax9.set_title('Performance Summary', fontsize=12, fontweight='bold')
        
        # 10. Threading Scaling for Different Time Windows
        ax10 = plt.subplot(4, 3, 10)
        selected_windows = [0.001, 0.01, 0.1, 1.0]  # 1ms, 10ms, 100ms, 1s
        
        for window in selected_windows:
            if window in self.time_windows:
                window_idx = self.time_windows.index(window)
                thread_gb = []
                for thread_count in self.thread_counts:
                    max_gb = 0
                    for compat_mode in self.compat_modes:
                        if (self.results['aggregate_results'][compat_mode][thread_count] and 
                            window_idx < len(self.results['aggregate_results'][compat_mode][thread_count])):
                            gb = self.results['aggregate_results'][compat_mode][thread_count][window_idx]['bytes_mean'] / (1024**3)
                            max_gb = max(max_gb, gb)
                    thread_gb.append(max_gb)
                
                ax10.plot(self.thread_counts, thread_gb, 'o-',
                         label=f'{_format_time(window)}', linewidth=2, markersize=4)
        
        ax10.set_title('Threading Scaling by Time Window', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Thread Count')
        ax10.set_ylabel('Data Transferred (GB)')
        ax10.set_xscale('log', base=2)
        ax10.set_yscale('log')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Results saved to {save_path}")
        plt.show()

    def print_summary(self):
        """Print a comprehensive summary of time-based benchmark results."""
        print("\n" + "="*80)
        print("KV CACHE TIME-BASED BANDWIDTH BENCHMARK SUMMARY")
        print("="*80)
        
        # System info
        info = self.results['system_info']
        print(f"System: AMD Threadripper Pro")
        print(f"CPUs: {info['cpu_count']} physical, {info['cpu_count_logical']} logical")
        print(f"Memory: {info['memory_total'] / (1024**3):.1f} GB")
        print(f"GPUs: {info['num_gpus']}")
        print(f"Chunk size: {self.chunk_size / (1024**2):.1f} MB")
        print()
        
        # GDS status
        print("GDS STATUS BY COMPATIBILITY MODE:")
        for compat_mode in self.compat_modes:
            gds_gpus = sum(1 for gpu_id in range(info['num_gpus']) 
                          if self.results['gds_status'].get(gpu_id, {}).get(compat_mode, False))
            print(f"KVIKIO_COMPAT_MODE={compat_mode}: {gds_gpus}/{info['num_gpus']} GPUs with GDS enabled")
        print()
        
        # Performance by time window
        print("PEAK DATA TRANSFER BY TIME WINDOW:")
        print(f"{'Time Window':<12} {'COMPAT_MODE=OFF':<20} {'COMPAT_MODE=ON':<20} {'Best Config':<30}")
        print("-" * 85)
        
        for i, time_window in enumerate(self.time_windows):
            time_str = _format_time(time_window)
            
            max_off = 0
            max_on = 0
            best_config = ""
            best_overall = 0
            
            for compat_mode in self.compat_modes:
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_results'][compat_mode][thread_count] and
                        i < len(self.results['aggregate_results'][compat_mode][thread_count])):
                        
                        gb = self.results['aggregate_results'][compat_mode][thread_count][i]['bytes_mean'] / (1024**3)
                        bw = self.results['aggregate_results'][compat_mode][thread_count][i]['bandwidth_mean']
                        
                        if compat_mode == 'OFF':
                            max_off = max(max_off, gb)
                        else:
                            max_on = max(max_on, gb)
                        
                        if gb > best_overall:
                            best_overall = gb
                            best_config = f"{compat_mode}, {thread_count}T, {bw:.1f}GB/s"
            
            print(f"{time_str:<12} {max_off:<20.3f}GB {max_on:<20.3f}GB {best_config:<30}")
        print()
        
        # Key time windows summary
        print("KEY TIME WINDOWS PERFORMANCE:")
        key_windows = [0.0001, 0.001, 0.01, 0.1, 1.0]  # 0.1ms, 1ms, 10ms, 100ms, 1s
        
        for window in key_windows:
            if window in self.time_windows:
                idx = self.time_windows.index(window)
                print(f"\n{_format_time(window)} Window:")
                
                for compat_mode in self.compat_modes:
                    max_gb = 0
                    max_bw = 0
                    best_threads = 1
                    
                    for thread_count in self.thread_counts:
                        if (self.results['aggregate_results'][compat_mode][thread_count] and
                            idx < len(self.results['aggregate_results'][compat_mode][thread_count])):
                            
                            item = self.results['aggregate_results'][compat_mode][thread_count][idx]
                            gb = item['bytes_mean'] / (1024**3)
                            gb_std = item['bytes_std'] / (1024**3)
                            bw = item['bandwidth_mean']
                            
                            if gb > max_gb:
                                max_gb = gb
                                max_bw = bw
                                best_threads = thread_count
                    
                    print(f"  COMPAT_MODE={compat_mode}: {max_gb:.3f}±{gb_std:.3f} GB "
                          f"({max_bw:.2f} GB/s) with {best_threads} threads")
        
        print("\n*** PERFORMANCE RECOMMENDATIONS ***")
        
        # Find overall best configuration
        best_gb_1s = 0
        best_config_1s = None
        if 1.0 in self.time_windows:
            idx_1s = self.time_windows.index(1.0)
            for compat_mode in self.compat_modes:
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_results'][compat_mode][thread_count] and
                        idx_1s < len(self.results['aggregate_results'][compat_mode][thread_count])):
                        
                        gb = self.results['aggregate_results'][compat_mode][thread_count][idx_1s]['bytes_mean'] / (1024**3)
                        if gb > best_gb_1s:
                            best_gb_1s = gb
                            best_config_1s = (compat_mode, thread_count)
        
        if best_config_1s:
            print(f"[RECOMMENDED] Best configuration for sustained transfers (1s): "
                  f"COMPAT_MODE={best_config_1s[0]}, {best_config_1s[1]} threads")
            print(f"              Achieves {best_gb_1s:.2f} GB transfer in 1 second")
        
        # Check latency performance
        if 0.0001 in self.time_windows:  # 0.1ms
            idx_100us = self.time_windows.index(0.0001)
            max_bytes_100us = 0
            best_config_100us = None
            
            for compat_mode in self.compat_modes:
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_results'][compat_mode][thread_count] and
                        idx_100us < len(self.results['aggregate_results'][compat_mode][thread_count])):
                        
                        mb = self.results['aggregate_results'][compat_mode][thread_count][idx_100us]['bytes_mean'] / (1024**2)
                        if mb > max_bytes_100us:
                            max_bytes_100us = mb
                            best_config_100us = (compat_mode, thread_count)
            
            if best_config_100us:
                print(f"[LOW LATENCY] Best configuration for 0.1ms transfers: "
                      f"COMPAT_MODE={best_config_100us[0]}, {best_config_100us[1]} threads")
                print(f"              Achieves {max_bytes_100us:.2f} MB in 0.1ms")
        
        print("="*80)

    def save_results(self):
        """Save results to JSON file."""
        # Convert results to JSON-serializable format
        json_results = {
            'time_windows': self.time_windows,
            'thread_counts': self.thread_counts,
            'compat_modes': self.compat_modes,
            'chunk_size': self.chunk_size,
            'system_info': self.results['system_info'],
            'gds_status': self.results['gds_status']
        }
        
        # Convert gpu_results
        json_results['gpu_results'] = {}
        for compat_mode in self.results['gpu_results']:
            json_results['gpu_results'][compat_mode] = {}
            for thread_count in self.results['gpu_results'][compat_mode]:
                json_results['gpu_results'][compat_mode][str(thread_count)] = {}
                for gpu_id in self.results['gpu_results'][compat_mode][thread_count]:
                    gpu_data = self.results['gpu_results'][compat_mode][thread_count][gpu_id]
                    # Convert numpy arrays to lists
                    converted_data = []
                    for item in gpu_data:
                        converted_item = {}
                        for key, value in item.items():
                            if isinstance(value, np.ndarray):
                                converted_item[key] = value.tolist()
                            elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                                converted_item[key] = float(value)
                            else:
                                converted_item[key] = value
                        converted_data.append(converted_item)
                    json_results['gpu_results'][compat_mode][str(thread_count)][str(gpu_id)] = converted_data
        
        # Convert aggregate_results
        json_results['aggregate_results'] = {}
        for compat_mode in self.results['aggregate_results']:
            json_results['aggregate_results'][compat_mode] = {}
            for thread_count in self.results['aggregate_results'][compat_mode]:
                agg_data = self.results['aggregate_results'][compat_mode][thread_count]
                converted_agg_data = []
                for item in agg_data:
                    converted_item = {}
                    for key, value in item.items():
                        if isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                            converted_item[key] = float(value)
                        else:
                            converted_item[key] = value
                    converted_agg_data.append(converted_item)
                json_results['aggregate_results'][compat_mode][str(thread_count)] = converted_agg_data
        
        # Convert usage stats
        json_results['cpu_usage'] = {}
        json_results['ram_usage'] = {}
        for compat_mode in self.results['cpu_usage']:
            json_results['cpu_usage'][compat_mode] = {}
            json_results['ram_usage'][compat_mode] = {}
            for thread_count in self.results['cpu_usage'][compat_mode]:
                json_results['cpu_usage'][compat_mode][str(thread_count)] = \
                    dict(self.results['cpu_usage'][compat_mode][thread_count])
                json_results['ram_usage'][compat_mode][str(thread_count)] = \
                    dict(self.results['ram_usage'][compat_mode][thread_count])
        
        with open("kv_cache_time_benchmark_results.json", "w") as f:
            json.dump(json_results, f, indent=2)
        
        logger.info("Results saved to kv_cache_time_benchmark_results.json")

def main():
    """Main function to run the time-based benchmark."""
    test_file = "/mnt/kvcache/huge_test_4tb.bin"
    
    try:
        # Check PyTorch CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in PyTorch")
        
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        # Initialize and run benchmark
        # You can adjust chunk_size here - smaller chunks allow more iterations in short time windows
        # but may have more overhead. 64MB is a good balance.
        tester = KVCacheTimeBandwidthTester(test_file, chunk_size=64 * 1024 * 1024)
        results = tester.run_benchmark()
        
        # Generate visualizations, summary and save results
        tester.visualize_results()
        tester.print_summary()
        tester.save_results()
        
        logger.info("Benchmark completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()