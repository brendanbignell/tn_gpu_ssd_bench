#!/usr/bin/env python3
"""
KV Cache Tensor Transfer Bandwidth Benchmark - GDS Optimized
Tests bandwidth of moving KV cache tensors from NVME storage to GPU devices.
Optimized for GPU Direct Storage (GDS) using PyTorch + Kvikio.
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

def run_single_config_benchmark(test_file_path: str, transfer_sizes: List[int], 
                               compat_mode: str, thread_count: int,
                               num_gpus: int, file_size: int, result_queue: Queue):
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
            args=(gpu_id, test_file_path, transfer_sizes, 
                  compat_mode, thread_count, file_size, gpu_queue)
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

def gpu_worker_process_single_config(gpu_id: int, test_file_path: str, transfer_sizes: List[int], 
                                    compat_mode: str, thread_count: int, file_size: int, result_queue: Queue):
    """
    Worker process for testing a single GPU with a specific (COMPAT_MODE, NTHREADS) configuration.
    Assumes both environment variables are already set.
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
    
    # Results storage (single configuration, so no nested dimensions)
    gpu_results = []
    cpu_usage_results = {}
    ram_usage_results = {}
    
    # Test GDS capability for this configuration
    gds_enabled = False
    try:
        # Multiple attempts with different buffer sizes
        for test_size in [1024, 4096, 16384, 65536]:  # 1KB, 4KB, 16KB, 64KB
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
    
    # Test each transfer size
    for transfer_size in transfer_sizes:
        size_str = _format_size(transfer_size)
        
        try:
            # Memory check
            torch.cuda.empty_cache()
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            
            if transfer_size > free_mem * 0.8:
                logger.warning(f"GPU {gpu_id}: Transfer size {size_str} exceeds available memory")
                gpu_results.append(0.0)
                continue
            
            # Allocate GPU buffer
            gpu_buffer = torch.empty(transfer_size // 4, dtype=torch.float32, device=device)
            
            # Start system monitoring
            monitor = SystemMonitor(sampling_interval=0.05)
            monitor.start_monitoring()
            
            # Performance measurement
            times = []
            successful_transfers = 0
            num_iterations = 5
            align = 1024 * 1024  # 1MB alignment
            
            # Warmup
            for warmup_iter in range(2):
                try:
                    # Print GDS status
                    print(f"GPU {gpu_id} warmup: GDS {'ENABLED' if gds_enabled else 'DISABLED'}")

                    if gds_enabled:
                        with kvikio.CuFile(test_file_path, "rb") as f:
                            future = f.pread(gpu_buffer, transfer_size, 0)
                            bytes_read = future.get()
                    else:
                        # Host fallback
                        with open(test_file_path, "rb") as f:
                            data = f.read(transfer_size)
                            if len(data) == transfer_size:
                                host_array = np.frombuffer(data, dtype=np.uint8)
                                host_tensor = torch.from_numpy(host_array).float()
                                copy_size = min(len(host_tensor), len(gpu_buffer))
                                gpu_buffer[:copy_size] = host_tensor[:copy_size].to(device)
                    torch.cuda.synchronize()
                except Exception:
                    continue
            
            # Actual measurements
            for i in range(num_iterations):
                try:
                    # Random offset with 1MB alignment
                    max_offset = file_size - transfer_size
                    offset = np.random.randint(0, max_offset // align) * align
                    
                    start_time = time.perf_counter()
                    
                    if gds_enabled:
                        # TRUE GDS PATH
                        with kvikio.CuFile(test_file_path, "rb") as f:
                            future = f.pread(gpu_buffer, transfer_size, offset)
                            bytes_read = future.get()
                    else:
                        # Host-mediated fallback
                        with open(test_file_path, "rb") as f:
                            f.seek(offset)
                            data = f.read(transfer_size)
                            bytes_read = len(data)
                            if bytes_read == transfer_size:
                                host_array = np.frombuffer(data, dtype=np.uint8)
                                host_tensor = torch.from_numpy(host_array).float()
                                copy_size = min(len(host_tensor), len(gpu_buffer))
                                gpu_buffer[:copy_size] = host_tensor[:copy_size].to(device)
                    
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    
                    if bytes_read == transfer_size:
                        times.append(end_time - start_time)
                        successful_transfers += 1
                    
                except Exception as e:
                    logger.error(f"GPU {gpu_id}: Transfer {i+1} failed: {e}")
                    continue
            
            # Stop monitoring
            system_stats = monitor.stop_monitoring()
            
            # Calculate bandwidth
            if times:
                avg_time = np.mean(times)
                bandwidth_gbps = (transfer_size / avg_time) / (1024**3)
                mode_str = "GDS" if gds_enabled else "Host"
                logger.info(f"GPU {gpu_id}: {size_str}, {thread_count}T, {compat_mode}: "
                          f"{bandwidth_gbps:.2f} GB/s ({mode_str}) "
                          f"({successful_transfers}/{num_iterations} successful)")
                gpu_results.append(bandwidth_gbps)
            else:
                logger.error(f"GPU {gpu_id}: No successful transfers for {size_str}")
                gpu_results.append(0.0)
            
            # Store system stats
            size_mb = transfer_size // (1024**2)
            cpu_usage_results[size_mb] = {
                'avg': system_stats.get('avg_cpu_percent', 0),
                'max': system_stats.get('max_cpu_percent', 0)
            }
            ram_usage_results[size_mb] = {
                'avg_percent': system_stats.get('avg_ram_percent', 0),
                'max_percent': system_stats.get('max_ram_percent', 0),
                'avg_gb': system_stats.get('avg_ram_gb', 0)
            }
            
            # Cleanup
            del gpu_buffer
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Test failed for {size_str}: {e}")
            gpu_results.append(0.0)
            
            size_mb = transfer_size // (1024**2)
            cpu_usage_results[size_mb] = {'avg': 0, 'max': 0}
            ram_usage_results[size_mb] = {
                'avg_percent': 0, 'max_percent': 0, 'avg_gb': 0
            }
    
    result_queue.put((gpu_id, gpu_results, cpu_usage_results, ram_usage_results, gds_enabled))
    logger.info(f"GPU {gpu_id}: Worker process completed for COMPAT_MODE={compat_mode}, NTHREADS={thread_count}")

def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    if size_bytes >= 1024**3:
        return f"{size_bytes // (1024**3)}GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes // (1024**2)}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes // 1024}KB"
    else:
        return f"{size_bytes}B"

class KVCacheBandwidthTester:
    def __init__(self, test_file_path: str = "/mnt/kvcache/huge_test_4tb.bin"):
        """
        Initialize the bandwidth tester with GDS optimization.
        """
        self.test_file_path = test_file_path
        
        # Initialize NVML for GPU detection
        pynvml.nvmlInit()
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        
        # Transfer sizes from 256KB to 16GB
        self.transfer_sizes = [2**i for i in range(18, 35)]
        
        # Thread counts to test
        self.thread_counts = [1, 2, 4, 8, 16, 32, 64]
        
        # KVIKIO compatibility modes to test
        self.compat_modes = ['OFF', 'ON']
        
        # Verify test file exists
        if not os.path.exists(self.test_file_path):
            raise FileNotFoundError(f"Test file not found: {self.test_file_path}")
        
        self.file_size = os.path.getsize(self.test_file_path)
        logger.info(f"Test file size: {self.file_size / (1024**4):.2f} TB")
        
        # Results structure - now includes compat_mode dimension
        self.results = {
            'transfer_sizes': self.transfer_sizes,
            'thread_counts': self.thread_counts,
            'compat_modes': self.compat_modes,
            'gpu_bandwidths': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),  # [compat_mode][thread_count][gpu_id]
            'aggregate_bandwidth': defaultdict(lambda: defaultdict(list)),  # [compat_mode][thread_count]
            'cpu_usage': defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),  # [compat_mode][thread_count][size_mb]
            'ram_usage': defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),  # [compat_mode][thread_count][size_mb]
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
        Run the complete bandwidth benchmark using multiprocessing for true parallelism.
        Tests each (KVIKIO_COMPAT_MODE, KVIKIO_NTHREADS) combination sequentially to avoid GPU memory conflicts.
        """
        logger.info("Starting KV Cache bandwidth benchmark with GDS optimization")
        logger.info(f"Transfer sizes: {[_format_size(size) for size in self.transfer_sizes]}")
        logger.info(f"Thread counts: {self.thread_counts}")
        logger.info(f"KVIKIO compatibility modes: {self.compat_modes}")
        
        # Use multiprocessing for parallel testing
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        total_configs = len(self.compat_modes) * len(self.thread_counts)
        logger.info(f"Testing {total_configs} configurations sequentially to avoid GPU memory conflicts...")
        
        start_time = time.time()
        config_num = 0
        
        # FIXED: Run configurations sequentially to avoid GPU memory conflicts
        for compat_mode in self.compat_modes:
            for thread_count in self.thread_counts:
                config_num += 1
                logger.info(f"=== RUNNING CONFIG {config_num}/{total_configs}: COMPAT_MODE={compat_mode}, NTHREADS={thread_count} ===")
                
                result_queue = Queue()
                
                # Start single configuration process
                p = Process(
                    target=run_single_config_benchmark,
                    args=(self.test_file_path, self.transfer_sizes, 
                          compat_mode, thread_count, self.num_gpus, self.file_size, result_queue)
                )
                p.start()
                
                # Wait for this configuration to complete before starting next one
                p.join()
                
                # Collect results from this configuration
                if not result_queue.empty():
                    compat_mode_result, thread_count_result, all_gpu_results, all_cpu_results, all_ram_results, all_gds_status = result_queue.get()
                    
                    logger.info(f"Processing results from COMPAT_MODE={compat_mode_result}, NTHREADS={thread_count_result}")
                    logger.info(f"  Received GPU results keys: {list(all_gpu_results.keys())}")
                    
                    # Store individual GPU results
                    for gpu_id in all_gpu_results:
                        gpu_data = all_gpu_results[gpu_id]  # This is now a simple list of bandwidths
                        if compat_mode_result not in self.results['gpu_bandwidths']:
                            self.results['gpu_bandwidths'][compat_mode_result] = defaultdict(dict)
                        if thread_count_result not in self.results['gpu_bandwidths'][compat_mode_result]:
                            self.results['gpu_bandwidths'][compat_mode_result][thread_count_result] = {}
                        
                        self.results['gpu_bandwidths'][compat_mode_result][thread_count_result][gpu_id] = gpu_data
                        
                        measurements = len(gpu_data)
                        if measurements > 0:
                            max_bw = max(gpu_data)
                            logger.info(f"  Stored GPU {gpu_id} {compat_mode_result}/{thread_count_result}T: "
                                       f"{measurements} measurements, max={max_bw:.2f} GB/s")
                    
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
                        logger.info(f"GPU {gpu_id} COMPAT_MODE={compat_mode_result} NTHREADS={thread_count_result} GDS: {'ENABLED' if all_gds_status[gpu_id] else 'DISABLED'}")
                
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
        
        # Calculate aggregate bandwidth for each configuration
        for compat_mode in self.compat_modes:
            if compat_mode not in self.results['aggregate_bandwidth']:
                self.results['aggregate_bandwidth'][compat_mode] = defaultdict(list)
                
            for thread_count in self.thread_counts:
                aggregate_bandwidths = []
                for i in range(len(self.transfer_sizes)):
                    total_bw = 0
                    for gpu_id in range(self.num_gpus):
                        if (compat_mode in self.results['gpu_bandwidths'] and
                            thread_count in self.results['gpu_bandwidths'][compat_mode] and
                            gpu_id in self.results['gpu_bandwidths'][compat_mode][thread_count] and 
                            i < len(self.results['gpu_bandwidths'][compat_mode][thread_count][gpu_id])):
                            total_bw += self.results['gpu_bandwidths'][compat_mode][thread_count][gpu_id][i]
                    aggregate_bandwidths.append(total_bw)
                self.results['aggregate_bandwidth'][compat_mode][thread_count] = aggregate_bandwidths
                
                # Debug logging
                if aggregate_bandwidths:
                    max_agg = max(aggregate_bandwidths)
                    if max_agg > 0:
                        logger.info(f"Aggregate BW for {compat_mode}/{thread_count}T: max={max_agg:.2f} GB/s")
        
        # Debug: Print summary of what was collected
        logger.info("=== DATA COLLECTION SUMMARY ===")
        for compat_mode in self.compat_modes:
            for gpu_id in range(self.num_gpus):
                gds_status = self.results['gds_status'].get(gpu_id, {}).get(compat_mode, False)
                logger.info(f"GPU {gpu_id} COMPAT_MODE={compat_mode}: GDS={'ENABLED' if gds_status else 'DISABLED'}")
                
                # Check if we have data for this GPU/mode
                total_measurements = 0
                peak_bandwidth = 0
                for thread_count in self.thread_counts:
                    if (compat_mode in self.results['gpu_bandwidths'] and
                        thread_count in self.results['gpu_bandwidths'][compat_mode] and
                        gpu_id in self.results['gpu_bandwidths'][compat_mode][thread_count]):
                        measurements = len(self.results['gpu_bandwidths'][compat_mode][thread_count][gpu_id])
                        total_measurements += measurements
                        if measurements > 0:
                            max_bw = max(self.results['gpu_bandwidths'][compat_mode][thread_count][gpu_id])
                            peak_bandwidth = max(peak_bandwidth, max_bw)
                            logger.info(f"  {thread_count}T: {measurements} measurements, max={max_bw:.2f} GB/s")
                
                if total_measurements == 0:
                    logger.warning(f"  NO DATA COLLECTED for GPU {gpu_id} COMPAT_MODE={compat_mode}")
                else:
                    logger.info(f"  TOTAL: {total_measurements} measurements, peak={peak_bandwidth:.2f} GB/s")
        logger.info("=== END DATA COLLECTION SUMMARY ===")
        
        logger.info("Benchmark data collection completed")
        return self.results

    def visualize_results(self, save_path: str = "kv_cache_bandwidth_results_gds.png"):
        """
        Create comprehensive visualization of benchmark results including KVIKIO compat modes.
        """
        # Convert transfer sizes to MB for plotting
        transfer_sizes_mb = [size / (1024**2) for size in self.transfer_sizes]
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(24, 20))
        
        # Set style
        sns.set_style("whitegrid")
        colors_compat = {'OFF': 'blue', 'ON': 'red'}
        colors_threads = sns.color_palette("husl", len(self.thread_counts))
        
        # 1. Aggregate Bandwidth Comparison: COMPAT_MODE OFF vs ON
        ax1 = plt.subplot(4, 3, 1)
        for compat_mode in self.compat_modes:
            best_bandwidths = []
            for i, transfer_size in enumerate(self.transfer_sizes):
                max_bw = 0
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_bandwidth'][compat_mode][thread_count] and 
                        i < len(self.results['aggregate_bandwidth'][compat_mode][thread_count])):
                        bw = self.results['aggregate_bandwidth'][compat_mode][thread_count][i]
                        if bw > max_bw:
                            max_bw = bw
                best_bandwidths.append(max_bw)
            
            ax1.plot(transfer_sizes_mb, best_bandwidths, 'o-', 
                    label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode], 
                    linewidth=2, markersize=6)
        
        ax1.set_title('Peak Aggregate Bandwidth: COMPAT_MODE Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Transfer Size (MB)')
        ax1.set_ylabel('Peak Bandwidth (GB/s)')
        ax1.set_xscale('log', base=2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Individual GPU Performance for Best COMPAT_MODE
        ax2 = plt.subplot(4, 3, 2)
        # Find best compat mode (highest peak)
        best_compat_mode = 'OFF'
        max_peak = 0
        for compat_mode in self.compat_modes:
            for thread_count in self.thread_counts:
                if self.results['aggregate_bandwidth'][compat_mode][thread_count]:
                    peak = max(self.results['aggregate_bandwidth'][compat_mode][thread_count])
                    if peak > max_peak:
                        max_peak = peak
                        best_compat_mode = compat_mode
        
        # Find best thread count for best compat mode
        best_thread_count = 1
        max_bw = 0
        for thread_count in self.thread_counts:
            if self.results['aggregate_bandwidth'][best_compat_mode][thread_count]:
                bw = max(self.results['aggregate_bandwidth'][best_compat_mode][thread_count])
                if bw > max_bw:
                    max_bw = bw
                    best_thread_count = thread_count
        
        for gpu_id in range(self.num_gpus):
            if (gpu_id in self.results['gpu_bandwidths'][best_compat_mode][best_thread_count]):
                bandwidths = self.results['gpu_bandwidths'][best_compat_mode][best_thread_count][gpu_id]
                # Only plot if there's actual data (not all zeros)
                if any(bw > 0 for bw in bandwidths):
                    ax2.plot(transfer_sizes_mb, bandwidths, 'o-', 
                            label=f'GPU {gpu_id}', linewidth=2, markersize=4)
        
        ax2.set_title(f'Individual GPU Performance\n(COMPAT_MODE={best_compat_mode}, {best_thread_count} threads)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Transfer Size (MB)')
        ax2.set_ylabel('Bandwidth (GB/s)')
        ax2.set_xscale('log', base=2)
        
        # Only add legend if there are labeled lines
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend()
        
        ax2.grid(True, alpha=0.3)
        
        # 3. Thread Count Performance for Each COMPAT_MODE (256MB transfers)
        ax3 = plt.subplot(4, 3, 3)
        # Find 256MB transfer size index
        target_size_mb = 256
        selected_size_idx = None
        for i, size in enumerate(self.transfer_sizes):
            if size // (1024**2) == target_size_mb:
                selected_size_idx = i
                break
        
        if selected_size_idx is None:
            # If exact 256MB not found, find closest
            size_mb_list = [s // (1024**2) for s in self.transfer_sizes]
            selected_size_idx = min(range(len(size_mb_list)), key=lambda i: abs(size_mb_list[i] - target_size_mb))
        
        for compat_mode in self.compat_modes:
            thread_performance = []
            for thread_count in self.thread_counts:
                if (self.results['aggregate_bandwidth'][compat_mode][thread_count] and 
                    selected_size_idx < len(self.results['aggregate_bandwidth'][compat_mode][thread_count])):
                    bw = self.results['aggregate_bandwidth'][compat_mode][thread_count][selected_size_idx]
                    thread_performance.append(bw)
                else:
                    thread_performance.append(0)
            
            ax3.plot(self.thread_counts, thread_performance, 'o-', 
                    label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode],
                    linewidth=2, markersize=6)
        
        selected_size_mb = self.transfer_sizes[selected_size_idx] // (1024**2)
        ax3.set_title(f'Threading Performance ({selected_size_mb}MB transfers)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Thread Count')
        ax3.set_ylabel('Aggregate Bandwidth (GB/s)')
        ax3.set_xscale('log', base=2)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. CPU Usage Comparison
        ax4 = plt.subplot(4, 3, 4)
        for compat_mode in self.compat_modes:
            cpu_by_size = []
            for transfer_size in self.transfer_sizes:
                size_mb = transfer_size // (1024**2)
                cpu_vals = []
                for thread_count in self.thread_counts:
                    if size_mb in self.results['cpu_usage'][compat_mode][thread_count]:
                        cpu_vals.append(self.results['cpu_usage'][compat_mode][thread_count][size_mb]['avg'])
                cpu_by_size.append(np.mean(cpu_vals) if cpu_vals else 0)
            
            ax4.plot(transfer_sizes_mb, cpu_by_size, 'o-', 
                    label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode],
                    linewidth=2, markersize=4)
        
        ax4.set_title('Average CPU Usage by Transfer Size', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Transfer Size (MB)')
        ax4.set_ylabel('CPU Usage (%)')
        ax4.set_xscale('log', base=2)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. RAM Usage Comparison
        ax5 = plt.subplot(4, 3, 5)
        for compat_mode in self.compat_modes:
            ram_by_size = []
            for transfer_size in self.transfer_sizes:
                size_mb = transfer_size // (1024**2)
                ram_vals = []
                for thread_count in self.thread_counts:
                    if size_mb in self.results['ram_usage'][compat_mode][thread_count]:
                        ram_vals.append(self.results['ram_usage'][compat_mode][thread_count][size_mb]['avg_gb'])
                ram_by_size.append(np.mean(ram_vals) if ram_vals else 0)
            
            ax5.plot(transfer_sizes_mb, ram_by_size, 'o-', 
                    label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode],
                    linewidth=2, markersize=4)
        
        ax5.set_title('Average RAM Usage by Transfer Size', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Transfer Size (MB)')
        ax5.set_ylabel('RAM Usage (GB)')
        ax5.set_xscale('log', base=2)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Threading Scaling for Best COMPAT_MODE
        ax6 = plt.subplot(4, 3, 6)
        selected_sizes = [64, 256, 1024]  # MB
        for size_mb in selected_sizes:
            if size_mb in [s//(1024**2) for s in self.transfer_sizes]:
                size_idx = [s//(1024**2) for s in self.transfer_sizes].index(size_mb)
                thread_scaling = []
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_bandwidth'][best_compat_mode][thread_count] and 
                        size_idx < len(self.results['aggregate_bandwidth'][best_compat_mode][thread_count])):
                        bw = self.results['aggregate_bandwidth'][best_compat_mode][thread_count][size_idx]
                        thread_scaling.append(bw)
                    else:
                        thread_scaling.append(0)
                
                ax6.plot(self.thread_counts, thread_scaling, 'o-', 
                        label=f'{size_mb}MB', linewidth=2, markersize=4)
        
        ax6.set_title(f'Threading Scaling (COMPAT_MODE={best_compat_mode})', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Thread Count')
        ax6.set_ylabel('Aggregate Bandwidth (GB/s)')
        ax6.set_xscale('log', base=2)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Threading Scaling for COMPAT_MODE=ON
        ax7 = plt.subplot(4, 3, 7)
        selected_sizes = [64, 256, 1024]  # MB
        for size_mb in selected_sizes:
            if size_mb in [s//(1024**2) for s in self.transfer_sizes]:
                size_idx = [s//(1024**2) for s in self.transfer_sizes].index(size_mb)
                thread_scaling = []
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_bandwidth']['ON'][thread_count] and 
                        size_idx < len(self.results['aggregate_bandwidth']['ON'][thread_count])):
                        bw = self.results['aggregate_bandwidth']['ON'][thread_count][size_idx]
                        thread_scaling.append(bw)
                    else:
                        thread_scaling.append(0)
                
                ax7.plot(self.thread_counts, thread_scaling, 'o-', 
                        label=f'{size_mb}MB', linewidth=2, markersize=4)
        
        ax7.set_title('Threading Scaling (COMPAT_MODE=ON)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Thread Count')
        ax7.set_ylabel('Aggregate Bandwidth (GB/s)')
        ax7.set_xscale('log', base=2)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Performance Summary Table
        ax8 = plt.subplot(4, 3, 8)
        ax8.axis('off')
        
        # Create summary table comparing COMPAT_MODEs
        summary_data = []
        for compat_mode in self.compat_modes:
            max_agg = 0
            avg_agg = 0
            for thread_count in self.thread_counts:
                if self.results['aggregate_bandwidth'][compat_mode][thread_count]:
                    thread_max = max(self.results['aggregate_bandwidth'][compat_mode][thread_count])
                    thread_avg = np.mean(self.results['aggregate_bandwidth'][compat_mode][thread_count])
                    if thread_max > max_agg:
                        max_agg = thread_max
                    avg_agg += thread_avg
            
            avg_agg /= len(self.thread_counts)
            gds_count = sum(1 for gpu_id in range(self.num_gpus) 
                           if self.results['gds_status'].get(gpu_id, {}).get(compat_mode, False))
            
            summary_data.append([
                f'COMPAT_MODE={compat_mode}',
                f'{max_agg:.1f}',
                f'{avg_agg:.1f}',
                f'{gds_count}/{self.num_gpus}'
            ])
        
        table = ax8.table(cellText=summary_data,
                         colLabels=['Mode', 'Peak (GB/s)', 'Avg (GB/s)', 'GDS GPUs'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax8.set_title('Performance Summary by COMPAT_MODE', fontsize=12, fontweight='bold')
        
        # 9. Best Configuration Heatmap
        ax9 = plt.subplot(4, 3, 9)
        best_config_matrix = np.zeros((len(self.compat_modes), len(self.thread_counts)))
        for i, compat_mode in enumerate(self.compat_modes):
            for j, thread_count in enumerate(self.thread_counts):
                if self.results['aggregate_bandwidth'][compat_mode][thread_count]:
                    best_config_matrix[i, j] = max(self.results['aggregate_bandwidth'][compat_mode][thread_count])
        
        im9 = ax9.imshow(best_config_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax9.set_title('Peak Bandwidth by Configuration', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Thread Count')
        ax9.set_ylabel('COMPAT_MODE')
        ax9.set_xticks(range(len(self.thread_counts)))
        ax9.set_xticklabels(self.thread_counts)
        ax9.set_yticks(range(len(self.compat_modes)))
        ax9.set_yticklabels(self.compat_modes)
        plt.colorbar(im9, ax=ax9, shrink=0.6, label='Bandwidth (GB/s)')
        
        # 10. Scaling Efficiency Analysis
        ax10 = plt.subplot(4, 3, 10)
        for compat_mode in self.compat_modes:
            scaling_efficiency = []
            for i, transfer_size in enumerate(self.transfer_sizes):
                # Find max individual GPU performance for this transfer size
                max_individual = 0
                for thread_count in self.thread_counts:
                    for gpu_id in range(self.num_gpus):
                        if (gpu_id in self.results['gpu_bandwidths'][compat_mode][thread_count] and
                            i < len(self.results['gpu_bandwidths'][compat_mode][thread_count][gpu_id])):
                            gpu_bw = self.results['gpu_bandwidths'][compat_mode][thread_count][gpu_id][i]
                            if gpu_bw > max_individual:
                                max_individual = gpu_bw
                
                # Find max aggregate for this transfer size
                max_aggregate = 0
                for thread_count in self.thread_counts:
                    if (self.results['aggregate_bandwidth'][compat_mode][thread_count] and
                        i < len(self.results['aggregate_bandwidth'][compat_mode][thread_count])):
                        agg_bw = self.results['aggregate_bandwidth'][compat_mode][thread_count][i]
                        if agg_bw > max_aggregate:
                            max_aggregate = agg_bw
                
                # Calculate efficiency
                if max_individual > 0:
                    efficiency = (max_aggregate / (max_individual * self.num_gpus)) * 100
                    scaling_efficiency.append(efficiency)
                else:
                    scaling_efficiency.append(0)
            
            ax10.plot(transfer_sizes_mb, scaling_efficiency, 'o-', 
                     label=f'COMPAT_MODE={compat_mode}', color=colors_compat[compat_mode],
                     linewidth=2, markersize=4)
        
        ax10.set_title('Multi-GPU Scaling Efficiency', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Transfer Size (MB)')
        ax10.set_ylabel('Scaling Efficiency (%)')
        ax10.set_xscale('log', base=2)
        ax10.set_ylim(0, 100)
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive results saved to {save_path}")
        plt.show()

    def print_summary(self):
        """Print a comprehensive summary of benchmark results including COMPAT_MODE comparison."""
        print("\n" + "="*80)
        print("KV CACHE BANDWIDTH BENCHMARK SUMMARY (GDS OPTIMIZED)")
        print("="*80)
        
        # System info
        info = self.results['system_info']
        print(f"System: AMD Threadripper Pro")
        print(f"CPUs: {info['cpu_count']} physical, {info['cpu_count_logical']} logical")
        print(f"Memory: {info['memory_total'] / (1024**3):.1f} GB")
        print(f"GPUs: {info['num_gpus']}")
        print()
        
        # GDS status by compat mode
        print("GDS STATUS BY COMPATIBILITY MODE:")
        for compat_mode in self.compat_modes:
            gds_gpus = sum(1 for gpu_id in range(info['num_gpus']) 
                          if self.results['gds_status'].get(gpu_id, {}).get(compat_mode, False))
            print(f"KVIKIO_COMPAT_MODE={compat_mode}: {gds_gpus}/{info['num_gpus']} GPUs with GDS enabled")
            for gpu_id in range(info['num_gpus']):
                if gpu_id in self.results['gds_status']:
                    status = "[ENABLED]" if self.results['gds_status'][gpu_id].get(compat_mode, False) else "[DISABLED]"
                    print(f"  GPU {gpu_id}: {status}")
        print()
        
        # Find best performance across all configurations
        best_configs = {}
        overall_best = {'bw': 0, 'compat_mode': 'OFF', 'thread_count': 1, 'size': self.transfer_sizes[0]}
        
        for compat_mode in self.compat_modes:
            max_individual = 0
            max_aggregate = 0
            best_thread_count = 1
            best_size_agg = self.transfer_sizes[0]
            
            for thread_count in self.thread_counts:
                if self.results['aggregate_bandwidth'][compat_mode][thread_count]:
                    thread_max_agg = max(self.results['aggregate_bandwidth'][compat_mode][thread_count])
                    if thread_max_agg > max_aggregate:
                        max_aggregate = thread_max_agg
                        best_thread_count = thread_count
                        max_idx = self.results['aggregate_bandwidth'][compat_mode][thread_count].index(thread_max_agg)
                        best_size_agg = self.transfer_sizes[max_idx]
                
                for gpu_id in range(self.num_gpus):
                    if (gpu_id in self.results['gpu_bandwidths'][compat_mode][thread_count] and 
                        self.results['gpu_bandwidths'][compat_mode][thread_count][gpu_id]):
                        gpu_max = max(self.results['gpu_bandwidths'][compat_mode][thread_count][gpu_id])
                        if gpu_max > max_individual:
                            max_individual = gpu_max
            
            best_configs[compat_mode] = {
                'max_individual': max_individual,
                'max_aggregate': max_aggregate,
                'best_thread_count': best_thread_count,
                'best_size_agg': best_size_agg
            }
            
            # Track overall best
            if max_aggregate > overall_best['bw']:
                overall_best = {
                    'bw': max_aggregate,
                    'compat_mode': compat_mode,
                    'thread_count': best_thread_count,
                    'size': best_size_agg
                }
        
        # Overall best performance
        print("OVERALL PEAK PERFORMANCE:")
        print(f"Peak Individual GPU Bandwidth: {max(best_configs[mode]['max_individual'] for mode in self.compat_modes):.2f} GB/s")
        print(f"Peak Aggregate Bandwidth: {overall_best['bw']:.2f} GB/s")
        print(f"Best Configuration: COMPAT_MODE={overall_best['compat_mode']}, {overall_best['thread_count']} threads, {_format_size(overall_best['size'])} transfers")
        
        # Calculate scaling efficiency for best config
        best_mode = overall_best['compat_mode']
        max_individual_best = best_configs[best_mode]['max_individual']
        if max_individual_best > 0:
            scaling_efficiency = (overall_best['bw'] / (max_individual_best * self.num_gpus)) * 100
            print(f"Scaling Efficiency: {scaling_efficiency:.1f}%")
        else:
            print("Scaling Efficiency: N/A (no successful transfers)")
        print()
        
        # Performance comparison by COMPAT_MODE
        print("PERFORMANCE BY COMPAT_MODE:")
        print(f"{'Mode':<12} {'Peak (GB/s)':<12} {'Avg (GB/s)':<12} {'Best Config':<25} {'GDS GPUs':<10}")
        print("-" * 80)
        
        for compat_mode in self.compat_modes:
            config = best_configs[compat_mode]
            peak_bw = config['max_aggregate']
            
            # Calculate average across all configurations
            all_bandwidths = []
            for thread_count in self.thread_counts:
                if self.results['aggregate_bandwidth'][compat_mode][thread_count]:
                    all_bandwidths.extend(self.results['aggregate_bandwidth'][compat_mode][thread_count])
            avg_bw = np.mean(all_bandwidths) if all_bandwidths else 0
            
            best_config_str = f"{config['best_thread_count']}T, {_format_size(config['best_size_agg'])}"
            gds_count = sum(1 for gpu_id in range(info['num_gpus']) 
                           if self.results['gds_status'].get(gpu_id, {}).get(compat_mode, False))
            
            print(f"{compat_mode:<12} {peak_bw:<12.2f} {avg_bw:<12.2f} {best_config_str:<25} {gds_count}/{info['num_gpus']:<10}")
        print()
        
        # Threading performance for best compat mode
        print(f"THREADING PERFORMANCE (COMPAT_MODE={best_mode}):")
        print(f"{'Threads':<8} {'Peak (GB/s)':<12} {'Avg (GB/s)':<12} {'Best Size':<12} {'Efficiency':<12}")
        print("-" * 64)
        
        for thread_count in self.thread_counts:
            if self.results['aggregate_bandwidth'][best_mode][thread_count]:
                peak_bw = max(self.results['aggregate_bandwidth'][best_mode][thread_count])
                avg_bw = np.mean(self.results['aggregate_bandwidth'][best_mode][thread_count])
                best_idx = self.results['aggregate_bandwidth'][best_mode][thread_count].index(peak_bw)
                best_size = self.transfer_sizes[best_idx] // (1024**2)
                
                # Calculate efficiency for this thread count
                thread_max_individual = 0
                for gpu_id in range(self.num_gpus):
                    if (gpu_id in self.results['gpu_bandwidths'][best_mode][thread_count] and
                        self.results['gpu_bandwidths'][best_mode][thread_count][gpu_id]):
                        gpu_max = max(self.results['gpu_bandwidths'][best_mode][thread_count][gpu_id])
                        if gpu_max > thread_max_individual:
                            thread_max_individual = gpu_max
                
                if thread_max_individual > 0:
                    efficiency = (peak_bw / (thread_max_individual * self.num_gpus)) * 100
                else:
                    efficiency = 0
                
                print(f"{thread_count:<8} {peak_bw:<12.2f} {avg_bw:<12.2f} {best_size:<12}MB {efficiency:<12.1f}%")
            else:
                print(f"{thread_count:<8} {'0.00':<12} {'0.00':<12} {'N/A':<12} {'0.0':<12}%")
        print()
        
        # Performance recommendations
        print("*** PERFORMANCE RECOMMENDATIONS ***")
        
        # Compare compat modes
        off_peak = best_configs.get('OFF', {}).get('max_aggregate', 0)
        on_peak = best_configs.get('ON', {}).get('max_aggregate', 0)
        
        if off_peak > on_peak * 1.05:  # 5% threshold
            print(f"[RECOMMENDED] Use KVIKIO_COMPAT_MODE=OFF for best performance ({off_peak:.1f} vs {on_peak:.1f} GB/s)")
        elif on_peak > off_peak * 1.05:
            print(f"[RECOMMENDED] Use KVIKIO_COMPAT_MODE=ON for best performance ({on_peak:.1f} vs {off_peak:.1f} GB/s)")
        else:
            print(f"[NEUTRAL] Both COMPAT_MODEs perform similarly (OFF: {off_peak:.1f}, ON: {on_peak:.1f} GB/s)")
        
        print(f"[RECOMMENDED] Optimal thread count: {overall_best['thread_count']} threads")
        print(f"[RECOMMENDED] Optimal transfer size: {_format_size(overall_best['size'])}")
        
        # GDS status check
        all_gds_enabled = all(
            all(self.results['gds_status'].get(gpu_id, {}).get(mode, False) 
                for gpu_id in range(info['num_gpus']))
            for mode in self.compat_modes
        )
        
        if all_gds_enabled:
            print("[SUCCESS] GDS is enabled on all GPUs for both compat modes - optimal configuration!")
        else:
            print("[WARNING] GDS is not enabled on all GPUs - performance may be suboptimal")
            print("          Check CUDA drivers, file system, and cuFile installation")
        
        print("="*80)

    def save_results(self):
        """Save results to JSON file with COMPAT_MODE data."""
        # Convert results to JSON-serializable format
        json_results = {
            'transfer_sizes': [int(x) for x in self.results['transfer_sizes']],
            'thread_counts': self.results['thread_counts'],
            'compat_modes': self.results['compat_modes'],
            'system_info': self.results['system_info'],
            'gds_status': self.results['gds_status']
        }
        
        # Convert gpu_bandwidths - now has compat_mode dimension
        json_results['gpu_bandwidths'] = {}
        for compat_mode in self.results['gpu_bandwidths']:
            json_results['gpu_bandwidths'][compat_mode] = {}
            for thread_count in self.results['gpu_bandwidths'][compat_mode]:
                json_results['gpu_bandwidths'][compat_mode][str(thread_count)] = {}
                for gpu_id in self.results['gpu_bandwidths'][compat_mode][thread_count]:
                    json_results['gpu_bandwidths'][compat_mode][str(thread_count)][str(gpu_id)] = \
                        [float(x) for x in self.results['gpu_bandwidths'][compat_mode][thread_count][gpu_id]]
        
        # Convert aggregate_bandwidth - now has compat_mode dimension
        json_results['aggregate_bandwidth'] = {}
        for compat_mode in self.results['aggregate_bandwidth']:
            json_results['aggregate_bandwidth'][compat_mode] = {}
            for thread_count in self.results['aggregate_bandwidth'][compat_mode]:
                json_results['aggregate_bandwidth'][compat_mode][str(thread_count)] = \
                    [float(x) for x in self.results['aggregate_bandwidth'][compat_mode][thread_count]]
        
        # Convert usage stats - now has compat_mode dimension
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
        
        with open("kv_cache_benchmark_results_gds.json", "w") as f:
            json.dump(json_results, f, indent=2)
        
        logger.info("Results saved to kv_cache_benchmark_results_gds.json")

def diagnose_gds_issues(test_file_path: str):
    """Diagnose potential GDS issues."""
    logger.info("=== GDS DIAGNOSTIC INFORMATION ===")
    
    # Check file permissions
    try:
        import stat
        file_stat = os.stat(test_file_path)
        permissions = stat.filemode(file_stat.st_mode)
        logger.info(f"File permissions: {permissions}")
        logger.info(f"File size: {file_stat.st_size / (1024**4):.2f} TB")
        
        # Test basic file access
        with open(test_file_path, "rb") as f:
            test_data = f.read(1024)
            logger.info(f"Basic file read: {'SUCCESS' if len(test_data) == 1024 else 'FAILED'}")
    except Exception as e:
        logger.error(f"File access test failed: {e}")
    
    # Check environment variables
    logger.info("Environment variables:")
    for var in ['KVIKIO_COMPAT_MODE', 'KVIKIO_NTHREADS', 'CUDA_VISIBLE_DEVICES']:
        value = os.environ.get(var, 'NOT_SET')
        logger.info(f"  {var}={value}")
    
    # Check CUDA and PyTorch
    try:
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch device count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name}, {props.total_memory//1024//1024}MB")
    except Exception as e:
        logger.error(f"CUDA diagnostic failed: {e}")
    
    # Test Kvikio basic functionality
    try:
        logger.info("Testing Kvikio basic functionality...")
        # Import kvikio here for diagnostic purposes only
        import kvikio
        test_buffer = np.zeros(1024, dtype=np.uint8)
        with kvikio.CuFile(test_file_path, "rb") as f:
            future = f.pread(test_buffer, 1024, 0)
            bytes_read = future.get()
            logger.info(f"Kvikio basic test: {'SUCCESS' if bytes_read == 1024 else 'FAILED'} ({bytes_read} bytes)")
    except Exception as e:
        logger.error(f"Kvikio basic test failed: {e}")
    
    logger.info("=== END GDS DIAGNOSTIC ===")

def main():
    """Main function to run the GDS-optimized benchmark."""
    test_file = "/mnt/kvcache/huge_test_4tb.bin"
    
    try:
        # Check PyTorch CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in PyTorch")
        
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        # Run GDS diagnostics first
        diagnose_gds_issues(test_file)
        
        # Initialize and run benchmark
        tester = KVCacheBandwidthTester(test_file)
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