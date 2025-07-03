#!/usr/bin/env python3
"""
Enhanced KV Cache Tensor Transfer Bandwidth Benchmark - GDS Optimized with Compute Interference Testing
Tests bandwidth of moving KV cache tensors from NVME storage to GPU devices while measuring
interference with concurrent GPU computations (matrix multiplication).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process, Queue, set_start_method
import psutil
import gc
from typing import List, Dict, Tuple, Optional, Literal
import logging
from collections import defaultdict
import json
from threading import Thread, Event
from queue import Queue as ThreadQueue
import signal

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

class GPUComputeWorker:
    """Manages GPU compute workloads (matrix multiplication) for interference testing."""
    
    def __init__(self, device: torch.device, matrix_size: int = 4096, 
                 compute_intensity: Literal['light', 'medium', 'heavy'] = 'medium'):
        self.device = device
        self.matrix_size = matrix_size
        self.compute_intensity = compute_intensity
        self.running = False
        self.compute_thread = None
        self.stop_event = Event()
        self.metrics_queue = ThreadQueue()
        
        # Configure workload based on intensity
        self.intensity_configs = {
            'light': {'matrix_size': 2048, 'batch_size': 1, 'sleep_ms': 10},
            'medium': {'matrix_size': 4096, 'batch_size': 2, 'sleep_ms': 5},
            'heavy': {'matrix_size': 8192, 'batch_size': 4, 'sleep_ms': 0}
        }
        
        config = self.intensity_configs[compute_intensity]
        self.matrix_size = config['matrix_size']
        self.batch_size = config['batch_size']
        self.sleep_ms = config['sleep_ms']
        
        # Pre-allocate matrices for compute workload
        self._initialize_matrices()
        
        # Metrics tracking
        self.total_flops = 0
        self.total_time = 0
        self.iterations = 0
        
    def _initialize_matrices(self):
        """Pre-allocate matrices for compute workload."""
        try:
            # Allocate matrices on GPU
            self.matrix_a = torch.randn(
                self.batch_size, self.matrix_size, self.matrix_size, 
                device=self.device, dtype=torch.float32
            )
            self.matrix_b = torch.randn(
                self.batch_size, self.matrix_size, self.matrix_size, 
                device=self.device, dtype=torch.float32
            )
            self.matrix_c = torch.empty(
                self.batch_size, self.matrix_size, self.matrix_size, 
                device=self.device, dtype=torch.float32
            )
            
            # Warmup
            for _ in range(5):
                torch.matmul(self.matrix_a, self.matrix_b, out=self.matrix_c)
            torch.cuda.synchronize()
            
            logger.info(f"Initialized compute workload: {self.compute_intensity} intensity, "
                       f"{self.matrix_size}x{self.matrix_size} matrices, batch_size={self.batch_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize compute matrices: {e}")
            raise
    
    def start(self):
        """Start the compute workload in a separate thread."""
        if not self.running:
            self.running = True
            self.stop_event.clear()
            self.compute_thread = Thread(target=self._compute_loop)
            self.compute_thread.start()
            logger.info("Started GPU compute workload")
    
    def stop(self) -> Dict[str, float]:
        """Stop the compute workload and return metrics."""
        if self.running:
            self.running = False
            self.stop_event.set()
            self.compute_thread.join()
            
            # Calculate metrics
            avg_gflops = 0
            if self.iterations > 0 and self.total_time > 0:
                # FLOPS calculation for matrix multiplication: 2 * n^3 * batch_size
                flops_per_iteration = 2 * (self.matrix_size ** 3) * self.batch_size
                total_flops = flops_per_iteration * self.iterations
                avg_gflops = (total_flops / self.total_time) / 1e9
            
            metrics = {
                'total_iterations': self.iterations,
                'total_time': self.total_time,
                'avg_gflops': avg_gflops,
                'compute_intensity': self.compute_intensity
            }
            
            logger.info(f"Stopped GPU compute workload: {self.iterations} iterations, "
                       f"{avg_gflops:.2f} GFLOPS average")
            
            return metrics
        
        return {'total_iterations': 0, 'total_time': 0, 'avg_gflops': 0}
    
    def _compute_loop(self):
        """Main compute loop running matrix multiplications."""
        start_time = time.perf_counter()
        
        # Create a dedicated CUDA stream for compute
        compute_stream = torch.cuda.Stream(device=self.device)
        
        while self.running and not self.stop_event.is_set():
            try:
                with torch.cuda.stream(compute_stream):
                    # Perform matrix multiplication
                    iter_start = time.perf_counter()
                    torch.matmul(self.matrix_a, self.matrix_b, out=self.matrix_c)
                    compute_stream.synchronize()
                    iter_end = time.perf_counter()
                    
                    self.iterations += 1
                    
                    # Optional sleep to control intensity
                    if self.sleep_ms > 0:
                        time.sleep(self.sleep_ms / 1000.0)
                    
                    # Periodically report metrics
                    if self.iterations % 100 == 0:
                        elapsed = time.perf_counter() - start_time
                        flops_per_iter = 2 * (self.matrix_size ** 3) * self.batch_size
                        gflops = (flops_per_iter * self.iterations / elapsed) / 1e9
                        self.metrics_queue.put({
                            'timestamp': time.time(),
                            'iterations': self.iterations,
                            'current_gflops': gflops
                        })
                        
            except Exception as e:
                logger.error(f"Error in compute loop: {e}")
                break
        
        self.total_time = time.perf_counter() - start_time
    
    def get_current_metrics(self) -> Optional[Dict]:
        """Get the most recent metrics without stopping."""
        try:
            # Get all available metrics and return the most recent
            latest_metrics = None
            while not self.metrics_queue.empty():
                latest_metrics = self.metrics_queue.get_nowait()
            return latest_metrics
        except:
            return None


class SystemMonitor:
    """Enhanced monitor for CPU, RAM, and GPU usage during transfers."""
    
    def __init__(self, sampling_interval: float = 0.1, gpu_id: int = 0):
        self.sampling_interval = sampling_interval
        self.gpu_id = gpu_id
        self.monitoring = False
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_memory_usage = []
        self.timestamps = []
        from threading import Thread
        self.monitor_thread = None
        
        # Initialize NVML for GPU monitoring
        try:
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.gpu_monitoring_available = True
        except:
            self.gpu_monitoring_available = False
            logger.warning(f"GPU monitoring not available for GPU {gpu_id}")
    
    def start_monitoring(self):
        """Start monitoring system resources."""
        from threading import Thread
        self.monitoring = True
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_memory_usage = []
        self.timestamps = []
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return average usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        results = {
            'avg_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'max_cpu_percent': np.max(self.cpu_usage) if self.cpu_usage else 0,
            'avg_ram_percent': np.mean(self.ram_usage) if self.ram_usage else 0,
            'max_ram_percent': np.max(self.ram_usage) if self.ram_usage else 0,
            'avg_ram_gb': np.mean([r * psutil.virtual_memory().total / (100 * 1024**3) for r in self.ram_usage]) if self.ram_usage else 0
        }
        
        if self.gpu_monitoring_available and self.gpu_usage:
            results.update({
                'avg_gpu_percent': np.mean(self.gpu_usage),
                'max_gpu_percent': np.max(self.gpu_usage),
                'avg_gpu_memory_percent': np.mean(self.gpu_memory_usage),
                'max_gpu_memory_percent': np.max(self.gpu_memory_usage)
            })
        
        return results
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                # CPU and RAM monitoring
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                self.cpu_usage.append(cpu_percent)
                self.ram_usage.append(ram_percent)
                
                # GPU monitoring
                if self.gpu_monitoring_available:
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        
                        self.gpu_usage.append(utilization.gpu)
                        gpu_mem_percent = (mem_info.used / mem_info.total) * 100
                        self.gpu_memory_usage.append(gpu_mem_percent)
                    except:
                        pass
                
                self.timestamps.append(time.time())
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                break


def run_single_config_benchmark_with_compute(test_file_path: str, transfer_sizes: List[int], 
                                           compat_mode: str, thread_count: int,
                                           num_gpus: int, file_size: int, 
                                           compute_scenario: str, result_queue: Queue):
    """
    Enhanced benchmark for a single configuration with compute interference testing.
    """
    # Set environment variables BEFORE importing kvikio
    os.environ['KVIKIO_COMPAT_MODE'] = compat_mode
    os.environ['KVIKIO_NTHREADS'] = str(thread_count)
    logger.info(f"=== CONFIG: COMPAT_MODE={compat_mode}, NTHREADS={thread_count}, COMPUTE={compute_scenario} ===")
    
    # Import kvikio after setting environment variables
    try:
        import kvikio
        logger.info(f"Successfully imported kvikio with COMPAT_MODE={compat_mode}, NTHREADS={thread_count}")
    except ImportError as e:
        logger.error(f"Import failed for config {compat_mode}/{thread_count}: {e}")
        result_queue.put((compat_mode, thread_count, compute_scenario, {}, {}, {}, {}, {}))
        return
    
    # Initialize results storage
    all_gpu_results = {}
    all_cpu_results = {}
    all_ram_results = {}
    all_gds_status = {}
    all_compute_results = {}
    
    # Create GPU worker processes
    gpu_queue = Queue()
    gpu_processes = []
    
    for gpu_id in range(num_gpus):
        p = Process(
            target=gpu_worker_process_with_compute,
            args=(gpu_id, test_file_path, transfer_sizes, 
                  compat_mode, thread_count, file_size, compute_scenario, gpu_queue)
        )
        p.start()
        gpu_processes.append(p)
    
    # Wait for completion
    for p in gpu_processes:
        p.join()
    
    # Collect results
    while not gpu_queue.empty():
        gpu_id, gpu_results, cpu_results, ram_results, gds_enabled, compute_metrics = gpu_queue.get()
        all_gpu_results[gpu_id] = gpu_results
        all_cpu_results[gpu_id] = cpu_results
        all_ram_results[gpu_id] = ram_results
        all_gds_status[gpu_id] = gds_enabled
        all_compute_results[gpu_id] = compute_metrics
        
        logger.info(f"CONFIG {compat_mode}/{thread_count}T/{compute_scenario}: "
                   f"Collected GPU {gpu_id} results, GDS={'ENABLED' if gds_enabled else 'DISABLED'}")
    
    # Send results back
    result_queue.put((compat_mode, thread_count, compute_scenario, 
                     all_gpu_results, all_cpu_results, all_ram_results, 
                     all_gds_status, all_compute_results))
    logger.info(f"=== CONFIG {compat_mode}/{thread_count}T/{compute_scenario} COMPLETED ===")


def gpu_worker_process_with_compute(gpu_id: int, test_file_path: str, transfer_sizes: List[int], 
                                   compat_mode: str, thread_count: int, file_size: int, 
                                   compute_scenario: str, result_queue: Queue):
    """
    Enhanced GPU worker that can run transfers with concurrent compute workloads.
    """
    # Import kvikio in this process
    try:
        import kvikio
        logger.info(f"GPU {gpu_id} worker: imported kvikio")
    except ImportError as e:
        logger.error(f"GPU {gpu_id} worker: kvikio import failed: {e}")
        result_queue.put((gpu_id, {}, {}, {}, False, {}))
        return
    
    # Isolate GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    
    logger.info(f"GPU {gpu_id} worker started for {compat_mode}/{thread_count}T/{compute_scenario}")
    
    # Results storage
    gpu_results = []
    cpu_usage_results = {}
    ram_usage_results = {}
    compute_metrics_results = {}
    
    # Test GDS capability
    gds_enabled = False
    try:
        for test_size in [1024, 4096, 16384, 65536]:
            test_buffer = torch.empty(test_size // 4, dtype=torch.float32, device=device)
            try:
                with kvikio.CuFile(test_file_path, "rb") as f:
                    future = f.pread(test_buffer, test_size, 0)
                    bytes_read = future.get()
                    if bytes_read == test_size:
                        gds_enabled = True
                        logger.info(f"GPU {gpu_id}: GDS ENABLED")
                        break
            except:
                continue
            finally:
                del test_buffer
    except Exception as e:
        logger.error(f"GPU {gpu_id}: GDS test exception: {e}")
    
    # Initialize compute worker if needed
    compute_worker = None
    if compute_scenario != 'transfer_only':
        intensity = compute_scenario.replace('transfer_', '').replace('_compute', '')
        if intensity not in ['light', 'medium', 'heavy']:
            intensity = 'medium'
        
        try:
            compute_worker = GPUComputeWorker(device, compute_intensity=intensity)
            logger.info(f"GPU {gpu_id}: Initialized {intensity} compute worker")
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Failed to initialize compute worker: {e}")
            compute_worker = None
    
    # Test each transfer size
    for transfer_size in transfer_sizes:
        size_str = _format_size(transfer_size)
        
        try:
            # Memory check
            torch.cuda.empty_cache()
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            
            # Reserve memory for compute if needed
            compute_memory_reserve = 0
            if compute_worker:
                # Estimate memory needed for compute matrices
                matrix_memory = compute_worker.batch_size * compute_worker.matrix_size * compute_worker.matrix_size * 4 * 3
                compute_memory_reserve = matrix_memory
            
            if transfer_size + compute_memory_reserve > free_mem * 0.8:
                logger.warning(f"GPU {gpu_id}: Transfer size {size_str} exceeds available memory")
                gpu_results.append(0.0)
                compute_metrics_results[transfer_size // (1024**2)] = {'avg_gflops': 0}
                continue
            
            # Allocate GPU buffer for transfers
            gpu_buffer = torch.empty(transfer_size // 4, dtype=torch.float32, device=device)
            
            # Start system monitoring
            monitor = SystemMonitor(sampling_interval=0.05, gpu_id=0)
            monitor.start_monitoring()
            
            # Start compute workload if applicable
            if compute_worker and compute_scenario.startswith('transfer_'):
                compute_worker.start()
                time.sleep(0.1)  # Let compute stabilize
            
            # Performance measurement
            times = []
            successful_transfers = 0
            num_iterations = 5
            align = 1024 * 1024  # 1MB alignment
            
            # Warmup
            for warmup_iter in range(2):
                try:
                    if gds_enabled:
                        with kvikio.CuFile(test_file_path, "rb") as f:
                            future = f.pread(gpu_buffer, transfer_size, 0)
                            bytes_read = future.get()
                    else:
                        with open(test_file_path, "rb") as f:
                            data = f.read(transfer_size)
                            if len(data) == transfer_size:
                                host_array = np.frombuffer(data, dtype=np.uint8)
                                host_tensor = torch.from_numpy(host_array).float()
                                copy_size = min(len(host_tensor), len(gpu_buffer))
                                gpu_buffer[:copy_size] = host_tensor[:copy_size].to(device)
                    torch.cuda.synchronize()
                except:
                    continue
            
            # Actual measurements
            for i in range(num_iterations):
                try:
                    # Random offset
                    max_offset = file_size - transfer_size
                    offset = np.random.randint(0, max_offset // align) * align
                    
                    start_time = time.perf_counter()
                    
                    if gds_enabled:
                        with kvikio.CuFile(test_file_path, "rb") as f:
                            future = f.pread(gpu_buffer, transfer_size, offset)
                            bytes_read = future.get()
                    else:
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
            
            # Stop compute workload and get metrics
            compute_metrics = {'avg_gflops': 0}
            if compute_worker and compute_scenario.startswith('transfer_'):
                compute_metrics = compute_worker.stop()
            
            # Stop monitoring
            system_stats = monitor.stop_monitoring()
            
            # Run compute-only scenario if needed
            if compute_scenario == 'compute_only' and compute_worker:
                logger.info(f"GPU {gpu_id}: Running compute-only workload for {size_str} duration")
                avg_transfer_time = np.mean(times) if times else 1.0
                compute_duration = avg_transfer_time * num_iterations
                
                compute_worker.start()
                time.sleep(compute_duration)
                compute_metrics = compute_worker.stop()
            
            # Calculate bandwidth
            if times:
                avg_time = np.mean(times)
                bandwidth_gbps = (transfer_size / avg_time) / (1024**3)
                mode_str = "GDS" if gds_enabled else "Host"
                logger.info(f"GPU {gpu_id}: {size_str}, {compute_scenario}: "
                          f"{bandwidth_gbps:.2f} GB/s ({mode_str}), "
                          f"{compute_metrics.get('avg_gflops', 0):.2f} GFLOPS")
                gpu_results.append(bandwidth_gbps)
            else:
                logger.error(f"GPU {gpu_id}: No successful transfers for {size_str}")
                gpu_results.append(0.0)
            
            # Store metrics
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
            compute_metrics_results[size_mb] = compute_metrics
            
            # Cleanup
            del gpu_buffer
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Test failed for {size_str}: {e}")
            gpu_results.append(0.0)
            
            size_mb = transfer_size // (1024**2)
            cpu_usage_results[size_mb] = {'avg': 0, 'max': 0}
            ram_usage_results[size_mb] = {'avg_percent': 0, 'max_percent': 0, 'avg_gb': 0}
            compute_metrics_results[size_mb] = {'avg_gflops': 0}
    
    # Cleanup compute worker
    if compute_worker:
        del compute_worker
    
    result_queue.put((gpu_id, gpu_results, cpu_usage_results, ram_usage_results, 
                     gds_enabled, compute_metrics_results))
    logger.info(f"GPU {gpu_id}: Worker completed for {compute_scenario}")


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


class EnhancedKVCacheBandwidthTester:
    """Enhanced tester with compute interference measurement capabilities."""
    
    def __init__(self, test_file_path: str = "/mnt/kvcache/huge_test_4tb.bin"):
        self.test_file_path = test_file_path
        
        # Initialize NVML
        pynvml.nvmlInit()
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        
        # Transfer sizes from 256KB to 16GB
        self.transfer_sizes = [2**i for i in range(18, 35)]
        
        # Thread counts to test
        self.thread_counts = [1, 2, 4, 8, 16, 32, 64]
        
        # KVIKIO compatibility modes
        self.compat_modes = ['OFF', 'ON']
        
        # Compute scenarios
        self.compute_scenarios = [
            'transfer_only',      # Baseline: transfers without compute
            'compute_only',       # Baseline: compute without transfers
            'transfer_light_compute',   # Transfers with light compute load
            'transfer_medium_compute',  # Transfers with medium compute load
            'transfer_heavy_compute'    # Transfers with heavy compute load
        ]
        
        # Verify test file
        if not os.path.exists(self.test_file_path):
            raise FileNotFoundError(f"Test file not found: {self.test_file_path}")
        
        self.file_size = os.path.getsize(self.test_file_path)
        logger.info(f"Test file size: {self.file_size / (1024**4):.2f} TB")
        
        # Enhanced results structure
        self.results = {
            'transfer_sizes': self.transfer_sizes,
            'thread_counts': self.thread_counts,
            'compat_modes': self.compat_modes,
            'compute_scenarios': self.compute_scenarios,
            'gpu_bandwidths': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))),
            'compute_performance': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))),
            'aggregate_bandwidth': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
            'cpu_usage': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))),
            'ram_usage': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))),
            'system_info': self._get_system_info(),
            'gds_status': {}
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
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'num_gpus': self.num_gpus
        }
    
    def run_benchmark(self, test_scenarios: Optional[List[str]] = None) -> Dict:
        """
        Run the enhanced benchmark with compute interference testing.
        
        Args:
            test_scenarios: List of scenarios to test. If None, tests all scenarios.
        """
        if test_scenarios is None:
            test_scenarios = self.compute_scenarios
        
        logger.info("Starting Enhanced KV Cache bandwidth benchmark with compute interference testing")
        logger.info(f"Transfer sizes: {[_format_size(size) for size in self.transfer_sizes]}")
        logger.info(f"Thread counts: {self.thread_counts}")
        logger.info(f"KVIKIO compatibility modes: {self.compat_modes}")
        logger.info(f"Compute scenarios: {test_scenarios}")
        
        # Use multiprocessing
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        total_configs = len(self.compat_modes) * len(self.thread_counts) * len(test_scenarios)
        logger.info(f"Testing {total_configs} configurations...")
        
        start_time = time.time()
        config_num = 0
        
        # Run configurations sequentially
        for compute_scenario in test_scenarios:
            for compat_mode in self.compat_modes:
                for thread_count in self.thread_counts:
                    config_num += 1
                    logger.info(f"\n=== RUNNING CONFIG {config_num}/{total_configs}: "
                               f"SCENARIO={compute_scenario}, COMPAT_MODE={compat_mode}, NTHREADS={thread_count} ===")
                    
                    result_queue = Queue()
                    
                    # Start configuration process
                    p = Process(
                        target=run_single_config_benchmark_with_compute,
                        args=(self.test_file_path, self.transfer_sizes, 
                              compat_mode, thread_count, self.num_gpus, self.file_size,
                              compute_scenario, result_queue)
                    )
                    p.start()
                    p.join()
                    
                    # Collect results
                    if not result_queue.empty():
                        (compat_mode_result, thread_count_result, scenario_result,
                         all_gpu_results, all_cpu_results, all_ram_results, 
                         all_gds_status, all_compute_results) = result_queue.get()
                        
                        # Store results
                        for gpu_id in all_gpu_results:
                            gpu_data = all_gpu_results[gpu_id]
                            self.results['gpu_bandwidths'][scenario_result][compat_mode_result][thread_count_result][gpu_id] = gpu_data
                            
                            # Store compute metrics
                            if gpu_id in all_compute_results:
                                self.results['compute_performance'][scenario_result][compat_mode_result][thread_count_result][gpu_id] = all_compute_results[gpu_id]
                        
                        # Store system usage
                        for gpu_id in all_cpu_results:
                            self.results['cpu_usage'][scenario_result][compat_mode_result][thread_count_result].update(all_cpu_results[gpu_id])
                        
                        for gpu_id in all_ram_results:
                            self.results['ram_usage'][scenario_result][compat_mode_result][thread_count_result].update(all_ram_results[gpu_id])
                        
                        # Store GDS status
                        for gpu_id in all_gds_status:
                            if gpu_id not in self.results['gds_status']:
                                self.results['gds_status'][gpu_id] = {}
                            self.results['gds_status'][gpu_id][compat_mode_result] = all_gds_status[gpu_id]
                    
                    # Progress update
                    elapsed_time = time.time() - start_time
                    if config_num > 0:
                        avg_time_per_config = elapsed_time / config_num
                        estimated_remaining = avg_time_per_config * (total_configs - config_num)
                        logger.info(f"Progress: {config_num}/{total_configs} completed. "
                                   f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {estimated_remaining:.1f}s")
        
        # Calculate aggregate bandwidths
        for scenario in test_scenarios:
            for compat_mode in self.compat_modes:
                for thread_count in self.thread_counts:
                    aggregate_bandwidths = []
                    for i in range(len(self.transfer_sizes)):
                        total_bw = 0
                        for gpu_id in range(self.num_gpus):
                            if (gpu_id in self.results['gpu_bandwidths'][scenario][compat_mode][thread_count] and 
                                i < len(self.results['gpu_bandwidths'][scenario][compat_mode][thread_count][gpu_id])):
                                total_bw += self.results['gpu_bandwidths'][scenario][compat_mode][thread_count][gpu_id][i]
                        aggregate_bandwidths.append(total_bw)
                    self.results['aggregate_bandwidth'][scenario][compat_mode][thread_count] = aggregate_bandwidths
        
        total_time = time.time() - start_time
        logger.info(f"\nBenchmark completed in {total_time:.1f} seconds")
        
        return self.results
    
    def visualize_interference_results(self, save_path: str = "kv_cache_interference_results.png"):
        """Create comprehensive visualization of interference results."""
        # Convert transfer sizes to MB
        transfer_sizes_mb = [size / (1024**2) for size in self.transfer_sizes]
        
        # Create figure
        fig = plt.figure(figsize=(24, 20))
        sns.set_style("whitegrid")
        
        # Define colors
        scenario_colors = {
            'transfer_only': 'blue',
            'compute_only': 'gray',
            'transfer_light_compute': 'green',
            'transfer_medium_compute': 'orange',
            'transfer_heavy_compute': 'red'
        }
        
        # 1. Transfer Performance Impact by Compute Load
        ax1 = plt.subplot(3, 3, 1)
        
        # Find best configuration
        best_config = self._find_best_config_for_transfers()
        compat_mode, thread_count = best_config['compat_mode'], best_config['thread_count']
        
        for scenario in ['transfer_only', 'transfer_light_compute', 'transfer_medium_compute', 'transfer_heavy_compute']:
            if scenario in self.results['aggregate_bandwidth']:
                bandwidths = self.results['aggregate_bandwidth'][scenario][compat_mode][thread_count]
                if bandwidths:
                    ax1.plot(transfer_sizes_mb, bandwidths, 'o-', 
                            label=scenario.replace('transfer_', '').replace('_compute', ''),
                            color=scenario_colors[scenario], linewidth=2, markersize=4)
        
        ax1.set_title(f'Transfer Performance vs Compute Load\n(COMPAT={compat_mode}, {thread_count}T)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Transfer Size (MB)')
        ax1.set_ylabel('Aggregate Bandwidth (GB/s)')
        ax1.set_xscale('log', base=2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bandwidth Degradation Percentage
        ax2 = plt.subplot(3, 3, 2)
        
        # Calculate degradation for each compute scenario
        baseline_bw = self.results['aggregate_bandwidth']['transfer_only'][compat_mode][thread_count]
        
        for scenario in ['transfer_light_compute', 'transfer_medium_compute', 'transfer_heavy_compute']:
            if scenario in self.results['aggregate_bandwidth']:
                scenario_bw = self.results['aggregate_bandwidth'][scenario][compat_mode][thread_count]
                if scenario_bw and baseline_bw:
                    degradation = [(1 - s/b) * 100 if b > 0 else 0 
                                  for s, b in zip(scenario_bw, baseline_bw)]
                    ax2.plot(transfer_sizes_mb, degradation, 'o-',
                            label=scenario.replace('transfer_', '').replace('_compute', ''),
                            color=scenario_colors[scenario], linewidth=2, markersize=4)
        
        ax2.set_title('Bandwidth Degradation due to Compute Load', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Transfer Size (MB)')
        ax2.set_ylabel('Bandwidth Degradation (%)')
        ax2.set_xscale('log', base=2)
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Compute Performance Under Transfer Load
        ax3 = plt.subplot(3, 3, 3)
        
        # Get compute performance for different scenarios
        compute_data = {}
        for scenario in self.compute_scenarios:
            if 'compute' in scenario:
                total_gflops = 0
                gpu_count = 0
                
                for gpu_id in range(self.num_gpus):
                    if gpu_id in self.results['compute_performance'][scenario][compat_mode][thread_count]:
                        # Average across all transfer sizes
                        gpu_compute = self.results['compute_performance'][scenario][compat_mode][thread_count][gpu_id]
                        avg_gflops = np.mean([m.get('avg_gflops', 0) for m in gpu_compute.values()])
                        if avg_gflops > 0:
                            total_gflops += avg_gflops
                            gpu_count += 1
                
                if gpu_count > 0:
                    compute_data[scenario] = total_gflops
        
        if compute_data:
            scenarios = list(compute_data.keys())
            gflops = list(compute_data.values())
            colors = [scenario_colors.get(s, 'gray') for s in scenarios]
            
            bars = ax3.bar(range(len(scenarios)), gflops, color=colors)
            ax3.set_xticks(range(len(scenarios)))
            ax3.set_xticklabels([s.replace('transfer_', '').replace('_compute', '') for s in scenarios], 
                               rotation=45, ha='right')
            ax3.set_ylabel('Aggregate GFLOPS')
            ax3.set_title('Compute Performance by Scenario', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, gflops):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.0f}', ha='center', va='bottom')
        
        # 4. Per-GPU Performance Heatmap
        ax4 = plt.subplot(3, 3, 4)
        
        # Create heatmap data for transfer performance
        gpu_perf_matrix = np.zeros((self.num_gpus, len(self.compute_scenarios)))
        
        for j, scenario in enumerate(self.compute_scenarios):
            if 'transfer' in scenario:
                for i in range(self.num_gpus):
                    if i in self.results['gpu_bandwidths'][scenario][compat_mode][thread_count]:
                        gpu_bw = self.results['gpu_bandwidths'][scenario][compat_mode][thread_count][i]
                        if gpu_bw:
                            gpu_perf_matrix[i, j] = max(gpu_bw)
        
        im = ax4.imshow(gpu_perf_matrix, aspect='auto', cmap='viridis')
        ax4.set_title('Peak Transfer Performance by GPU and Scenario', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Scenario')
        ax4.set_ylabel('GPU ID')
        ax4.set_xticks(range(len(self.compute_scenarios)))
        ax4.set_xticklabels([s.replace('transfer_', '').replace('_compute', '') 
                            for s in self.compute_scenarios], rotation=45, ha='right')
        ax4.set_yticks(range(self.num_gpus))
        plt.colorbar(im, ax=ax4, label='Bandwidth (GB/s)')
        
        # 5. CPU Usage Comparison
        ax5 = plt.subplot(3, 3, 5)
        
        for scenario in ['transfer_only', 'transfer_heavy_compute']:
            if scenario in self.results['cpu_usage']:
                cpu_by_size = []
                for transfer_size in self.transfer_sizes:
                    size_mb = transfer_size // (1024**2)
                    if size_mb in self.results['cpu_usage'][scenario][compat_mode][thread_count]:
                        cpu_val = self.results['cpu_usage'][scenario][compat_mode][thread_count][size_mb]['avg']
                        cpu_by_size.append(cpu_val)
                    else:
                        cpu_by_size.append(0)
                
                ax5.plot(transfer_sizes_mb, cpu_by_size, 'o-',
                        label=scenario.replace('transfer_', '').replace('_compute', ''),
                        color=scenario_colors[scenario], linewidth=2, markersize=4)
        
        ax5.set_title('CPU Usage: Transfer Only vs Heavy Compute', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Transfer Size (MB)')
        ax5.set_ylabel('CPU Usage (%)')
        ax5.set_xscale('log', base=2)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Interference Summary Table
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        
        # Calculate summary statistics
        summary_data = []
        baseline_peak = 0
        
        if 'transfer_only' in self.results['aggregate_bandwidth']:
            baseline_bw = self.results['aggregate_bandwidth']['transfer_only'][compat_mode][thread_count]
            if baseline_bw:
                baseline_peak = max(baseline_bw)
        
        for scenario in self.compute_scenarios:
            if scenario in self.results['aggregate_bandwidth']:
                scenario_bw = self.results['aggregate_bandwidth'][scenario][compat_mode][thread_count]
                if scenario_bw:
                    peak_bw = max(scenario_bw)
                    avg_bw = np.mean(scenario_bw)
                    
                    # Calculate degradation
                    if baseline_peak > 0 and 'transfer' in scenario and scenario != 'transfer_only':
                        degradation = (1 - peak_bw / baseline_peak) * 100
                    else:
                        degradation = 0
                    
                    # Get compute performance
                    avg_gflops = 0
                    if scenario in self.results['compute_performance']:
                        gflops_list = []
                        for gpu_id in range(self.num_gpus):
                            if gpu_id in self.results['compute_performance'][scenario][compat_mode][thread_count]:
                                gpu_compute = self.results['compute_performance'][scenario][compat_mode][thread_count][gpu_id]
                                for m in gpu_compute.values():
                                    if 'avg_gflops' in m:
                                        gflops_list.append(m['avg_gflops'])
                        if gflops_list:
                            avg_gflops = np.mean(gflops_list) * self.num_gpus
                    
                    summary_data.append([
                        scenario.replace('transfer_', '').replace('_compute', ''),
                        f'{peak_bw:.1f}',
                        f'{avg_bw:.1f}',
                        f'{degradation:.1f}%' if degradation > 0 else 'N/A',
                        f'{avg_gflops:.0f}' if avg_gflops > 0 else 'N/A'
                    ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Scenario', 'Peak BW\n(GB/s)', 'Avg BW\n(GB/s)', 
                                   'BW Loss', 'GFLOPS'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        ax6.set_title('Performance Summary', fontsize=12, fontweight='bold')
        
        # 7. Transfer Size Impact on Interference
        ax7 = plt.subplot(3, 3, 7)
        
        # Calculate interference ratio for different transfer sizes
        size_categories = {
            'Small (≤1MB)': lambda s: s <= 1,
            'Medium (1MB-256MB)': lambda s: 1 < s <= 256,
            'Large (>256MB)': lambda s: s > 256
        }
        
        interference_by_category = {}
        
        for category, size_filter in size_categories.items():
            interferences = []
            
            for i, size in enumerate(self.transfer_sizes):
                size_mb = size // (1024**2)
                if size_filter(size_mb):
                    baseline = self.results['aggregate_bandwidth']['transfer_only'][compat_mode][thread_count][i]
                    heavy = self.results['aggregate_bandwidth']['transfer_heavy_compute'][compat_mode][thread_count][i]
                    
                    if baseline > 0:
                        interference = (1 - heavy / baseline) * 100
                        interferences.append(interference)
            
            if interferences:
                interference_by_category[category] = np.mean(interferences)
        
        if interference_by_category:
            categories = list(interference_by_category.keys())
            interferences = list(interference_by_category.values())
            
            bars = ax7.bar(range(len(categories)), interferences, color=['lightblue', 'orange', 'darkred'])
            ax7.set_xticks(range(len(categories)))
            ax7.set_xticklabels(categories, rotation=45, ha='right')
            ax7.set_ylabel('Average Bandwidth Loss (%)')
            ax7.set_title('Interference by Transfer Size Category', fontsize=12, fontweight='bold')
            ax7.set_ylim(0, 100)
            ax7.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, interferences):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}%', ha='center', va='bottom')
        
        # 8. COMPAT_MODE Comparison Under Load
        ax8 = plt.subplot(3, 3, 8)
        
        # Compare COMPAT_MODE performance under heavy compute load
        for compat in self.compat_modes:
            if 'transfer_heavy_compute' in self.results['aggregate_bandwidth']:
                bw = self.results['aggregate_bandwidth']['transfer_heavy_compute'][compat][thread_count]
                if bw:
                    ax8.plot(transfer_sizes_mb, bw, 'o-',
                            label=f'COMPAT_MODE={compat}',
                            linewidth=2, markersize=4)
        
        ax8.set_title(f'COMPAT_MODE Performance Under Heavy Compute\n({thread_count} threads)',
                     fontsize=12, fontweight='bold')
        ax8.set_xlabel('Transfer Size (MB)')
        ax8.set_ylabel('Aggregate Bandwidth (GB/s)')
        ax8.set_xscale('log', base=2)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Recommendations
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Generate recommendations based on results
        recommendations = self._generate_interference_recommendations()
        
        rec_text = "INTERFERENCE ANALYSIS RECOMMENDATIONS\n\n"
        for i, rec in enumerate(recommendations, 1):
            rec_text += f"{i}. {rec}\n\n"
        
        ax9.text(0.05, 0.95, rec_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Interference results saved to {save_path}")
        plt.show()
    
    def _find_best_config_for_transfers(self) -> Dict:
        """Find the best configuration for transfer performance."""
        best_config = {'compat_mode': 'OFF', 'thread_count': 1, 'peak_bw': 0}
        
        for compat_mode in self.compat_modes:
            for thread_count in self.thread_counts:
                if 'transfer_only' in self.results['aggregate_bandwidth']:
                    bw = self.results['aggregate_bandwidth']['transfer_only'][compat_mode][thread_count]
                    if bw:
                        peak = max(bw)
                        if peak > best_config['peak_bw']:
                            best_config = {
                                'compat_mode': compat_mode,
                                'thread_count': thread_count,
                                'peak_bw': peak
                            }
        
        return best_config
    
    def _generate_interference_recommendations(self) -> List[str]:
        """Generate recommendations based on interference analysis."""
        recommendations = []
        
        # Find best config
        best_config = self._find_best_config_for_transfers()
        compat_mode = best_config['compat_mode']
        thread_count = best_config['thread_count']
        
        # Calculate average degradation
        if ('transfer_only' in self.results['aggregate_bandwidth'] and 
            'transfer_heavy_compute' in self.results['aggregate_bandwidth']):
            
            baseline = self.results['aggregate_bandwidth']['transfer_only'][compat_mode][thread_count]
            heavy = self.results['aggregate_bandwidth']['transfer_heavy_compute'][compat_mode][thread_count]
            
            if baseline and heavy:
                degradations = [(1 - h/b) * 100 for h, b in zip(heavy, baseline) if b > 0]
                avg_degradation = np.mean(degradations)
                
                if avg_degradation < 10:
                    recommendations.append("Excellent isolation: Heavy compute workloads cause <10% bandwidth degradation")
                elif avg_degradation < 25:
                    recommendations.append("Good isolation: Heavy compute causes moderate (10-25%) bandwidth degradation")
                else:
                    recommendations.append(f"Poor isolation: Heavy compute causes {avg_degradation:.0f}% bandwidth degradation")
                    recommendations.append("Consider using separate GPUs for compute and transfers if possible")
        
        # Check if GDS helps with interference
        gds_enabled = any(self.results['gds_status'].get(gpu_id, {}).get(compat_mode, False) 
                         for gpu_id in range(self.num_gpus))
        
        if gds_enabled:
            recommendations.append("GDS is enabled - this should reduce CPU overhead and improve isolation")
        else:
            recommendations.append("Enable GDS for better CPU offload and potentially better isolation")
        
        # Size-specific recommendations
        small_degradation = []
        large_degradation = []
        
        for i, size in enumerate(self.transfer_sizes):
            size_mb = size // (1024**2)
            if size_mb <= 1:  # Small transfers
                if i < len(baseline) and i < len(heavy) and baseline[i] > 0:
                    small_degradation.append((1 - heavy[i]/baseline[i]) * 100)
            elif size_mb >= 256:  # Large transfers
                if i < len(baseline) and i < len(heavy) and baseline[i] > 0:
                    large_degradation.append((1 - heavy[i]/baseline[i]) * 100)
        
        if small_degradation and large_degradation:
            avg_small = np.mean(small_degradation)
            avg_large = np.mean(large_degradation)
            
            if avg_small > avg_large + 10:
                recommendations.append("Small transfers are more affected by compute interference")
                recommendations.append("Consider batching small transfers when running compute workloads")
            elif avg_large > avg_small + 10:
                recommendations.append("Large transfers are more affected by compute interference")
                recommendations.append("Consider scheduling large transfers during compute idle periods")
        
        # Thread count recommendation
        recommendations.append(f"Optimal thread count for transfers: {thread_count}")
        
        return recommendations
    
    def print_interference_summary(self):
        """Print a summary of interference results."""
        print("\n" + "="*80)
        print("COMPUTE INTERFERENCE ANALYSIS SUMMARY")
        print("="*80)
        
        # Find best config
        best_config = self._find_best_config_for_transfers()
        compat_mode = best_config['compat_mode']
        thread_count = best_config['thread_count']
        
        print(f"\nBest Configuration: COMPAT_MODE={compat_mode}, {thread_count} threads")
        print(f"Peak Bandwidth (no compute): {best_config['peak_bw']:.2f} GB/s")
        
        # Performance under different compute loads
        print("\nPERFORMANCE IMPACT BY COMPUTE LOAD:")
        print(f"{'Scenario':<25} {'Peak BW (GB/s)':<15} {'Avg BW (GB/s)':<15} {'Degradation':<15} {'Compute (GFLOPS)':<20}")
        print("-" * 90)
        
        baseline_peak = 0
        baseline_avg = 0
        
        for scenario in self.compute_scenarios:
            if scenario in self.results['aggregate_bandwidth']:
                bw = self.results['aggregate_bandwidth'][scenario][compat_mode][thread_count]
                if bw:
                    peak_bw = max(bw)
                    avg_bw = np.mean(bw)
                    
                    if scenario == 'transfer_only':
                        baseline_peak = peak_bw
                        baseline_avg = avg_bw
                        degradation_str = "baseline"
                    elif baseline_peak > 0 and 'transfer' in scenario:
                        degradation = (1 - peak_bw / baseline_peak) * 100
                        degradation_str = f"{degradation:.1f}%"
                    else:
                        degradation_str = "N/A"
                    
                    # Get compute performance
                    total_gflops = 0
                    if scenario in self.results['compute_performance']:
                        for gpu_id in range(self.num_gpus):
                            if gpu_id in self.results['compute_performance'][scenario][compat_mode][thread_count]:
                                gpu_compute = self.results['compute_performance'][scenario][compat_mode][thread_count][gpu_id]
                                for metrics in gpu_compute.values():
                                    if 'avg_gflops' in metrics:
                                        total_gflops += metrics['avg_gflops']
                    
                    compute_str = f"{total_gflops:.1f}" if total_gflops > 0 else "N/A"
                    
                    scenario_name = scenario.replace('transfer_', '').replace('_compute', '')
                    print(f"{scenario_name:<25} {peak_bw:<15.2f} {avg_bw:<15.2f} {degradation_str:<15} {compute_str:<20}")
        
        print("\n" + "="*80)
    
    def save_enhanced_results(self):
        """Save enhanced results including compute interference data."""
        # Convert to JSON-serializable format
        json_results = {
            'transfer_sizes': [int(x) for x in self.results['transfer_sizes']],
            'thread_counts': self.results['thread_counts'],
            'compat_modes': self.results['compat_modes'],
            'compute_scenarios': self.results['compute_scenarios'],
            'system_info': self.results['system_info'],
            'gds_status': self.results['gds_status']
        }
        
        # Convert nested defaultdicts to regular dicts
        for key in ['gpu_bandwidths', 'compute_performance', 'aggregate_bandwidth', 'cpu_usage', 'ram_usage']:
            json_results[key] = {}
            for scenario in self.results[key]:
                json_results[key][scenario] = {}
                for compat_mode in self.results[key][scenario]:
                    json_results[key][scenario][compat_mode] = {}
                    for thread_count in self.results[key][scenario][compat_mode]:
                        json_results[key][scenario][compat_mode][str(thread_count)] = dict(self.results[key][scenario][compat_mode][thread_count])
        
        with open("kv_cache_interference_benchmark_results.json", "w") as f:
            json.dump(json_results, f, indent=2)
        
        logger.info("Enhanced results saved to kv_cache_interference_benchmark_results.json")


def main():
    """Main function to run the enhanced benchmark."""
    test_file = "/mnt/kvcache/huge_test_4tb.bin"
    
    try:
        # Check PyTorch CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in PyTorch")
        
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        # Initialize enhanced tester
        tester = EnhancedKVCacheBandwidthTester(test_file)
        
        # Run benchmark with all scenarios (or specify subset)
        # For testing, you might want to start with just a few scenarios:
        # results = tester.run_benchmark(['transfer_only', 'transfer_heavy_compute'])
        
        # Run full benchmark
        results = tester.run_benchmark()
        
        # Generate visualizations and summaries
        tester.visualize_interference_results()
        tester.print_interference_summary()
        tester.save_enhanced_results()
        
        logger.info("Enhanced benchmark completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()