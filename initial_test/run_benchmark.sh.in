#!/bin/bash
# KV Cache Bandwidth Benchmark Runner
# Optimized for AMD Threadripper Pro WX3995

# Set environment variables for optimal performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=64
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Kvikio optimizations for COMAT mode
export KVIKIO_BOUNCE_BUFFER_SIZE=16777216
export KVIKIO_GDS_THRESHOLD=1048576
export KVIKIO_THREAD_POOL_NTHREADS=32

# System optimizations
echo 'Setting system optimizations...'
sudo sh -c 'echo mq-deadline > /sys/block/md*/queue/scheduler' 2>/dev/null || true
sudo sh -c 'echo 0 > /sys/block/md*/queue/rotational' 2>/dev/null || true
sudo sh -c 'echo 1024 > /sys/block/md*/queue/nr_requests' 2>/dev/null || true

# Set CPU governor to performance
sudo cpupower frequency-set -g performance 2>/dev/null || echo 'cpupower not available'

# Clear filesystem caches
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo 'Starting KV Cache bandwidth benchmark...'
echo 'Target: AMD Threadripper Pro WX3995 with 11-NVME RAID0'
echo '================================================='

# Run the benchmark
./kv_cache_bandwidth_tester $@

echo 'Benchmark completed. Check results:'
echo '  - kv_cache_bandwidth_results.png (performance graph)'
echo '  - kv_cache_benchmark_results.csv (raw data)'
