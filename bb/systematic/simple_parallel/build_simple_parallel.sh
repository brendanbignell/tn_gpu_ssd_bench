#!/bin/bash

echo "Building FIXED simple parallel optimization test..."
echo "Using OpenMP for clean parallelism (fixed structured block issues)"
echo ""
echo "Target: Your 11x NVMe PCIe4 array with 1MB chunks"
echo "Goal: Scale from 7 GB/s to 15-30+ GB/s using parallel readers"
echo ""

make clean
make

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Running simple parallel scaling test..."
    python3 simple_parallel_scaling_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1
    
    echo ""
    echo "Test versions:"
    echo "• Single reader: Your excellent baseline"
    echo "• Dual reader: 2 parallel readers for array parallelism"  
    echo "• Quad reader: 4 parallel readers for higher utilization"
    echo "• 1MB aligned: Blocks matching your array chunk size"
    echo "• Multi-FD: 2 file descriptors (fallback if OpenMP fails)"
    echo ""
    echo "Expected results:"
    echo "• 2x+ speedup = Array loves parallel access!"
    echo "• 1.5x+ speedup = Good scaling, try more readers"
    echo "• <1.2x speedup = Array prefers sequential access"
    
else
    echo "Build failed!"
    echo ""
    echo "If this still fails, try the multi-FD version which doesn't use OpenMP:"
    echo "  → The multi_fd_simple function uses sequential dual file descriptors"
    echo "  → Should work even if OpenMP isn't available"
    exit 1
fi