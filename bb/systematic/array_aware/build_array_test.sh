#!/bin/bash

echo "Building array-aware optimization test (fixed)..."
echo "Targeting your 11x NVMe PCIe4 array with 1MB chunks"
echo "Goal: Scale from 7 GB/s to 15-30+ GB/s by utilizing array parallelism"
echo ""

make clean
make

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Running array scaling test..."
    python3 array_scaling_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1
    
    echo ""
    echo "Analysis:"
    echo "• Your single-threaded code: ~7 GB/s (excellent for single thread)"
    echo "• Your 11x NVMe array potential: 30+ GB/s" 
    echo "• Gap to close: 4-5x improvement possible with parallelism"
    echo ""
    echo "If multi-threaded versions show big improvements:"
    echo "  → Your array responds well to parallel access"
    echo "  → Consider scaling to even more readers"
    echo ""
    echo "If single-threaded is still best:"
    echo "  → Your array prefers large sequential access"
    echo "  → Focus on larger block sizes or array tuning"
    
else
    echo "Build failed!"
    exit 1
fi