# build_block_optimization.sh
#!/bin/bash

echo "Building Block Size Optimization Test"
echo "====================================="
echo "Previous results showed:"
echo "• 1MB blocks: 24% improvement (11.32 GB/s)"
echo "• 4MB blocks: 9.09 GB/s (baseline)"
echo "• Parallel approaches: Slower (array prefers sequential)"
echo ""
echo "Testing block sizes: 512KB, 1MB, 2MB, 4MB, 8MB, 16MB"
echo "Goal: Find absolute optimal block size for 30%+ array utilization"
echo ""

make clean
make

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Running block size optimization test..."
    python3 block_size_optimization_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1
    
    echo ""
    echo "This test will:"
    echo "• Confirm 1MB blocks are optimal (matching array 1MB chunks)"
    echo "• Test if larger blocks (8MB, 16MB) help further"
    echo "• Test higher queue depth with optimal block size"
    echo "• Find the absolute best configuration for your array"
    
else
    echo "Build failed!"
    exit 1
fi