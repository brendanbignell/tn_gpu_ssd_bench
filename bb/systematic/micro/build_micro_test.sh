#!/bin/bash

echo "Building micro-optimization test..."
echo "Testing tiny tweaks to your already excellent code..."
make clean
make

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Running quick micro-optimization test..."
    python3 micro_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1 --quick
    
    echo ""
    echo "Micro-optimizations being tested:"
    echo "1. High-priority CUDA stream"
    echo "2. Batched io_uring submits"  
    echo "3. 6MB blocks (vs 4MB)"
    echo "4. Non-blocking CUDA stream"
    echo ""
    echo "To run full test with 3 runs:"
    echo "python3 micro_test.py /mnt/kvcache/huge_test_4tb.bin 3.0 1"
else
    echo "Build failed!"
    exit 1
fi