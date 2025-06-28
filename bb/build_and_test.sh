#!/bin/bash

echo "Building ultra-fast I/O library..."
make clean
make

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Running basic test..."
    python3 ultra_fast_test.py /mnt/kvcache/huge_test_4tb.bin --size 1.0 --device 1
    
    echo ""
    echo "To run parameter sweep:"
    echo "python3 ultra_fast_test.py /mnt/kvcache/huge_test_4tb.bin --size 5.0 --device 1 --sweep"
    
    echo ""
    echo "To test custom configuration:"
    echo "python3 ultra_fast_test.py /mnt/kvcache/huge_test_4tb.bin --size 5.0 --device 1 --config 128 16 8 2"
else
    echo "Build failed!"
    exit 1
fi