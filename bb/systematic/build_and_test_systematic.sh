#!/bin/bash

echo "Building fixed systematic optimization test..."
make clean
make

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Running quick test to find best version..."
    python3 systematic_test.py /mnt/kvcache/huge_test_4tb.bin 2.0 1 --quick
    
    echo ""
    echo "To run full test with 3 runs per version:"
    echo "python3 systematic_test.py /mnt/kvcache/huge_test_4tb.bin 5.0 1"
else
    echo "Build failed!"
    exit 1
fi