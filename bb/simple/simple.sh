#!/bin/bash

echo "Building enhanced I/O library (simple version)..."
make clean
make

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Running comparison test..."
    python3 simple_test.py /mnt/kvcache/huge_test_4tb.bin 1.0 1
else
    echo "Build failed!"
    exit 1
fi