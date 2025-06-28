#!/bin/bash

# RAID Performance Benchmark Script
# Generates CSV data for charting performance across block sizes and queue depths

OUTPUT_FILE="/mnt/kvcache/raid_performance_data.csv"
MOUNT_POINT="/mnt/kvcache"
TEST_SIZE="2G"  # Test file size
RUNTIME="15"    # Test duration in seconds

# Array of block sizes to test (in fio format)
BLOCK_SIZES=("4k" "16k" "64k" "256k" "1m" "4m" "16m" "64m")

# Array of queue depths to test
QUEUE_DEPTHS=(1 2 4 8 16 32 64 128)

# Array of operations to test
OPERATIONS=("read" "write")

echo "=== RAID Performance Benchmark ==="
echo "Output file: $OUTPUT_FILE"
echo "Test size: $TEST_SIZE per test"
echo "Runtime: ${RUNTIME}s per test"
echo

# Check if mount point exists and is writable
if [ ! -d "$MOUNT_POINT" ] || [ ! -w "$MOUNT_POINT" ]; then
    echo "Error: $MOUNT_POINT is not accessible or writable"
    exit 1
fi

# Create CSV header
echo "Operation,BlockSize,QueueDepth,Bandwidth_MBps,Bandwidth_GBps,IOPS,AvgLatency_us,P95Latency_us" > "$OUTPUT_FILE"

# Function to run a single fio test
run_fio_test() {
    local operation=$1
    local block_size=$2
    local queue_depth=$3
    local test_file="${MOUNT_POINT}/fio_test_${operation}_${block_size}_qd${queue_depth}"
    
    echo "Testing: $operation, BS=$block_size, QD=$queue_depth"
    
    # Run fio test with machine-readable output
    fio_output=$(sudo fio \
        --name=test \
        --ioengine=libaio \
        --iodepth=$queue_depth \
        --rw=$operation \
        --bs=$block_size \
        --direct=1 \
        --size=$TEST_SIZE \
        --runtime=$RUNTIME \
        --time_based \
        --numjobs=1 \
        --group_reporting \
        --filename="$test_file" \
        --output-format=json \
        2>/dev/null)
    
    # Parse JSON output
    if [ $? -eq 0 ] && [ -n "$fio_output" ]; then
        # Extract bandwidth (in KB/s), IOPS, and latency using basic parsing
        bandwidth_kbps=$(echo "$fio_output" | grep -o '"bw":[0-9]*' | head -1 | cut -d':' -f2)
        iops=$(echo "$fio_output" | grep -o '"iops":[0-9.]*' | head -1 | cut -d':' -f2)
        avg_lat=$(echo "$fio_output" | grep -o '"lat_ns":{"min":[0-9]*,"max":[0-9]*,"mean":[0-9.]*' | cut -d',' -f3 | cut -d':' -f2)
        p95_lat=$(echo "$fio_output" | grep -o '"95.000000":[0-9.]*' | cut -d':' -f2)
        
        # Convert bandwidth from KB/s to MB/s and GB/s
        if [ -n "$bandwidth_kbps" ]; then
            bandwidth_mbps=$(echo "scale=2; $bandwidth_kbps / 1024" | bc -l)
            bandwidth_gbps=$(echo "scale=3; $bandwidth_kbps / 1024 / 1024" | bc -l)
        else
            bandwidth_mbps="0"
            bandwidth_gbps="0"
        fi
        
        # Convert latency from nanoseconds to microseconds
        if [ -n "$avg_lat" ]; then
            avg_lat_us=$(echo "scale=2; $avg_lat / 1000" | bc -l)
        else
            avg_lat_us="0"
        fi
        
        if [ -n "$p95_lat" ]; then
            p95_lat_us=$(echo "scale=2; $p95_lat / 1000" | bc -l)
        else
            p95_lat_us="0"
        fi
        
        # Default values if parsing failed
        [ -z "$iops" ] && iops="0"
        
        # Write to CSV
        echo "$operation,$block_size,$queue_depth,$bandwidth_mbps,$bandwidth_gbps,$iops,$avg_lat_us,$p95_lat_us" >> "$OUTPUT_FILE"
    else
        echo "Failed to run test for $operation, $block_size, QD$queue_depth"
        echo "$operation,$block_size,$queue_depth,0,0,0,0,0" >> "$OUTPUT_FILE"
    fi
    
    # Clean up test file
    rm -f "$test_file"
}

# Function to run simplified fio test (fallback)
run_simple_fio_test() {
    local operation=$1
    local block_size=$2
    local queue_depth=$3
    local test_file="${MOUNT_POINT}/fio_test_${operation}_${block_size}_qd${queue_depth}"
    
    echo "Testing (simple): $operation, BS=$block_size, QD=$queue_depth"
    
    # Run fio test with text output
    fio_output=$(sudo fio \
        --name=test \
        --ioengine=libaio \
        --iodepth=$queue_depth \
        --rw=$operation \
        --bs=$block_size \
        --direct=1 \
        --size=$TEST_SIZE \
        --runtime=$RUNTIME \
        --time_based \
        --numjobs=1 \
        --filename="$test_file" \
        2>/dev/null | grep -E "(BW=|IOPS=|avg=)")
    
    # Parse text output
    bandwidth_str=$(echo "$fio_output" | grep "BW=" | head -1)
    iops_str=$(echo "$fio_output" | grep "IOPS=" | head -1)
    
    # Extract bandwidth (look for GB/s or MB/s)
    if echo "$bandwidth_str" | grep -q "GiB/s"; then
        bandwidth_gbps=$(echo "$bandwidth_str" | grep -o '[0-9.]*GiB/s' | cut -d'G' -f1)
        bandwidth_mbps=$(echo "scale=2; $bandwidth_gbps * 1024" | bc -l)
    elif echo "$bandwidth_str" | grep -q "MiB/s"; then
        bandwidth_mbps=$(echo "$bandwidth_str" | grep -o '[0-9.]*MiB/s' | cut -d'M' -f1)
        bandwidth_gbps=$(echo "scale=3; $bandwidth_mbps / 1024" | bc -l)
    else
        bandwidth_mbps="0"
        bandwidth_gbps="0"
    fi
    
    # Extract IOPS
    if [ -n "$iops_str" ]; then
        iops=$(echo "$iops_str" | grep -o 'IOPS=[0-9.]*[kK]\?' | cut -d'=' -f2 | sed 's/k/*1000/' | sed 's/K/*1000/' | bc -l 2>/dev/null || echo "0")
    else
        iops="0"
    fi
    
    # Write to CSV (simplified - no latency data)
    echo "$operation,$block_size,$queue_depth,$bandwidth_mbps,$bandwidth_gbps,$iops,0,0" >> "$OUTPUT_FILE"
    
    # Clean up test file
    rm -f "$test_file"
}

# Check if bc is available for calculations
if ! command -v bc &> /dev/null; then
    echo "Installing bc for calculations..."
    sudo apt update && sudo apt install -y bc
fi

# Check if fio is available
if ! command -v fio &> /dev/null; then
    echo "Installing fio for disk benchmarking..."
    sudo apt update && sudo apt install -y fio
fi

echo "Starting benchmark tests..."
echo "Total tests: $((${#OPERATIONS[@]} * ${#BLOCK_SIZES[@]} * ${#QUEUE_DEPTHS[@]}))"
echo

test_count=0
total_tests=$((${#OPERATIONS[@]} * ${#BLOCK_SIZES[@]} * ${#QUEUE_DEPTHS[@]}))

# Run all test combinations
for operation in "${OPERATIONS[@]}"; do
    for block_size in "${BLOCK_SIZES[@]}"; do
        for queue_depth in "${QUEUE_DEPTHS[@]}"; do
            test_count=$((test_count + 1))
            echo "[$test_count/$total_tests]"
            
            # Try JSON output first, fall back to simple parsing
            run_simple_fio_test "$operation" "$block_size" "$queue_depth"
            
            # Brief pause between tests
            sleep 1
        done
    done
done

echo
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT_FILE"
echo
echo "CSV format: Operation,BlockSize,QueueDepth,Bandwidth_MBps,Bandwidth_GBps,IOPS,AvgLatency_us,P95Latency_us"
echo
echo "To view results:"
echo "head -20 $OUTPUT_FILE"
echo
echo "Sample data analysis:"
echo "# Best read performance:"
echo "grep '^read,' $OUTPUT_FILE | sort -t',' -k5 -nr | head -5"
echo "# Best write performance:"
echo "grep '^write,' $OUTPUT_FILE | sort -t',' -k5 -nr | head -5"