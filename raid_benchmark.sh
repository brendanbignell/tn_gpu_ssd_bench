#!/bin/bash

# Simplified RAID Performance Benchmark Script

OUTPUT_FILE="/mnt/kvcache/raid_performance_simple.csv"
MOUNT_POINT="/mnt/kvcache"
TEST_SIZE="8G"
RUNTIME="30"
NUM_JOBS=8  # Use the proven configuration that achieved 54.9 GB/s

# Array of block sizes to test
BLOCK_SIZES=("4k" "16k" "64k" "256k" "1m" "4m" "16m" "64m")

# Array of queue depths to test  
QUEUE_DEPTHS=(1 2 4 8 16 32 64 128)

# Array of operations to test
OPERATIONS=("read" "write")

echo "=== Simplified RAID Performance Benchmark ==="
echo "Using proven configuration: $NUM_JOBS jobs, ${TEST_SIZE} files, ${RUNTIME}s runtime"
echo "Output file: $OUTPUT_FILE"
echo

# Check if mount point exists and is writable
if [ ! -d "$MOUNT_POINT" ] || [ ! -w "$MOUNT_POINT" ]; then
    echo "Error: $MOUNT_POINT is not accessible or writable"
    exit 1
fi

# Create CSV header
echo "Operation,BlockSize,QueueDepth,NumJobs,Bandwidth_MBps,Bandwidth_GBps,IOPS" > "$OUTPUT_FILE"

# Function to run a single test with simplified parsing
run_simple_test() {
    local operation=$1
    local block_size=$2
    local queue_depth=$3
    local test_file="${MOUNT_POINT}/simple_test_${operation}_${block_size}_qd${queue_depth}"
    
    echo "Testing: $operation, BS=$block_size, QD=$queue_depth, Jobs=$NUM_JOBS"
    
    # Run fio test and capture output
    fio_output=$(sudo fio \
        --name=simple-test \
        --ioengine=libaio \
        --iodepth=$queue_depth \
        --rw=$operation \
        --bs=$block_size \
        --direct=1 \
        --size=$TEST_SIZE \
        --runtime=$RUNTIME \
        --time_based \
        --numjobs=$NUM_JOBS \
        --group_reporting \
        --filename="$test_file" \
        2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$fio_output" ]; then
        # Extract the FINAL summary line with total aggregate bandwidth
        # Look for "READ:" or "WRITE:" (uppercase) which is the aggregate summary
        summary_line=$(echo "$fio_output" | grep -E "^[[:space:]]*READ:|^[[:space:]]*WRITE:" | tail -1)
        
        if [ -n "$summary_line" ]; then
            echo "  Raw output: $summary_line"
            
            # Parse bandwidth - look for patterns like "bw=69.8GiB/s (75.0GB/s)"
            # Extract the GB/s value in parentheses first (this is the standard GB/s value)
            gb_per_sec=$(echo "$summary_line" | grep -oE '\([0-9.]+GB/s\)' | grep -oE '[0-9.]+')
            
            # If that fails, try to parse GiB/s and convert
            if [ -z "$gb_per_sec" ]; then
                gib_per_sec=$(echo "$summary_line" | grep -oE 'bw=[0-9.]+\.?[0-9]*GiB/s' | grep -oE '[0-9.]+\.?[0-9]*')
                if [ -n "$gib_per_sec" ]; then
                    # Convert GiB/s to GB/s (1 GiB = 1.073741824 GB)
                    gb_per_sec=$(echo "scale=3; $gib_per_sec * 1.073741824" | bc -l)
                fi
            fi
            
            # If that fails, try MiB/s  
            if [ -z "$gb_per_sec" ]; then
                mib_per_sec=$(echo "$summary_line" | grep -oE 'bw=[0-9.]+\.?[0-9]*MiB/s' | grep -oE '[0-9.]+\.?[0-9]*')
                if [ -n "$mib_per_sec" ]; then
                    gb_per_sec=$(echo "scale=3; $mib_per_sec / 1024 * 1.073741824" | bc -l)
                fi
            fi
            
            # Extract IOPS from the summary line or calculate from bandwidth
            iops=$(echo "$summary_line" | grep -oE 'IOPS=[0-9.]+[kK]?' | grep -oE '[0-9.]+[kK]?')
            if [[ "$iops" == *k* ]] || [[ "$iops" == *K* ]]; then
                iops_num=$(echo "$iops" | grep -oE '[0-9.]+')
                iops=$(echo "scale=0; $iops_num * 1000" | bc -l)
            fi
            
            # If IOPS not found in summary line, calculate from bandwidth and block size
            if [ -z "$iops" ] || [ "$iops" = "0" ]; then
                if [ -n "$gb_per_sec" ] && [ "$gb_per_sec" != "0" ]; then
                    # Convert block size to MB for calculation
                    case "$block_size" in
                        *k|*K)
                            block_size_mb=$(echo "scale=6; ${block_size%[kK]} / 1024" | bc -l)
                            ;;
                        *m|*M)
                            block_size_mb=$(echo "${block_size%[mM]}" | bc -l)
                            ;;
                        *g|*G)
                            block_size_mb=$(echo "scale=0; ${block_size%[gG]} * 1024" | bc -l)
                            ;;
                        *)
                            block_size_mb=$(echo "scale=6; $block_size / 1024 / 1024" | bc -l)
                            ;;
                    esac
                    
                    # Calculate IOPS = (GB/s * 1000 MB/GB) / (block_size_MB)
                    mb_per_sec_calc=$(echo "scale=2; $gb_per_sec * 1000" | bc -l)
                    iops=$(echo "scale=0; $mb_per_sec_calc / $block_size_mb" | bc -l)
                fi
            fi
            
            # Calculate MB/s from GB/s
            if [ -n "$gb_per_sec" ]; then
                mb_per_sec=$(echo "scale=2; $gb_per_sec * 1000" | bc -l)
            else
                mb_per_sec="0"
                gb_per_sec="0"
            fi
            
            # Default values if parsing failed
            [ -z "$iops" ] && iops="0"
            
            echo "  Parsed: ${gb_per_sec} GB/s, ${iops} IOPS"
            
            # Write to CSV
            echo "$operation,$block_size,$queue_depth,$NUM_JOBS,$mb_per_sec,$gb_per_sec,$iops" >> "$OUTPUT_FILE"
        else
            echo "  Failed to find summary line"
            echo "$operation,$block_size,$queue_depth,$NUM_JOBS,0,0,0" >> "$OUTPUT_FILE"
        fi
    else
        echo "  Failed to run fio"
        echo "$operation,$block_size,$queue_depth,$NUM_JOBS,0,0,0" >> "$OUTPUT_FILE"
    fi
    
    # Clean up test file
    rm -f "$test_file"
    
    # Brief pause between tests
    sleep 2
}

# Check if bc is available
if ! command -v bc &> /dev/null; then
    echo "Installing bc for calculations..."
    sudo apt update && sudo apt install -y bc
fi

echo "Starting simplified benchmark tests..."
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
            
            run_simple_test "$operation" "$block_size" "$queue_depth"
            echo
        done
    done
done

echo "Simplified benchmark complete!"
echo "Results saved to: $OUTPUT_FILE"
echo
echo "Quick analysis:"
echo "# Best read performance:"
echo "grep '^read,' $OUTPUT_FILE | sort -t',' -k6 -nr | head -5"
echo "# Best write performance:"
echo "grep '^write,' $OUTPUT_FILE | sort -t',' -k6 -nr | head -5"