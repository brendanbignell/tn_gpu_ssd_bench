#!/bin/bash

# Drive Technology Comparison Benchmark Script
# Tests 4-drive arrays of different technologies

OUTPUT_FILE="/mnt/drive_comparison_results.csv"
TEST_SIZE="8G"
RUNTIME="30"
NUM_JOBS=6  # Optimal for 4-drive arrays

# Test configurations
declare -A MOUNT_POINTS
MOUNT_POINTS[WD_4TB]="/mnt/wd_test"
MOUNT_POINTS[SKHynix_2TB]="/mnt/skhynix_test"  
MOUNT_POINTS[Intel_Optane_32GB]="/mnt/optane_test"

# Array of block sizes to test (focused on key sizes)
BLOCK_SIZES=("4k" "64k" "1m" "16m" "64m")

# Array of queue depths to test (focused on key depths)
QUEUE_DEPTHS=(1 8 32 128)

# Array of operations to test
OPERATIONS=("read" "write")

echo "=== Drive Technology Comparison Benchmark ==="
echo "Testing 4-drive RAID 0 arrays of different technologies"
echo "Output file: $OUTPUT_FILE"
echo

# Create CSV header
echo "DriveType,Operation,BlockSize,QueueDepth,NumJobs,Bandwidth_MBps,Bandwidth_GBps,IOPS" > "$OUTPUT_FILE"

# Function to run a single test
run_comparison_test() {
    local drive_type=$1
    local operation=$2
    local block_size=$3
    local queue_depth=$4
    local mount_point=${MOUNT_POINTS[$drive_type]}
    local test_file="${mount_point}/test_${operation}_${block_size}_qd${queue_depth}"
    
    echo "Testing: $drive_type, $operation, BS=$block_size, QD=$queue_depth, Jobs=$NUM_JOBS"
    
    # Check if mount point exists and is writable
    if [ ! -d "$mount_point" ] || [ ! -w "$mount_point" ]; then
        echo "  ERROR: $mount_point not accessible"
        echo "$drive_type,$operation,$block_size,$queue_depth,$NUM_JOBS,0,0,0" >> "$OUTPUT_FILE"
        return
    fi
    
    # Run fio test and capture output
    fio_output=$(sudo fio \
        --name=comparison-test \
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
        summary_line=$(echo "$fio_output" | grep -E "^[[:space:]]*READ:|^[[:space:]]*WRITE:" | tail -1)
        
        if [ -n "$summary_line" ]; then
            echo "  Raw output: $summary_line"
            
            # Parse bandwidth - look for patterns like "bw=69.8GiB/s (75.0GB/s)"
            gb_per_sec=$(echo "$summary_line" | grep -oE '\([0-9.]+GB/s\)' | grep -oE '[0-9.]+')
            
            # If that fails, try to parse GiB/s and convert
            if [ -z "$gb_per_sec" ]; then
                gib_per_sec=$(echo "$summary_line" | grep -oE 'bw=[0-9.]+\.?[0-9]*GiB/s' | grep -oE '[0-9.]+\.?[0-9]*')
                if [ -n "$gib_per_sec" ]; then
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
            
            # Extract IOPS from the summary line
            iops=$(echo "$summary_line" | grep -oE 'IOPS=[0-9.]+[kK]?' | grep -oE '[0-9.]+[kK]?')
            if [[ "$iops" == *k* ]] || [[ "$iops" == *K* ]]; then
                iops_num=$(echo "$iops" | grep -oE '[0-9.]+')
                iops=$(echo "scale=0; $iops_num * 1000" | bc -l)
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
            echo "$drive_type,$operation,$block_size,$queue_depth,$NUM_JOBS,$mb_per_sec,$gb_per_sec,$iops" >> "$OUTPUT_FILE"
        else
            echo "  Failed to find summary line"
            echo "$drive_type,$operation,$block_size,$queue_depth,$NUM_JOBS,0,0,0" >> "$OUTPUT_FILE"
        fi
    else
        echo "  Failed to run fio"
        echo "$drive_type,$operation,$block_size,$queue_depth,$NUM_JOBS,0,0,0" >> "$OUTPUT_FILE"
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

echo "Starting drive comparison benchmark..."

# Calculate total tests
total_tests=$((${#MOUNT_POINTS[@]} * ${#OPERATIONS[@]} * ${#BLOCK_SIZES[@]} * ${#QUEUE_DEPTHS[@]}))
echo "Total tests: $total_tests"
echo

test_count=0

# Run all test combinations
for drive_type in "${!MOUNT_POINTS[@]}"; do
    echo "=== Testing $drive_type ==="
    for operation in "${OPERATIONS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            for queue_depth in "${QUEUE_DEPTHS[@]}"; do
                test_count=$((test_count + 1))
                echo "[$test_count/$total_tests]"
                
                run_comparison_test "$drive_type" "$operation" "$block_size" "$queue_depth"
                echo
            done
        done
    done
    echo
done

echo "Drive comparison benchmark complete!"
echo "Results saved to: $OUTPUT_FILE"
echo
echo "Quick analysis:"
echo "# Best read performance by drive type:"
echo "grep '^WD_4TB,read,' $OUTPUT_FILE | sort -t',' -k7 -nr | head -3"
echo "grep '^SKHynix_2TB,read,' $OUTPUT_FILE | sort -t',' -k7 -nr | head -3"  
echo "grep '^Intel_Optane_32GB,read,' $OUTPUT_FILE | sort -t',' -k7 -nr | head -3"
echo
echo "# Best write performance by drive type:"
echo "grep '^WD_4TB,write,' $OUTPUT_FILE | sort -t',' -k7 -nr | head -3"
echo "grep '^SKHynix_2TB,write,' $OUTPUT_FILE | sort -t',' -k7 -nr | head -3"
echo "grep '^Intel_Optane_32GB,write,' $OUTPUT_FILE | sort -t',' -k7 -nr | head -3"