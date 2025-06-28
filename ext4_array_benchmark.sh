#!/bin/bash

# Ext4 Array Scaling Benchmark
# Tests 2x2 vs 4-drive arrays using Ext4 filesystem

OUTPUT_FILE="/mnt/array_scaling_ext4_results.csv"
TEST_SIZE="8G"
RUNTIME="30"

# Test configurations for 2-drive arrays (use fewer jobs)
NUM_JOBS_2DRIVE=4

# Test configurations for 4-drive arrays (use more jobs)  
NUM_JOBS_4DRIVE=6

# Array configurations
declare -A ARRAY_CONFIGS
# 2x2 configurations (will be tested in parallel)
ARRAY_CONFIGS[WD_2x2_Parallel]="/mnt/wd_array1_ext4,/mnt/wd_array2_ext4"
ARRAY_CONFIGS[SKHynix_2x2_Parallel]="/mnt/sk_array1_ext4,/mnt/sk_array2_ext4"

# 4-drive configurations (single arrays)
ARRAY_CONFIGS[WD_4Drive_Single]="/mnt/wd_4drive_ext4"
ARRAY_CONFIGS[SKHynix_4Drive_Single]="/mnt/sk_4drive_ext4"

# Block sizes to test (focused on key sizes)
BLOCK_SIZES=("4k" "64k" "1m" "16m" "64m")

# Queue depths to test
QUEUE_DEPTHS=(1 8 32 128)

# Operations to test
OPERATIONS=("read" "write")

echo "=== Ext4 Array Scaling Benchmark ==="
echo "Output file: $OUTPUT_FILE"
echo

# Create CSV header
echo "ArrayConfig,Operation,BlockSize,QueueDepth,NumJobs,Bandwidth_MBps,Bandwidth_GBps,IOPS,ArrayType,Filesystem" > "$OUTPUT_FILE"

# Function to convert block size to MB for IOPS calculation
convert_block_size_to_mb() {
    local block_size=$1
    case "$block_size" in
        *k|*K) echo "scale=6; ${block_size%[kK]} / 1024" | bc -l ;;
        *m|*M) echo "${block_size%[mM]}" | bc -l ;;
        *g|*G) echo "scale=0; ${block_size%[gG]} * 1024" | bc -l ;;
        *) echo "scale=6; $block_size / 1024 / 1024" | bc -l ;;
    esac
}

# Function to run parallel tests on 2x2 arrays
run_parallel_test() {
    local array_config=$1
    local operation=$2
    local block_size=$3
    local queue_depth=$4
    
    local mount_points=(${ARRAY_CONFIGS[$array_config]//,/ })
    local mount1=${mount_points[0]}
    local mount2=${mount_points[1]}
    
    echo "Testing: $array_config, $operation, BS=$block_size, QD=$queue_depth (Parallel, Ext4)"
    
    # Check if both mount points exist
    if [ ! -d "$mount1" ] || [ ! -w "$mount1" ] || [ ! -d "$mount2" ] || [ ! -w "$mount2" ]; then
        echo "  ERROR: Mount points not accessible"
        echo "$array_config,$operation,$block_size,$queue_depth,$NUM_JOBS_2DRIVE,0,0,0,2x2_Parallel,Ext4" >> "$OUTPUT_FILE"
        return
    fi
    
    # Run parallel fio tests on both arrays simultaneously
    local temp_results="/tmp/parallel_results_ext4_$$"
    
    {
        sudo fio \
            --name=array1-test \
            --ioengine=libaio \
            --iodepth=$queue_depth \
            --rw=$operation \
            --bs=$block_size \
            --direct=1 \
            --size=$TEST_SIZE \
            --runtime=$RUNTIME \
            --time_based \
            --numjobs=$NUM_JOBS_2DRIVE \
            --group_reporting \
            --filename="$mount1/test_${operation}_${block_size}_qd${queue_depth}" \
            2>/dev/null | grep -E "^[[:space:]]*READ:|^[[:space:]]*WRITE:" | tail -1 > "${temp_results}_1" &
        
        sudo fio \
            --name=array2-test \
            --ioengine=libaio \
            --iodepth=$queue_depth \
            --rw=$operation \
            --bs=$block_size \
            --direct=1 \
            --size=$TEST_SIZE \
            --runtime=$RUNTIME \
            --time_based \
            --numjobs=$NUM_JOBS_2DRIVE \
            --group_reporting \
            --filename="$mount2/test_${operation}_${block_size}_qd${queue_depth}" \
            2>/dev/null | grep -E "^[[:space:]]*READ:|^[[:space:]]*WRITE:" | tail -1 > "${temp_results}_2" &
        
        wait
    }
    
    # Parse results from both arrays
    local total_gb_per_sec=0
    local total_mb_per_sec=0
    
    for i in 1 2; do
        if [ -f "${temp_results}_$i" ]; then
            local summary_line=$(cat "${temp_results}_$i")
            echo "  Array $i output: $summary_line"
            
            # Parse bandwidth
            local gb_per_sec=$(echo "$summary_line" | grep -oE '\([0-9.]+GB/s\)' | grep -oE '[0-9.]+')
            
            if [ -z "$gb_per_sec" ]; then
                local gib_per_sec=$(echo "$summary_line" | grep -oE 'bw=[0-9.]+\.?[0-9]*GiB/s' | grep -oE '[0-9.]+\.?[0-9]*')
                if [ -n "$gib_per_sec" ]; then
                    gb_per_sec=$(echo "scale=3; $gib_per_sec * 1.073741824" | bc -l)
                fi
            fi
            
            if [ -z "$gb_per_sec" ]; then
                local mib_per_sec=$(echo "$summary_line" | grep -oE 'bw=[0-9.]+\.?[0-9]*MiB/s' | grep -oE '[0-9.]+\.?[0-9]*')
                if [ -n "$mib_per_sec" ]; then
                    gb_per_sec=$(echo "scale=3; $mib_per_sec / 1024 * 1.073741824" | bc -l)
                fi
            fi
            
            if [ -n "$gb_per_sec" ] && [ "$gb_per_sec" != "0" ]; then
                total_gb_per_sec=$(echo "scale=3; $total_gb_per_sec + $gb_per_sec" | bc -l)
            fi
        fi
    done
    
    # Calculate aggregate results
    total_mb_per_sec=$(echo "scale=2; $total_gb_per_sec * 1000" | bc -l)
    
    # Calculate IOPS
    local block_size_mb=$(convert_block_size_to_mb "$block_size")
    local iops=0
    if [ "$total_mb_per_sec" != "0" ] && [ "$block_size_mb" != "0" ]; then
        iops=$(echo "scale=0; $total_mb_per_sec / $block_size_mb" | bc -l)
    fi
    
    echo "  Aggregate: ${total_gb_per_sec} GB/s, ${iops} IOPS"
    
    # Write to CSV
    echo "$array_config,$operation,$block_size,$queue_depth,$NUM_JOBS_2DRIVE,$total_mb_per_sec,$total_gb_per_sec,$iops,2x2_Parallel,Ext4" >> "$OUTPUT_FILE"
    
    # Clean up
    rm -f "${temp_results}_"* "$mount1/test_"* "$mount2/test_"*
    sleep 2
}

# Function to run single 4-drive array test
run_single_test() {
    local array_config=$1
    local operation=$2
    local block_size=$3
    local queue_depth=$4
    
    local mount_point=${ARRAY_CONFIGS[$array_config]}
    local test_file="${mount_point}/test_${operation}_${block_size}_qd${queue_depth}"
    
    echo "Testing: $array_config, $operation, BS=$block_size, QD=$queue_depth (Single, Ext4)"
    
    if [ ! -d "$mount_point" ] || [ ! -w "$mount_point" ]; then
        echo "  ERROR: $mount_point not accessible"
        echo "$array_config,$operation,$block_size,$queue_depth,$NUM_JOBS_4DRIVE,0,0,0,4Drive_Single,Ext4" >> "$OUTPUT_FILE"
        return
    fi
    
    # Run fio test
    local fio_output=$(sudo fio \
        --name=single-test \
        --ioengine=libaio \
        --iodepth=$queue_depth \
        --rw=$operation \
        --bs=$block_size \
        --direct=1 \
        --size=$TEST_SIZE \
        --runtime=$RUNTIME \
        --time_based \
        --numjobs=$NUM_JOBS_4DRIVE \
        --group_reporting \
        --filename="$test_file" \
        2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$fio_output" ]; then
        local summary_line=$(echo "$fio_output" | grep -E "^[[:space:]]*READ:|^[[:space:]]*WRITE:" | tail -1)
        
        if [ -n "$summary_line" ]; then
            echo "  Raw output: $summary_line"
            
            # Parse bandwidth
            local gb_per_sec=$(echo "$summary_line" | grep -oE '\([0-9.]+GB/s\)' | grep -oE '[0-9.]+')
            
            if [ -z "$gb_per_sec" ]; then
                local gib_per_sec=$(echo "$summary_line" | grep -oE 'bw=[0-9.]+\.?[0-9]*GiB/s' | grep -oE '[0-9.]+\.?[0-9]*')
                if [ -n "$gib_per_sec" ]; then
                    gb_per_sec=$(echo "scale=3; $gib_per_sec * 1.073741824" | bc -l)
                fi
            fi
            
            if [ -z "$gb_per_sec" ]; then
                local mib_per_sec=$(echo "$summary_line" | grep -oE 'bw=[0-9.]+\.?[0-9]*MiB/s' | grep -oE '[0-9.]+\.?[0-9]*')
                if [ -n "$mib_per_sec" ]; then
                    gb_per_sec=$(echo "scale=3; $mib_per_sec / 1024 * 1.073741824" | bc -l)
                fi
            fi
            
            local mb_per_sec=$(echo "scale=2; $gb_per_sec * 1000" | bc -l)
            
            # Calculate IOPS
            local block_size_mb=$(convert_block_size_to_mb "$block_size")
            local iops=0
            if [ "$mb_per_sec" != "0" ] && [ "$block_size_mb" != "0" ]; then
                iops=$(echo "scale=0; $mb_per_sec / $block_size_mb" | bc -l)
            fi
            
            echo "  Parsed: ${gb_per_sec} GB/s, ${iops} IOPS"
            echo "$array_config,$operation,$block_size,$queue_depth,$NUM_JOBS_4DRIVE,$mb_per_sec,$gb_per_sec,$iops,4Drive_Single,Ext4" >> "$OUTPUT_FILE"
        else
            echo "  Failed to parse results"
            echo "$array_config,$operation,$block_size,$queue_depth,$NUM_JOBS_4DRIVE,0,0,0,4Drive_Single,Ext4" >> "$OUTPUT_FILE"
        fi
    else
        echo "  Failed to run fio"
        echo "$array_config,$operation,$block_size,$queue_depth,$NUM_JOBS_4DRIVE,0,0,0,4Drive_Single,Ext4" >> "$OUTPUT_FILE"
    fi
    
    rm -f "$test_file"
    sleep 2
}

# Setup function to create Ext4 arrays
setup_ext4_arrays() {
    echo "Setting up Ext4 arrays..."
    
    # Stop any existing arrays
    sudo umount /mnt/*ext4* 2>/dev/null || true
    sudo mdadm --stop /dev/md4 /dev/md5 /dev/md6 /dev/md7 /dev/md8 /dev/md9 2>/dev/null || true
    
    # Create 2x2 arrays first
    echo "Creating 2x2 arrays..."
    sudo mdadm --create /dev/md4 --level=0 --raid-devices=2 --chunk=1024 \
        /dev/nvme15n1 /dev/nvme18n1  # WD Array 1
    
    sudo mdadm --create /dev/md5 --level=0 --raid-devices=2 --chunk=1024 \
        /dev/nvme19n1 /dev/nvme20n1  # WD Array 2
    
    sudo mdadm --create /dev/md6 --level=0 --raid-devices=2 --chunk=1024 \
        /dev/nvme0n1 /dev/nvme1n1    # SK Hynix Array 1
    
    sudo mdadm --create /dev/md7 --level=0 --raid-devices=2 --chunk=1024 \
        /dev/nvme2n1 /dev/nvme3n1    # SK Hynix Array 2
    
    # Create Ext4 filesystems with optimizations for large files
    echo "Creating optimized Ext4 filesystems..."
    sudo mkfs.ext4 -F -T largefile4 -E stride=256,stripe-width=512 /dev/md4
    sudo mkfs.ext4 -F -T largefile4 -E stride=256,stripe-width=512 /dev/md5
    sudo mkfs.ext4 -F -T largefile4 -E stride=256,stripe-width=512 /dev/md6
    sudo mkfs.ext4 -F -T largefile4 -E stride=256,stripe-width=512 /dev/md7
    
    # Create mount points and mount with performance optimizations
    sudo mkdir -p /mnt/{wd_array1_ext4,wd_array2_ext4,sk_array1_ext4,sk_array2_ext4}
    
    sudo mount -o noatime,data=writeback,barrier=0,nobh,errors=remount-ro /dev/md4 /mnt/wd_array1_ext4
    sudo mount -o noatime,data=writeback,barrier=0,nobh,errors=remount-ro /dev/md5 /mnt/wd_array2_ext4
    sudo mount -o noatime,data=writeback,barrier=0,nobh,errors=remount-ro /dev/md6 /mnt/sk_array1_ext4
    sudo mount -o noatime,data=writeback,barrier=0,nobh,errors=remount-ro /dev/md7 /mnt/sk_array2_ext4
    
    sudo chown -R $USER:$USER /mnt/{wd_array1_ext4,wd_array2_ext4,sk_array1_ext4,sk_array2_ext4}
    
    echo "2x2 Ext4 arrays ready"
}

# Setup function to create 4-drive Ext4 arrays
setup_4drive_ext4_arrays() {
    echo "Setting up 4-drive Ext4 arrays..."
    
    # Stop 2x2 arrays
    sudo umount /mnt/{wd_array1_ext4,wd_array2_ext4,sk_array1_ext4,sk_array2_ext4}
    sudo mdadm --stop /dev/md4 /dev/md5 /dev/md6 /dev/md7
    
    # Create 4-drive arrays
    sudo mdadm --create /dev/md8 --level=0 --raid-devices=4 --chunk=1024 \
        /dev/nvme15n1 /dev/nvme18n1 /dev/nvme19n1 /dev/nvme20n1  # WD 4-drive
    
    sudo mdadm --create /dev/md9 --level=0 --raid-devices=4 --chunk=1024 \
        /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1    # SK Hynix 4-drive
    
    # Create optimized Ext4 filesystems for 4-drive arrays
    sudo mkfs.ext4 -F -T largefile4 -E stride=256,stripe-width=1024 /dev/md8
    sudo mkfs.ext4 -F -T largefile4 -E stride=256,stripe-width=1024 /dev/md9
    
    # Mount with performance optimizations
    sudo mkdir -p /mnt/{wd_4drive_ext4,sk_4drive_ext4}
    sudo mount -o noatime,data=writeback,barrier=0,nobh,errors=remount-ro /dev/md8 /mnt/wd_4drive_ext4
    sudo mount -o noatime,data=writeback,barrier=0,nobh,errors=remount-ro /dev/md9 /mnt/sk_4drive_ext4
    
    sudo chown -R $USER:$USER /mnt/{wd_4drive_ext4,sk_4drive_ext4}
    
    echo "4-drive Ext4 arrays ready"
}

# Check if bc is available
if ! command -v bc &> /dev/null; then
    echo "Installing bc for calculations..."
    sudo apt update && sudo apt install -y bc
fi

echo "Step 1: Testing 2x2 Ext4 parallel arrays..."

# Setup 2x2 arrays
setup_ext4_arrays

# Test 2x2 arrays
parallel_configs=("WD_2x2_Parallel" "SKHynix_2x2_Parallel")
total_parallel_tests=$((${#parallel_configs[@]} * ${#OPERATIONS[@]} * ${#BLOCK_SIZES[@]} * ${#QUEUE_DEPTHS[@]}))

test_count=0
for array_config in "${parallel_configs[@]}"; do
    echo "=== Testing $array_config (Ext4) ==="
    for operation in "${OPERATIONS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            for queue_depth in "${QUEUE_DEPTHS[@]}"; do
                test_count=$((test_count + 1))
                echo "[$test_count/$total_parallel_tests]"
                run_parallel_test "$array_config" "$operation" "$block_size" "$queue_depth"
                echo
            done
        done
    done
done

echo "Step 2: Setting up and testing 4-drive Ext4 single arrays..."

# Setup and test 4-drive arrays
setup_4drive_ext4_arrays

single_configs=("WD_4Drive_Single" "SKHynix_4Drive_Single")
total_single_tests=$((${#single_configs[@]} * ${#OPERATIONS[@]} * ${#BLOCK_SIZES[@]} * ${#QUEUE_DEPTHS[@]}))

test_count=0
for array_config in "${single_configs[@]}"; do
    echo "=== Testing $array_config (Ext4) ==="
    for operation in "${OPERATIONS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            for queue_depth in "${QUEUE_DEPTHS[@]}"; do
                test_count=$((test_count + 1))
                echo "[$test_count/$total_single_tests]"
                run_single_test "$array_config" "$operation" "$block_size" "$queue_depth"
                echo
            done
        done
    done
done

echo "Ext4 array scaling benchmark complete!"
echo "Results saved to: $OUTPUT_FILE"
echo
echo "Quick analysis:"
echo "# Ext4 Performance Results:"
echo "echo 'WD 2x2 Parallel Ext4 best read:'"
echo "grep '^WD_2x2_Parallel,read,' $OUTPUT_FILE | sort -t',' -k7 -nr | head -3"
echo "echo 'WD 4-Drive Single Ext4 best read:'"  
echo "grep '^WD_4Drive_Single,read,' $OUTPUT_FILE | sort -t',' -k7 -nr | head -3"