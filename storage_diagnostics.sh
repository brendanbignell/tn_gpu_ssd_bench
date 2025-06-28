#!/bin/bash

echo "=== STORAGE DIAGNOSTIC REPORT ==="
echo "=================================="

echo -e "\n1. RAID Configuration:"
echo "----------------------"
cat /proc/mdstat
echo ""
lsblk -f | grep -E "(md|nvme)"

echo -e "\n2. RAID Details:"
echo "----------------"
sudo mdadm --detail /dev/md0  # Adjust if your RAID device is different

echo -e "\n3. NVMe Drive Info:"
echo "-------------------"
for drive in /dev/nvme*n1; do
    echo "Drive: $drive"
    sudo nvme id-ctrl "$drive" | grep -E "(mn|sn|fr)" || echo "Could not read $drive info"
done

echo -e "\n4. Queue Depths:"
echo "----------------"
for device in /sys/block/nvme*; do
    if [ -d "$device" ]; then
        echo "$(basename $device): $(cat $device/queue/nr_requests)"
    fi
done

echo -e "\n5. CPU Governor:"
echo "----------------"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

echo -e "\n6. Mount Options:"
echo "-----------------"
mount | grep /mnt/kvcache

echo -e "\n7. Filesystem Info:"
echo "-------------------"
df -hT /mnt/kvcache

echo -e "\n8. I/O Scheduler:"
echo "-----------------"
for device in /sys/block/nvme*; do
    if [ -d "$device" ]; then
        echo "$(basename $device): $(cat $device/queue/scheduler)"
    fi
done

echo -e "\n9. Drive Performance (basic test):"
echo "-----------------------------------"
echo "Testing individual drive performance..."
sudo hdparm -tT /dev/nvme0n1 2>/dev/null || echo "hdparm not available"

echo -e "\n10. Memory Info:"
echo "----------------"
free -h
echo "Available: $(cat /proc/meminfo | grep MemAvailable)"
