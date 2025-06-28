#!/bin/bash

# RAID0 Setup Script for LLM KV Cache Storage
# Optimized for 1MB-1GB transfers on Ubuntu 24.02
# WARNING: This will DESTROY all data on the selected drives!

set -euo pipefail

# Configuration
MOUNT_POINT="/mnt/kvcache"
RAID_DEVICE="/dev/md0"
EXPECTED_DRIVE_COUNT=11

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root. Use: sudo $0"
    fi
}

# Install required packages
install_dependencies() {
    log "Installing required packages..."
    apt update
    apt install -y mdadm xfsprogs nvme-cli lshw
}

# Get detailed drive information
get_drive_info() {
    local nvme_dev="$1"
    local nvme_ctrl=$(echo "$nvme_dev" | sed 's/n1$//')
    
    # Get model from nvme-cli
    local model=$(nvme id-ctrl "$nvme_dev" 2>/dev/null | grep -i "mn " | cut -d: -f2 | xargs || echo "Unknown")
    
    # Get PCIe bus info
    local pci_addr=$(ls -l /sys/block/$(basename "$nvme_dev") | grep -o '[0-9a-f]\{4\}:[0-9a-f]\{2\}:[0-9a-f]\{2\}\.[0-9]' | head -1)
    
    if [[ -n "$pci_addr" ]]; then
        # Get current link speed and width
        local lnk_sta=$(lspci -s "$pci_addr" -vvv 2>/dev/null | grep "LnkSta:" | head -1)
        local lnk_cap=$(lspci -s "$pci_addr" -vvv 2>/dev/null | grep "LnkCap:" | head -1)
        
        # Extract current speed
        local current_speed=$(echo "$lnk_sta" | grep -o 'Speed [0-9.]*GT/s' | grep -o '[0-9.]*')
        local max_speed=$(echo "$lnk_cap" | grep -o 'Speed [0-9.]*GT/s' | grep -o '[0-9.]*')
        
        # Extract current width
        local current_width=$(echo "$lnk_sta" | grep -o 'Width x[0-9]*' | grep -o '[0-9]*')
        local max_width=$(echo "$lnk_cap" | grep -o 'Width x[0-9]*' | grep -o '[0-9]*')
        
        echo "$nvme_dev|$model|$current_speed|$max_speed|$current_width|$max_width|$pci_addr"
    else
        echo "$nvme_dev|$model|Unknown|Unknown|Unknown|Unknown|Unknown"
    fi
}

# Identify PCIe4 NVMe drives (WD and SK Hynix)
identify_drives() {
    log "Scanning all NVMe drives..."
    
    # Array to store all drive info
    declare -a all_drives_info
    declare -a candidate_drives
    
    # Get info for all NVMe drives
    for nvme_dev in /dev/nvme*n1; do
        if [[ -b "$nvme_dev" ]]; then
            drive_info=$(get_drive_info "$nvme_dev")
            all_drives_info+=("$drive_info")
        fi
    done
    
    echo
    echo "=== All NVMe Drives Found ==="
    printf "%-15s %-40s %-8s %-8s %-6s %-6s %s\n" "Device" "Model" "CurSpd" "MaxSpd" "CurW" "MaxW" "PCI Address"
    printf "%-15s %-40s %-8s %-8s %-6s %-6s %s\n" "-------" "-----" "------" "------" "----" "----" "-----------"
    
    for drive_info in "${all_drives_info[@]}"; do
        IFS='|' read -r device model cur_speed max_speed cur_width max_width pci_addr <<< "$drive_info"
        printf "%-15s %-40s %-8s %-8s %-6s %-6s %s\n" "$device" "$model" "${cur_speed}GT/s" "${max_speed}GT/s" "x$cur_width" "x$max_width" "$pci_addr"
        
        # Check if it's a WD or SK Hynix drive
        if [[ "$model" =~ (WD|Western|SK.*[Hh]ynix|WDS|WD_BLACK|PC SN|SN[0-9]) ]]; then
            # Check if it's PCIe4 (16GT/s) or high-speed PCIe3 (8GT/s)
            if [[ "$cur_speed" == "16" ]] || [[ "$max_speed" == "16" ]]; then
                candidate_drives+=("$device")
            elif [[ "$cur_speed" == "8" ]] || [[ "$max_speed" == "8" ]]; then
                # Could be PCIe3 x4 or PCIe4 running at reduced speed
                warn "Drive $device may be PCIe3 (8GT/s): $model"
            fi
        fi
    done
    
    echo
    log "Candidate PCIe4 drives (WD/SK Hynix at 16GT/s):"
    if [[ ${#candidate_drives[@]} -eq 0 ]]; then
        warn "No PCIe4 WD/SK Hynix drives found automatically!"
        echo
        echo "Manual selection required. Please review the drive list above."
        echo "Enter the device names you want to include (space-separated):"
        echo "Example: /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1"
        read -r manual_selection
        if [[ -n "$manual_selection" ]]; then
            read -ra candidate_drives <<< "$manual_selection"
        else
            error "No drives selected"
        fi
    else
        for drive in "${candidate_drives[@]}"; do
            echo "  $drive"
        done
    fi
    
    echo
    echo "=== FINAL DRIVE SELECTION ==="
    drives=()
    for drive in "${candidate_drives[@]}"; do
        if [[ -b "$drive" ]]; then
            drives+=("$drive")
            # Get drive info for final confirmation
            for drive_info in "${all_drives_info[@]}"; do
                if [[ "$drive_info" =~ ^$drive\| ]]; then
                    IFS='|' read -r device model cur_speed max_speed cur_width max_width pci_addr <<< "$drive_info"
                    log "Selected: $device - $model (${cur_speed}GT/s, x$cur_width)"
                    break
                fi
            done
        else
            error "Drive $drive does not exist or is not a block device"
        fi
    done
    
    if [[ ${#drives[@]} -ne $EXPECTED_DRIVE_COUNT ]]; then
        warn "Expected $EXPECTED_DRIVE_COUNT drives, selected ${#drives[@]} drives"
        echo
        read -p "Continue with ${#drives[@]} drives? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Aborting due to drive count mismatch"
        fi
    fi
    
    if [[ ${#drives[@]} -eq 0 ]]; then
        error "No drives selected!"
    fi
}

# Confirm destructive operation
confirm_destruction() {
    echo
    warn "WARNING: This will PERMANENTLY DESTROY all data on the following drives:"
    printf '%s\n' "${drives[@]}"
    echo
    echo "Type 'DESTROY' to continue:"
    read -r confirmation
    if [[ "$confirmation" != "DESTROY" ]]; then
        error "Operation cancelled"
    fi
}

# Stop any existing RAID arrays using these drives
cleanup_existing() {
    log "Cleaning up any existing RAID arrays..."
    
    # Stop existing md0 if it exists
    if [[ -b "$RAID_DEVICE" ]]; then
        umount "$RAID_DEVICE" 2>/dev/null || true
        mdadm --stop "$RAID_DEVICE" || true
    fi
    
    # Remove any existing RAID signatures from drives
    for drive in "${drives[@]}"; do
        mdadm --zero-superblock "$drive" 2>/dev/null || true
        wipefs -a "$drive" 2>/dev/null || true
    done
}

# Create RAID0 array
create_raid() {
    log "Creating RAID0 array with ${#drives[@]} drives..."
    
    mdadm --create "$RAID_DEVICE" \
        --level=0 \
        --raid-devices=${#drives[@]} \
        --chunk=1024 \
        "${drives[@]}"
    
    # Wait for array to be ready
    log "Waiting for RAID array to initialize..."
    while ! mdadm --detail "$RAID_DEVICE" | grep -q "State : clean"; do
        sleep 2
    done
    
    log "RAID0 array created successfully"
    mdadm --detail "$RAID_DEVICE"
}

# Format with XFS optimized for KV cache workload
format_xfs() {
    log "Formatting with XFS filesystem optimized for KV cache..."
    
    # Calculate optimal stripe unit and stripe width
    # stripe_unit = chunk_size (1024K)
    # stripe_width = stripe_unit * number_of_drives
    stripe_unit=1048576  # 1024K in bytes
    stripe_width=$((stripe_unit * ${#drives[@]}))
    
    mkfs.xfs -f \
        -d su=${stripe_unit},sw=${#drives[@]} \
        -l size=128m \
        -i size=512 \
        -b size=4096 \
        "$RAID_DEVICE"
    
    log "XFS filesystem created with stripe optimization"
}

# Create mount point and mount
mount_filesystem() {
    log "Creating mount point and mounting filesystem..."
    
    mkdir -p "$MOUNT_POINT"
    
    # Mount with optimizations for KV cache workload
    mount -t xfs -o noatime,nodiratime,largeio,inode64,swalloc "$RAID_DEVICE" "$MOUNT_POINT"
    
    log "Filesystem mounted at $MOUNT_POINT"
}

# Update fstab for persistent mounting
update_fstab() {
    log "Updating /etc/fstab for persistent mounting..."
    
    # Get UUID of the filesystem
    uuid=$(blkid -s UUID -o value "$RAID_DEVICE")
    
    # Remove any existing entry for this mount point
    sed -i "\|$MOUNT_POINT|d" /etc/fstab
    
    # Add new entry
    echo "UUID=$uuid $MOUNT_POINT xfs noatime,nodiratime,largeio,inode64,swalloc 0 2" >> /etc/fstab
    
    log "fstab updated with UUID: $uuid"
}

# Save RAID configuration
save_raid_config() {
    log "Saving RAID configuration..."
    
    mdadm --detail --scan >> /etc/mdadm/mdadm.conf
    update-initramfs -u
    
    log "RAID configuration saved"
}

# Set optimal system parameters for KV cache workload
optimize_system() {
    log "Setting system optimizations for KV cache workload..."
    
    # Create optimization script
    cat > /etc/sysctl.d/99-kvcache-optimization.conf << EOF
# KV Cache Storage Optimizations
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.dirty_expire_centisecs = 1500
vm.dirty_writeback_centisecs = 500
vm.vfs_cache_pressure = 50

# NVMe queue depth optimization
# These will be applied per-device by udev rules
EOF

    # Create udev rules for NVMe optimization
    cat > /etc/udev/rules.d/99-nvme-kvcache.rules << EOF
# NVMe optimizations for KV cache workload
ACTION=="add|change", KERNEL=="nvme*n1", ATTR{queue/scheduler}="none"
ACTION=="add|change", KERNEL=="nvme*n1", ATTR{queue/nr_requests}="1024"
ACTION=="add|change", KERNEL=="nvme*n1", ATTR{queue/read_ahead_kb}="512"
EOF

    # Apply sysctl settings
    sysctl -p /etc/sysctl.d/99-kvcache-optimization.conf
    
    # Reload udev rules
    udevadm control --reload-rules
    udevadm trigger
    
    log "System optimizations applied"
}

# Display final information
show_results() {
    echo
    log "RAID0 KV Cache setup completed successfully!"
    echo
    echo "Configuration Summary:"
    echo "- RAID Device: $RAID_DEVICE"
    echo "- Mount Point: $MOUNT_POINT"
    echo "- Filesystem: XFS"
    echo "- Drives: ${#drives[@]}"
    echo "- Total Capacity: $(df -h "$MOUNT_POINT" | tail -n1 | awk '{print $2}')"
    echo "- Available Space: $(df -h "$MOUNT_POINT" | tail -n1 | awk '{print $4}')"
    echo
    echo "Performance optimizations:"
    echo "- RAID0 chunk size: 1024K"
    echo "- XFS stripe alignment: optimized"
    echo "- Mount options: noatime,nodiratime,largeio,inode64,swalloc"
    echo "- System parameters: tuned for KV cache workload"
    echo
    log "Your KV cache storage is ready for use!"
}

# Main execution
main() {
    log "Starting RAID0 KV Cache setup..."
    
    check_root
    install_dependencies
    identify_drives
    confirm_destruction
    cleanup_existing
    create_raid
    format_xfs
    mount_filesystem
    update_fstab
    save_raid_config
    optimize_system
    show_results
}

# Run main function
main "$@"