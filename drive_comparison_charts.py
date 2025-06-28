#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

def convert_block_size_to_bytes(block_size_str):
    """Convert block size string to bytes for proper sorting"""
    size_str = block_size_str.lower()
    
    if size_str.endswith('k'):
        return int(size_str[:-1]) * 1024
    elif size_str.endswith('m'):
        return int(size_str[:-1]) * 1024 * 1024
    elif size_str.endswith('g'):
        return int(size_str[:-1]) * 1024 * 1024 * 1024
    else:
        return int(size_str)

def sort_block_sizes(block_sizes):
    """Sort block sizes by actual size, not alphabetically"""
    return sorted(block_sizes, key=convert_block_size_to_bytes)

def create_drive_comparison_charts(df, output_dir="."):
    """Create drive technology comparison charts"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Custom color palette for drive types
    drive_colors = {
        'WD_4TB': '#2E86AB',           # Blue
        'SKHynix_2TB': '#A23B72',     # Purple  
        'Intel_Optane_32GB': '#F18F01' # Orange
    }
    
    # Get properly sorted block sizes
    all_block_sizes = sort_block_sizes(df['BlockSize'].unique())
    drive_types = df['DriveType'].unique()
    
    # 1. Peak Performance Comparison by Drive Type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Read performance comparison
    read_data = df[df['Operation'] == 'read']
    peak_read_by_drive = read_data.groupby('DriveType')['Bandwidth_GBps'].max()
    
    bars1 = ax1.bar(range(len(peak_read_by_drive)), peak_read_by_drive.values, 
                    color=[drive_colors[dt] for dt in peak_read_by_drive.index])
    ax1.set_xlabel('Drive Type')
    ax1.set_ylabel('Peak Read Bandwidth (GB/s)')
    ax1.set_title('Peak Read Performance Comparison\n4-Drive RAID 0 Arrays')
    ax1.set_xticks(range(len(peak_read_by_drive)))
    ax1.set_xticklabels([dt.replace('_', ' ') for dt in peak_read_by_drive.index])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f} GB/s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    # Write performance comparison
    write_data = df[df['Operation'] == 'write']
    peak_write_by_drive = write_data.groupby('DriveType')['Bandwidth_GBps'].max()
    
    bars2 = ax2.bar(range(len(peak_write_by_drive)), peak_write_by_drive.values,
                    color=[drive_colors[dt] for dt in peak_write_by_drive.index])
    ax2.set_xlabel('Drive Type')
    ax2.set_ylabel('Peak Write Bandwidth (GB/s)')
    ax2.set_title('Peak Write Performance Comparison\n4-Drive RAID 0 Arrays')
    ax2.set_xticks(range(len(peak_write_by_drive)))
    ax2.set_xticklabels([dt.replace('_', ' ') for dt in peak_write_by_drive.index])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f} GB/s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/peak_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance vs Block Size Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Read performance vs block size
    for drive_type in drive_types:
        drive_read_data = read_data[read_data['DriveType'] == drive_type]
        # Get peak performance for each block size
        peak_by_bs = drive_read_data.groupby('BlockSize')['Bandwidth_GBps'].max()
        # Sort by block size
        peak_by_bs_sorted = peak_by_bs.reindex(all_block_sizes).dropna()
        
        ax1.plot(peak_by_bs_sorted.index, peak_by_bs_sorted.values,
                marker='o', linewidth=3, markersize=8, 
                color=drive_colors[drive_type], 
                label=drive_type.replace('_', ' '))
    
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Peak Read Bandwidth (GB/s)')
    ax1.set_title('Read Performance vs Block Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Write performance vs block size
    for drive_type in drive_types:
        drive_write_data = write_data[write_data['DriveType'] == drive_type]
        # Get peak performance for each block size
        peak_by_bs = drive_write_data.groupby('BlockSize')['Bandwidth_GBps'].max()
        # Sort by block size
        peak_by_bs_sorted = peak_by_bs.reindex(all_block_sizes).dropna()
        
        ax2.plot(peak_by_bs_sorted.index, peak_by_bs_sorted.values,
                marker='s', linewidth=3, markersize=8,
                color=drive_colors[drive_type],
                label=drive_type.replace('_', ' '))
    
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Peak Write Bandwidth (GB/s)')
    ax2.set_title('Write Performance vs Block Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_vs_blocksize.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance vs Queue Depth Comparison  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Read performance vs queue depth (for 1M block size)
    for drive_type in drive_types:
        drive_read_data = read_data[(read_data['DriveType'] == drive_type) & 
                                   (read_data['BlockSize'] == '1m')]
        if not drive_read_data.empty:
            ax1.plot(drive_read_data['QueueDepth'], drive_read_data['Bandwidth_GBps'],
                    marker='o', linewidth=3, markersize=8,
                    color=drive_colors[drive_type],
                    label=drive_type.replace('_', ' '))
    
    ax1.set_xlabel('Queue Depth')
    ax1.set_ylabel('Read Bandwidth (GB/s)')
    ax1.set_title('Read Performance vs Queue Depth (1M Block Size)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Write performance vs queue depth (for 1M block size)
    for drive_type in drive_types:
        drive_write_data = write_data[(write_data['DriveType'] == drive_type) & 
                                     (write_data['BlockSize'] == '1m')]
        if not drive_write_data.empty:
            ax2.plot(drive_write_data['QueueDepth'], drive_write_data['Bandwidth_GBps'],
                    marker='s', linewidth=3, markersize=8,
                    color=drive_colors[drive_type],
                    label=drive_type.replace('_', ' '))
    
    ax2.set_xlabel('Queue Depth')
    ax2.set_ylabel('Write Bandwidth (GB/s)')
    ax2.set_title('Write Performance vs Queue Depth (1M Block Size)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_vs_queuedepth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. IOPS Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Peak IOPS by drive type for small block sizes
    small_block_data = df[df['BlockSize'].isin(['4k', '64k'])]
    peak_iops_by_drive = small_block_data.groupby('DriveType')['IOPS'].max()
    
    bars = ax.bar(range(len(peak_iops_by_drive)), peak_iops_by_drive.values,
                  color=[drive_colors[dt] for dt in peak_iops_by_drive.index])
    ax.set_xlabel('Drive Type')
    ax.set_ylabel('Peak IOPS')
    ax.set_title('Peak IOPS Comparison (4K-64K Block Sizes)\n4-Drive RAID 0 Arrays')
    ax.set_xticks(range(len(peak_iops_by_drive)))
    ax.set_xticklabels([dt.replace('_', ' ') for dt in peak_iops_by_drive.index])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/iops_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Comprehensive Performance Summary Table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table data
    summary_data = []
    for drive_type in drive_types:
        drive_data = df[df['DriveType'] == drive_type]
        read_data_dt = drive_data[drive_data['Operation'] == 'read']
        write_data_dt = drive_data[drive_data['Operation'] == 'write']
        
        peak_read = read_data_dt['Bandwidth_GBps'].max()
        peak_write = write_data_dt['Bandwidth_GBps'].max()
        peak_iops = drive_data['IOPS'].max()
        
        # Find best block sizes
        best_read_bs = read_data_dt.loc[read_data_dt['Bandwidth_GBps'].idxmax(), 'BlockSize']
        best_write_bs = write_data_dt.loc[write_data_dt['Bandwidth_GBps'].idxmax(), 'BlockSize']
        
        summary_data.append([
            drive_type.replace('_', ' '),
            f'{peak_read:.1f}',
            best_read_bs,
            f'{peak_write:.1f}',
            best_write_bs,
            f'{peak_iops:,.0f}'
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Drive Type', 'Peak Read\n(GB/s)', 'Best Read\nBlock Size', 
                              'Peak Write\n(GB/s)', 'Best Write\nBlock Size', 'Peak IOPS'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color the rows by drive type
    for i, drive_type in enumerate(drive_types):
        for j in range(len(summary_data[0])):
            table[(i+1, j)].set_facecolor([c/255 for c in [int(drive_colors[drive_type][1:3], 16), 
                                                           int(drive_colors[drive_type][3:5], 16), 
                                                           int(drive_colors[drive_type][5:7], 16), 0.3]])
    
    ax.set_title('Drive Technology Performance Summary\n4-Drive RAID 0 Arrays', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Drive comparison charts saved to {output_dir}/")
    print("Generated charts:")
    print("  - peak_performance_comparison.png")
    print("  - performance_vs_blocksize.png")
    print("  - performance_vs_queuedepth.png")
    print("  - iops_comparison.png")
    print("  - performance_summary_table.png")

def print_performance_summary(df):
    """Print a summary of drive performance comparison"""
    print("\n=== DRIVE TECHNOLOGY COMPARISON SUMMARY ===")
    
    drive_types = df['DriveType'].unique()
    
    print(f"\nTesting 4-drive RAID 0 arrays:")
    for drive_type in drive_types:
        drive_data = df[df['DriveType'] == drive_type]
        read_data = drive_data[drive_data['Operation'] == 'read']
        write_data = drive_data[drive_data['Operation'] == 'write']
        
        peak_read = read_data['Bandwidth_GBps'].max()
        peak_write = write_data['Bandwidth_GBps'].max()
        peak_iops = drive_data['IOPS'].max()
        
        print(f"\n{drive_type.replace('_', ' ')}:")
        print(f"  Peak Read:  {peak_read:.1f} GB/s")
        print(f"  Peak Write: {peak_write:.1f} GB/s") 
        print(f"  Peak IOPS:  {peak_iops:,.0f}")
    
    # Performance ratios
    print(f"\n=== Performance Ratios (vs Intel Optane) ===")
    optane_data = df[df['DriveType'] == 'Intel_Optane_32GB']
    if not optane_data.empty:
        optane_read = optane_data[optane_data['Operation'] == 'read']['Bandwidth_GBps'].max()
        optane_write = optane_data[optane_data['Operation'] == 'write']['Bandwidth_GBps'].max()
        
        for drive_type in drive_types:
            if drive_type != 'Intel_Optane_32GB':
                drive_data = df[df['DriveType'] == drive_type]
                drive_read = drive_data[drive_data['Operation'] == 'read']['Bandwidth_GBps'].max()
                drive_write = drive_data[drive_data['Operation'] == 'write']['Bandwidth_GBps'].max()
                
                read_ratio = drive_read / optane_read if optane_read > 0 else 0
                write_ratio = drive_write / optane_write if optane_write > 0 else 0
                
                print(f"\n{drive_type.replace('_', ' ')}:")
                print(f"  {read_ratio:.1f}x faster read")
                print(f"  {write_ratio:.1f}x faster write")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 drive_comparison_charts.py <comparison_csv_file>")
        print("Example: python3 drive_comparison_charts.py /mnt/drive_comparison_results.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please run the drive comparison benchmark first.")
        sys.exit(1)
    
    print(f"Loading drive comparison data from {csv_file}...")
    print(f"Loaded {len(df)} test results")
    print(f"Drive types: {df['DriveType'].unique()}")
    print(f"Operations: {df['Operation'].unique()}")
    print(f"Block sizes: {sorted(df['BlockSize'].unique(), key=lambda x: int(x[:-1]) * (1024 if x[-1]=='k' else 1024*1024))}")
    print(f"Queue depths: {sorted(df['QueueDepth'].unique())}")
    
    # Create charts
    print("\nGenerating drive comparison charts...")
    create_drive_comparison_charts(df)
    
    # Print summary
    print_performance_summary(df)

if __name__ == "__main__":
    main()