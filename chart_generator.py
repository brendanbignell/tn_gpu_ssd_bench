#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

def load_benchmark_data(csv_file):
    """Load benchmark data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please run the benchmark script first.")
        sys.exit(1)

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
        # Assume bytes if no unit
        return int(size_str)

def sort_block_sizes(block_sizes):
    """Sort block sizes by actual size, not alphabetically"""
    return sorted(block_sizes, key=convert_block_size_to_bytes)

def create_performance_charts(df, output_dir="/mnt/kvcache/charts"):
    """Create various performance charts"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Get properly sorted block sizes
    all_block_sizes = sort_block_sizes(df['BlockSize'].unique())
    
    # 1. Bandwidth vs Block Size (for different queue depths)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Read performance
    read_data = df[df['Operation'] == 'read']
    for qd in sorted(read_data['QueueDepth'].unique()):
        qd_data = read_data[read_data['QueueDepth'] == qd]
        # Sort by block size
        qd_data_sorted = qd_data.set_index('BlockSize').reindex(all_block_sizes).reset_index()
        ax1.plot(qd_data_sorted['BlockSize'], qd_data_sorted['Bandwidth_GBps'], 
                marker='o', label=f'QD={qd}', linewidth=2)
    
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Read Performance vs Block Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('linear')
    
    # Write performance
    write_data = df[df['Operation'] == 'write']
    for qd in sorted(write_data['QueueDepth'].unique()):
        qd_data = write_data[write_data['QueueDepth'] == qd]
        # Sort by block size
        qd_data_sorted = qd_data.set_index('BlockSize').reindex(all_block_sizes).reset_index()
        ax2.plot(qd_data_sorted['BlockSize'], qd_data_sorted['Bandwidth_GBps'], 
                marker='s', label=f'QD={qd}', linewidth=2)
    
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title('Write Performance vs Block Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bandwidth_vs_blocksize.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bandwidth vs Queue Depth (for different block sizes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Read performance - use properly sorted block sizes for legend
    block_sizes_to_plot = ['4k', '64k', '1m', '16m', '64m']  # Representative sizes in correct order
    for bs in block_sizes_to_plot:
        if bs in read_data['BlockSize'].values:
            bs_data = read_data[read_data['BlockSize'] == bs]
            ax1.plot(bs_data['QueueDepth'], bs_data['Bandwidth_GBps'], 
                    marker='o', label=f'{bs}', linewidth=2)
    
    ax1.set_xlabel('Queue Depth')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Read Performance vs Queue Depth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Write performance
    for bs in block_sizes_to_plot:
        if bs in write_data['BlockSize'].values:
            bs_data = write_data[write_data['BlockSize'] == bs]
            ax2.plot(bs_data['QueueDepth'], bs_data['Bandwidth_GBps'], 
                    marker='s', label=f'{bs}', linewidth=2)
    
    ax2.set_xlabel('Queue Depth')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title('Write Performance vs Queue Depth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bandwidth_vs_queuedepth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of Read Performance (with proper block size ordering)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create pivot tables for heatmaps with proper ordering
    read_pivot = read_data.pivot(index='QueueDepth', columns='BlockSize', values='Bandwidth_GBps')
    write_pivot = write_data.pivot(index='QueueDepth', columns='BlockSize', values='Bandwidth_GBps')
    
    # Reorder columns by block size
    read_pivot = read_pivot.reindex(columns=all_block_sizes)
    write_pivot = write_pivot.reindex(columns=all_block_sizes)
    
    # Read heatmap
    sns.heatmap(read_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Read Performance Heatmap (GB/s)')
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Queue Depth')
    
    # Write heatmap
    sns.heatmap(write_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax2)
    ax2.set_title('Write Performance Heatmap (GB/s)')
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Queue Depth')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. IOPS Performance (with proper block size ordering)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Read IOPS vs Block Size
    for qd in [1, 8, 32, 128]:  # Selected queue depths
        if qd in read_data['QueueDepth'].values:
            qd_data = read_data[read_data['QueueDepth'] == qd]
            # Sort by block size
            qd_data_sorted = qd_data.set_index('BlockSize').reindex(all_block_sizes).reset_index()
            ax1.plot(qd_data_sorted['BlockSize'], qd_data_sorted['IOPS'], 
                    marker='o', label=f'QD={qd}', linewidth=2)
    
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('IOPS')
    ax1.set_title('Read IOPS vs Block Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Write IOPS vs Block Size
    for qd in [1, 8, 32, 128]:  # Selected queue depths
        if qd in write_data['QueueDepth'].values:
            qd_data = write_data[write_data['QueueDepth'] == qd]
            # Sort by block size
            qd_data_sorted = qd_data.set_index('BlockSize').reindex(all_block_sizes).reset_index()
            ax2.plot(qd_data_sorted['BlockSize'], qd_data_sorted['IOPS'], 
                    marker='s', label=f'QD={qd}', linewidth=2)
    
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('IOPS')
    ax2.set_title('Write IOPS vs Block Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/iops_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Peak Performance Summary Chart (with proper block size ordering)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find peak performance for each operation and block size
    peak_read = read_data.groupby('BlockSize')['Bandwidth_GBps'].max()
    peak_write = write_data.groupby('BlockSize')['Bandwidth_GBps'].max()
    
    # Reorder by block size
    peak_read = peak_read.reindex(all_block_sizes).dropna()
    peak_write = peak_write.reindex(all_block_sizes).dropna()
    
    x_pos = np.arange(len(peak_read.index))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, peak_read.values, width, label='Read Peak', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, peak_write.values, width, label='Write Peak', alpha=0.8)
    
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Peak Bandwidth (GB/s)')
    ax.set_title('Peak Performance by Block Size')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(peak_read.index)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/peak_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved to {output_dir}/")
    print("Generated charts:")
    print("  - bandwidth_vs_blocksize.png")
    print("  - bandwidth_vs_queuedepth.png") 
    print("  - performance_heatmaps.png")
    print("  - iops_performance.png")
    print("  - peak_performance_summary.png")

def print_performance_summary(df):
    """Print a summary of key performance metrics"""
    print("\n=== PERFORMANCE SUMMARY ===")
    
    # Overall peak performance
    peak_read = df[df['Operation'] == 'read']['Bandwidth_GBps'].max()
    peak_write = df[df['Operation'] == 'write']['Bandwidth_GBps'].max()
    
    # Find the conditions for peak performance
    peak_read_row = df[df['Bandwidth_GBps'] == peak_read].iloc[0]
    peak_write_row = df[df['Bandwidth_GBps'] == peak_write].iloc[0]
    
    print(f"\nPeak Read Performance: {peak_read:.1f} GB/s")
    print(f"  Block Size: {peak_read_row['BlockSize']}, Queue Depth: {peak_read_row['QueueDepth']}")
    
    print(f"\nPeak Write Performance: {peak_write:.1f} GB/s")
    print(f"  Block Size: {peak_write_row['BlockSize']}, Queue Depth: {peak_write_row['QueueDepth']}")
    
    # Performance by block size
    print(f"\n=== Performance by Block Size (Peak Values) ===")
    for op in ['read', 'write']:
        print(f"\n{op.upper()} Performance:")
        op_data = df[df['Operation'] == op]
        peak_by_bs = op_data.groupby('BlockSize')['Bandwidth_GBps'].max()
        for bs, bw in peak_by_bs.items():
            print(f"  {bs:>4}: {bw:>6.1f} GB/s")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 chart_generator.py <benchmark_csv_file>")
        print("Example: python3 chart_generator.py /mnt/kvcache/raid_performance_data.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Load data
    print(f"Loading benchmark data from {csv_file}...")
    df = load_benchmark_data(csv_file)
    
    print(f"Loaded {len(df)} test results")
    print(f"Operations: {df['Operation'].unique()}")
    print(f"Block sizes: {df['BlockSize'].unique()}")
    print(f"Queue depths: {sorted(df['QueueDepth'].unique())}")
    
    # Create charts
    print("\nGenerating performance charts...")
    create_performance_charts(df)
    
    # Print summary
    print_performance_summary(df)

if __name__ == "__main__":
    main()