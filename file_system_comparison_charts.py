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

def load_and_combine_data(xfs_file, ext4_file):
    """Load and combine XFS and Ext4 data"""
    
    # Load XFS data
    try:
        xfs_df = pd.read_csv(xfs_file)
        # Add filesystem column if not present
        if 'Filesystem' not in xfs_df.columns:
            xfs_df['Filesystem'] = 'XFS'
        print(f"Loaded {len(xfs_df)} XFS results")
    except FileNotFoundError:
        print(f"Error: Could not find XFS file {xfs_file}")
        sys.exit(1)
    
    # Load Ext4 data
    try:
        ext4_df = pd.read_csv(ext4_file)
        # Add filesystem column if not present
        if 'Filesystem' not in ext4_df.columns:
            ext4_df['Filesystem'] = 'Ext4'
        print(f"Loaded {len(ext4_df)} Ext4 results")
    except FileNotFoundError:
        print(f"Error: Could not find Ext4 file {ext4_file}")
        sys.exit(1)
    
    # Combine dataframes
    combined_df = pd.concat([xfs_df, ext4_df], ignore_index=True)
    
    # Extract drive types
    combined_df['DriveType'] = combined_df['ArrayConfig'].str.extract(r'(WD|SKHynix)')
    
    print(f"Combined dataset: {len(combined_df)} total results")
    print(f"Filesystems: {combined_df['Filesystem'].unique()}")
    print(f"Drive types: {combined_df['DriveType'].unique()}")
    print(f"Array types: {combined_df['ArrayType'].unique()}")
    
    return combined_df

def create_filesystem_comparison_charts(df, output_dir="."):
    """Create XFS vs Ext4 filesystem comparison charts"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Color scheme
    fs_colors = {
        'XFS': '#2E86AB',      # Blue
        'Ext4': '#F18F01'      # Orange
    }
    
    array_colors = {
        '2x2_Parallel': '#E74C3C',    # Red
        '4Drive_Single': '#3498DB'    # Blue
    }
    
    # Get properly sorted block sizes
    all_block_sizes = sort_block_sizes(df['BlockSize'].unique())
    
    # 1. Peak Performance Comparison by Filesystem
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # WD Read Performance by Filesystem
    wd_data = df[df['DriveType'] == 'WD']
    wd_read = wd_data[wd_data['Operation'] == 'read']
    
    filesystem_performance = []
    for fs in ['XFS', 'Ext4']:
        for array_type in ['2x2_Parallel', '4Drive_Single']:
            fs_array_data = wd_read[(wd_read['Filesystem'] == fs) & (wd_read['ArrayType'] == array_type)]
            if not fs_array_data.empty:
                peak_perf = fs_array_data['Bandwidth_GBps'].max()
                filesystem_performance.append({
                    'Config': f'{array_type.replace("_", " ")}',
                    'Filesystem': fs,
                    'Performance': peak_perf
                })
    
    if filesystem_performance:
        fs_df = pd.DataFrame(filesystem_performance)
        configs = fs_df['Config'].unique()
        x_pos = np.arange(len(configs))
        width = 0.35
        
        xfs_vals = [fs_df[(fs_df['Config'] == config) & (fs_df['Filesystem'] == 'XFS')]['Performance'].iloc[0] 
                   if not fs_df[(fs_df['Config'] == config) & (fs_df['Filesystem'] == 'XFS')].empty else 0 
                   for config in configs]
        ext4_vals = [fs_df[(fs_df['Config'] == config) & (fs_df['Filesystem'] == 'Ext4')]['Performance'].iloc[0] 
                    if not fs_df[(fs_df['Config'] == config) & (fs_df['Filesystem'] == 'Ext4')].empty else 0 
                    for config in configs]
        
        bars1 = ax1.bar(x_pos - width/2, xfs_vals, width, label='XFS', color=fs_colors['XFS'])
        bars2 = ax1.bar(x_pos + width/2, ext4_vals, width, label='Ext4', color=fs_colors['Ext4'])
        
        ax1.set_xlabel('Array Configuration')
        ax1.set_ylabel('Peak Read Bandwidth (GB/s)')
        ax1.set_title('WD Drives: Filesystem Comparison (Read)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(configs)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
    
    # WD Write Performance by Filesystem
    wd_write = wd_data[wd_data['Operation'] == 'write']
    
    filesystem_performance_write = []
    for fs in ['XFS', 'Ext4']:
        for array_type in ['2x2_Parallel', '4Drive_Single']:
            fs_array_data = wd_write[(wd_write['Filesystem'] == fs) & (wd_write['ArrayType'] == array_type)]
            if not fs_array_data.empty:
                peak_perf = fs_array_data['Bandwidth_GBps'].max()
                filesystem_performance_write.append({
                    'Config': f'{array_type.replace("_", " ")}',
                    'Filesystem': fs,
                    'Performance': peak_perf
                })
    
    if filesystem_performance_write:
        fs_write_df = pd.DataFrame(filesystem_performance_write)
        
        xfs_write_vals = [fs_write_df[(fs_write_df['Config'] == config) & (fs_write_df['Filesystem'] == 'XFS')]['Performance'].iloc[0] 
                         if not fs_write_df[(fs_write_df['Config'] == config) & (fs_write_df['Filesystem'] == 'XFS')].empty else 0 
                         for config in configs]
        ext4_write_vals = [fs_write_df[(fs_write_df['Config'] == config) & (fs_write_df['Filesystem'] == 'Ext4')]['Performance'].iloc[0] 
                          if not fs_write_df[(fs_write_df['Config'] == config) & (fs_write_df['Filesystem'] == 'Ext4')].empty else 0 
                          for config in configs]
        
        bars3 = ax2.bar(x_pos - width/2, xfs_write_vals, width, label='XFS', color=fs_colors['XFS'])
        bars4 = ax2.bar(x_pos + width/2, ext4_write_vals, width, label='Ext4', color=fs_colors['Ext4'])
        
        ax2.set_xlabel('Array Configuration')
        ax2.set_ylabel('Peak Write Bandwidth (GB/s)')
        ax2.set_title('WD Drives: Filesystem Comparison (Write)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(configs)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
    
    # SK Hynix Read Performance by Filesystem
    sk_data = df[df['DriveType'] == 'SKHynix']
    sk_read = sk_data[sk_data['Operation'] == 'read']
    
    sk_filesystem_performance = []
    for fs in ['XFS', 'Ext4']:
        for array_type in ['2x2_Parallel', '4Drive_Single']:
            fs_array_data = sk_read[(sk_read['Filesystem'] == fs) & (sk_read['ArrayType'] == array_type)]
            if not fs_array_data.empty:
                peak_perf = fs_array_data['Bandwidth_GBps'].max()
                sk_filesystem_performance.append({
                    'Config': f'{array_type.replace("_", " ")}',
                    'Filesystem': fs,
                    'Performance': peak_perf
                })
    
    if sk_filesystem_performance:
        sk_fs_df = pd.DataFrame(sk_filesystem_performance)
        
        sk_xfs_vals = [sk_fs_df[(sk_fs_df['Config'] == config) & (sk_fs_df['Filesystem'] == 'XFS')]['Performance'].iloc[0] 
                      if not sk_fs_df[(sk_fs_df['Config'] == config) & (sk_fs_df['Filesystem'] == 'XFS')].empty else 0 
                      for config in configs]
        sk_ext4_vals = [sk_fs_df[(sk_fs_df['Config'] == config) & (sk_fs_df['Filesystem'] == 'Ext4')]['Performance'].iloc[0] 
                       if not sk_fs_df[(sk_fs_df['Config'] == config) & (sk_fs_df['Filesystem'] == 'Ext4')].empty else 0 
                       for config in configs]
        
        bars5 = ax3.bar(x_pos - width/2, sk_xfs_vals, width, label='XFS', color=fs_colors['XFS'])
        bars6 = ax3.bar(x_pos + width/2, sk_ext4_vals, width, label='Ext4', color=fs_colors['Ext4'])
        
        ax3.set_xlabel('Array Configuration')
        ax3.set_ylabel('Peak Read Bandwidth (GB/s)')
        ax3.set_title('SK Hynix Drives: Filesystem Comparison (Read)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(configs)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        for bars in [bars5, bars6]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
    
    # SK Hynix Write Performance by Filesystem
    sk_write = sk_data[sk_data['Operation'] == 'write']
    
    sk_filesystem_performance_write = []
    for fs in ['XFS', 'Ext4']:
        for array_type in ['2x2_Parallel', '4Drive_Single']:
            fs_array_data = sk_write[(sk_write['Filesystem'] == fs) & (sk_write['ArrayType'] == array_type)]
            if not fs_array_data.empty:
                peak_perf = fs_array_data['Bandwidth_GBps'].max()
                sk_filesystem_performance_write.append({
                    'Config': f'{array_type.replace("_", " ")}',
                    'Filesystem': fs,
                    'Performance': peak_perf
                })
    
    if sk_filesystem_performance_write:
        sk_fs_write_df = pd.DataFrame(sk_filesystem_performance_write)
        
        sk_xfs_write_vals = [sk_fs_write_df[(sk_fs_write_df['Config'] == config) & (sk_fs_write_df['Filesystem'] == 'XFS')]['Performance'].iloc[0] 
                            if not sk_fs_write_df[(sk_fs_write_df['Config'] == config) & (sk_fs_write_df['Filesystem'] == 'XFS')].empty else 0 
                            for config in configs]
        sk_ext4_write_vals = [sk_fs_write_df[(sk_fs_write_df['Config'] == config) & (sk_fs_write_df['Filesystem'] == 'Ext4')]['Performance'].iloc[0] 
                             if not sk_fs_write_df[(sk_fs_write_df['Config'] == config) & (sk_fs_write_df['Filesystem'] == 'Ext4')].empty else 0 
                             for config in configs]
        
        bars7 = ax4.bar(x_pos - width/2, sk_xfs_write_vals, width, label='XFS', color=fs_colors['XFS'])
        bars8 = ax4.bar(x_pos + width/2, sk_ext4_write_vals, width, label='Ext4', color=fs_colors['Ext4'])
        
        ax4.set_xlabel('Array Configuration')
        ax4.set_ylabel('Peak Write Bandwidth (GB/s)')
        ax4.set_title('SK Hynix Drives: Filesystem Comparison (Write)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(configs)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        for bars in [bars7, bars8]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax4.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/filesystem_peak_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance vs Block Size Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Read performance vs block size (4-drive single arrays)
    read_4drive = df[(df['Operation'] == 'read') & (df['ArrayType'] == '4Drive_Single')]
    
    for drive_type in ['WD', 'SKHynix']:
        for filesystem in ['XFS', 'Ext4']:
            drive_fs_data = read_4drive[(read_4drive['DriveType'] == drive_type) & 
                                       (read_4drive['Filesystem'] == filesystem)]
            if not drive_fs_data.empty:
                peak_by_bs = drive_fs_data.groupby('BlockSize')['Bandwidth_GBps'].max()
                peak_by_bs_sorted = peak_by_bs.reindex(all_block_sizes).dropna()
                
                linestyle = '-' if drive_type == 'WD' else '--'
                ax1.plot(peak_by_bs_sorted.index, peak_by_bs_sorted.values,
                        marker='o', linewidth=3, markersize=6, linestyle=linestyle,
                        color=fs_colors[filesystem], 
                        label=f'{drive_type} {filesystem}')
    
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Peak Read Bandwidth (GB/s)')
    ax1.set_title('Read Performance vs Block Size\n(4-Drive Single Arrays)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Write performance vs block size (4-drive single arrays)
    write_4drive = df[(df['Operation'] == 'write') & (df['ArrayType'] == '4Drive_Single')]
    
    for drive_type in ['WD', 'SKHynix']:
        for filesystem in ['XFS', 'Ext4']:
            drive_fs_data = write_4drive[(write_4drive['DriveType'] == drive_type) & 
                                        (write_4drive['Filesystem'] == filesystem)]
            if not drive_fs_data.empty:
                peak_by_bs = drive_fs_data.groupby('BlockSize')['Bandwidth_GBps'].max()
                peak_by_bs_sorted = peak_by_bs.reindex(all_block_sizes).dropna()
                
                linestyle = '-' if drive_type == 'WD' else '--'
                ax2.plot(peak_by_bs_sorted.index, peak_by_bs_sorted.values,
                        marker='s', linewidth=3, markersize=6, linestyle=linestyle,
                        color=fs_colors[filesystem], 
                        label=f'{drive_type} {filesystem}')
    
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Peak Write Bandwidth (GB/s)')
    ax2.set_title('Write Performance vs Block Size\n(4-Drive Single Arrays)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/filesystem_performance_vs_blocksize.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Filesystem Performance Ratio Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate XFS vs Ext4 performance ratios
    ratio_data = []
    
    for drive_type in ['WD', 'SKHynix']:
        for array_type in ['2x2_Parallel', '4Drive_Single']:
            for operation in ['read', 'write']:
                
                xfs_data = df[(df['DriveType'] == drive_type) & 
                             (df['ArrayType'] == array_type) & 
                             (df['Operation'] == operation) & 
                             (df['Filesystem'] == 'XFS')]
                
                ext4_data = df[(df['DriveType'] == drive_type) & 
                              (df['ArrayType'] == array_type) & 
                              (df['Operation'] == operation) & 
                              (df['Filesystem'] == 'Ext4')]
                
                if not xfs_data.empty and not ext4_data.empty:
                    xfs_peak = xfs_data['Bandwidth_GBps'].max()
                    ext4_peak = ext4_data['Bandwidth_GBps'].max()
                    
                    if ext4_peak > 0:
                        ratio = xfs_peak / ext4_peak
                        ratio_data.append({
                            'Configuration': f'{drive_type} {array_type.replace("_", " ")} {operation.title()}',
                            'Ratio': ratio,
                            'XFS_Performance': xfs_peak,
                            'Ext4_Performance': ext4_peak
                        })
    
    if ratio_data:
        ratio_df = pd.DataFrame(ratio_data)
        
        bars = ax.bar(range(len(ratio_df)), ratio_df['Ratio'],
                     color=['#2E86AB' if ratio >= 1.0 else '#F18F01' for ratio in ratio_df['Ratio']])
        
        # Add horizontal line at 1.0 (equal performance)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, 
                  label='Equal Performance', alpha=0.7)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Performance Ratio (XFS / Ext4)')
        ax.set_title('Filesystem Performance Comparison\n(>1.0 = XFS Better, <1.0 = Ext4 Better)')
        ax.set_xticks(range(len(ratio_df)))
        ax.set_xticklabels(ratio_df['Configuration'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ratio = ratio_df.iloc[i]['Ratio']
            ax.annotate(f'{ratio:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/filesystem_performance_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Filesystem comparison charts saved to {output_dir}/")
    print("Generated charts:")
    print("  - filesystem_peak_performance_comparison.png")
    print("  - filesystem_performance_vs_blocksize.png")
    print("  - filesystem_performance_ratios.png")

def print_filesystem_analysis(df):
    """Print detailed filesystem comparison analysis"""
    print("\n=== FILESYSTEM COMPARISON ANALYSIS ===")
    
    for drive_type in ['WD', 'SKHynix']:
        print(f"\n{drive_type} Drives:")
        drive_data = df[df['DriveType'] == drive_type]
        
        for array_type in ['2x2_Parallel', '4Drive_Single']:
            print(f"\n  {array_type.replace('_', ' ')}:")
            array_data = drive_data[drive_data['ArrayType'] == array_type]
            
            for operation in ['read', 'write']:
                op_data = array_data[array_data['Operation'] == operation]
                
                if not op_data.empty:
                    xfs_peak = op_data[op_data['Filesystem'] == 'XFS']['Bandwidth_GBps'].max() if not op_data[op_data['Filesystem'] == 'XFS'].empty else 0
                    ext4_peak = op_data[op_data['Filesystem'] == 'Ext4']['Bandwidth_GBps'].max() if not op_data[op_data['Filesystem'] == 'Ext4'].empty else 0
                    
                    print(f"    {operation.title()}:")
                    print(f"      XFS:  {xfs_peak:.1f} GB/s")
                    print(f"      Ext4: {ext4_peak:.1f} GB/s")
                    
                    if ext4_peak > 0:
                        ratio = xfs_peak / ext4_peak
                        if ratio > 1.05:
                            winner = f"XFS wins by {((ratio-1)*100):.1f}%"
                        elif ratio < 0.95:
                            winner = f"Ext4 wins by {((1/ratio-1)*100):.1f}%"
                        else:
                            winner = "Performance is equal"
                        print(f"      Result: {winner}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 filesystem_comparison_charts.py <xfs_csv_file> <ext4_csv_file>")
        print("Example: python3 filesystem_comparison_charts.py /mnt/array_scaling_comparison.csv /mnt/array_scaling_ext4_results.csv")
        sys.exit(1)
    
    xfs_file = sys.argv[1]
    ext4_file = sys.argv[2]
    
    # Load and combine data
    df = load_and_combine_data(xfs_file, ext4_file)
    
    # Create charts
    print("\nGenerating filesystem comparison charts...")
    create_filesystem_comparison_charts(df)
    
    # Print analysis
    print_filesystem_analysis(df)

if __name__ == "__main__":
    main()