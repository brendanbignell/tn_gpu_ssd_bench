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

def create_array_scaling_charts(df, output_dir="."):
    """Create array scaling comparison charts"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Color scheme
    colors = {
        '2x2_Parallel': '#E74C3C',    # Red
        '4Drive_Single': '#3498DB'    # Blue
    }
    
    # Get properly sorted block sizes
    all_block_sizes = sort_block_sizes(df['BlockSize'].unique())
    
    # Extract drive types and array types
    df['DriveType'] = df['ArrayConfig'].str.extract(r'(WD|SKHynix)')
    
    # 1. Peak Performance Comparison: 2x2 vs 4-Drive
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # WD Read Performance
    wd_data = df[df['DriveType'] == 'WD']
    wd_read = wd_data[wd_data['Operation'] == 'read']
    
    peak_wd_2x2_read = wd_read[wd_read['ArrayType'] == '2x2_Parallel']['Bandwidth_GBps'].max()
    peak_wd_4drive_read = wd_read[wd_read['ArrayType'] == '4Drive_Single']['Bandwidth_GBps'].max()
    
    bars1 = ax1.bar(['2x2 Parallel', '4-Drive Single'], 
                    [peak_wd_2x2_read, peak_wd_4drive_read],
                    color=[colors['2x2_Parallel'], colors['4Drive_Single']])
    ax1.set_ylabel('Peak Read Bandwidth (GB/s)')
    ax1.set_title('WD Drives: Peak Read Performance')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f} GB/s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    # WD Write Performance
    wd_write = wd_data[wd_data['Operation'] == 'write']
    peak_wd_2x2_write = wd_write[wd_write['ArrayType'] == '2x2_Parallel']['Bandwidth_GBps'].max()
    peak_wd_4drive_write = wd_write[wd_write['ArrayType'] == '4Drive_Single']['Bandwidth_GBps'].max()
    
    bars2 = ax2.bar(['2x2 Parallel', '4-Drive Single'],
                    [peak_wd_2x2_write, peak_wd_4drive_write],
                    color=[colors['2x2_Parallel'], colors['4Drive_Single']])
    ax2.set_ylabel('Peak Write Bandwidth (GB/s)')
    ax2.set_title('WD Drives: Peak Write Performance')
    ax2.grid(True, alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f} GB/s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    # SK Hynix Read Performance
    sk_data = df[df['DriveType'] == 'SKHynix']
    sk_read = sk_data[sk_data['Operation'] == 'read']
    
    peak_sk_2x2_read = sk_read[sk_read['ArrayType'] == '2x2_Parallel']['Bandwidth_GBps'].max()
    peak_sk_4drive_read = sk_read[sk_read['ArrayType'] == '4Drive_Single']['Bandwidth_GBps'].max()
    
    bars3 = ax3.bar(['2x2 Parallel', '4-Drive Single'],
                    [peak_sk_2x2_read, peak_sk_4drive_read],
                    color=[colors['2x2_Parallel'], colors['4Drive_Single']])
    ax3.set_ylabel('Peak Read Bandwidth (GB/s)')
    ax3.set_title('SK Hynix Drives: Peak Read Performance')
    ax3.grid(True, alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f} GB/s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    # SK Hynix Write Performance
    sk_write = sk_data[sk_data['Operation'] == 'write']
    peak_sk_2x2_write = sk_write[sk_write['ArrayType'] == '2x2_Parallel']['Bandwidth_GBps'].max()
    peak_sk_4drive_write = sk_write[sk_write['ArrayType'] == '4Drive_Single']['Bandwidth_GBps'].max()
    
    bars4 = ax4.bar(['2x2 Parallel', '4-Drive Single'],
                    [peak_sk_2x2_write, peak_sk_4drive_write],
                    color=[colors['2x2_Parallel'], colors['4Drive_Single']])
    ax4.set_ylabel('Peak Write Bandwidth (GB/s)')
    ax4.set_title('SK Hynix Drives: Peak Write Performance')
    ax4.grid(True, alpha=0.3)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f} GB/s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/peak_performance_2x2_vs_4drive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance vs Block Size Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # WD drives performance vs block size
    for array_type in ['2x2_Parallel', '4Drive_Single']:
        wd_type_data = wd_read[wd_read['ArrayType'] == array_type]
        if not wd_type_data.empty:
            peak_by_bs = wd_type_data.groupby('BlockSize')['Bandwidth_GBps'].max()
            peak_by_bs_sorted = peak_by_bs.reindex(all_block_sizes).dropna()
            
            label = '2x2 Parallel' if array_type == '2x2_Parallel' else '4-Drive Single'
            ax1.plot(peak_by_bs_sorted.index, peak_by_bs_sorted.values,
                    marker='o', linewidth=3, markersize=8,
                    color=colors[array_type], label=label)
    
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Peak Read Bandwidth (GB/s)')
    ax1.set_title('WD Drives: Read Performance vs Block Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SK Hynix drives performance vs block size
    for array_type in ['2x2_Parallel', '4Drive_Single']:
        sk_type_data = sk_read[sk_read['ArrayType'] == array_type]
        if not sk_type_data.empty:
            peak_by_bs = sk_type_data.groupby('BlockSize')['Bandwidth_GBps'].max()
            peak_by_bs_sorted = peak_by_bs.reindex(all_block_sizes).dropna()
            
            label = '2x2 Parallel' if array_type == '2x2_Parallel' else '4-Drive Single'
            ax2.plot(peak_by_bs_sorted.index, peak_by_bs_sorted.values,
                    marker='s', linewidth=3, markersize=8,
                    color=colors[array_type], label=label)
    
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Peak Read Bandwidth (GB/s)')
    ax2.set_title('SK Hynix Drives: Read Performance vs Block Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_vs_blocksize_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scaling Efficiency Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate scaling efficiency (how close to 2x performance we get)
    efficiency_data = []
    
    for drive_type in ['WD', 'SKHynix']:
        for operation in ['read', 'write']:
            drive_op_data = df[(df['DriveType'] == drive_type) & (df['Operation'] == operation)]
            
            if not drive_op_data.empty:
                peak_2x2 = drive_op_data[drive_op_data['ArrayType'] == '2x2_Parallel']['Bandwidth_GBps'].max()
                peak_4drive = drive_op_data[drive_op_data['ArrayType'] == '4Drive_Single']['Bandwidth_GBps'].max()
                
                if peak_4drive > 0:
                    efficiency = (peak_4drive / peak_2x2) if peak_2x2 > 0 else 0
                    efficiency_data.append({
                        'Configuration': f'{drive_type} {operation.title()}',
                        'Efficiency': efficiency,
                        'Peak_2x2': peak_2x2,
                        'Peak_4Drive': peak_4drive
                    })
    
    if efficiency_data:
        efficiency_df = pd.DataFrame(efficiency_data)
        
        bars = ax.bar(range(len(efficiency_df)), efficiency_df['Efficiency'],
                     color=['#E74C3C' if 'read' in config.lower() else '#3498DB' 
                           for config in efficiency_df['Configuration']])
        
        # Add horizontal line at 2.0 (perfect scaling)
        ax.axhline(y=2.0, color='green', linestyle='--', linewidth=2, 
                  label='Perfect 2x Scaling', alpha=0.7)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Scaling Factor (4-Drive / 2x2)')
        ax.set_title('Array Scaling Efficiency\n(Higher = Better, 2.0 = Perfect Scaling)')
        ax.set_xticks(range(len(efficiency_df)))
        ax.set_xticklabels(efficiency_df['Configuration'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            efficiency = efficiency_df.iloc[i]['Efficiency']
            ax.annotate(f'{efficiency:.2f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance Summary Table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table data
    summary_data = []
    for drive_type in ['WD', 'SKHynix']:
        for array_type in ['2x2_Parallel', '4Drive_Single']:
            type_data = df[(df['DriveType'] == drive_type) & (df['ArrayType'] == array_type)]
            
            if not type_data.empty:
                read_data = type_data[type_data['Operation'] == 'read']
                write_data = type_data[type_data['Operation'] == 'write']
                
                peak_read = read_data['Bandwidth_GBps'].max() if not read_data.empty else 0
                peak_write = write_data['Bandwidth_GBps'].max() if not write_data.empty else 0
                peak_iops = type_data['IOPS'].max() if not type_data.empty else 0
                
                array_label = '2x2 Parallel' if array_type == '2x2_Parallel' else '4-Drive Single'
                
                summary_data.append([
                    f'{drive_type} {array_label}',
                    f'{peak_read:.1f}',
                    f'{peak_write:.1f}',
                    f'{peak_iops:,.0f}',
                    f'{peak_read/peak_write:.2f}' if peak_write > 0 else 'N/A'
                ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Configuration', 'Peak Read\n(GB/s)', 'Peak Write\n(GB/s)', 
                              'Peak IOPS', 'Read/Write\nRatio'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Color rows alternately
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            if '2x2' in summary_data[i][0]:
                table[(i+1, j)].set_facecolor('#FFE5E5')  # Light red
            else:
                table[(i+1, j)].set_facecolor('#E5F2FF')  # Light blue
    
    ax.set_title('Array Scaling Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Array scaling charts saved to {output_dir}/")
    print("Generated charts:")
    print("  - peak_performance_2x2_vs_4drive.png")
    print("  - performance_vs_blocksize_scaling.png")
    print("  - scaling_efficiency_analysis.png")
    print("  - scaling_summary_table.png")

def print_scaling_analysis(df):
    """Print detailed scaling analysis"""
    print("\n=== ARRAY SCALING ANALYSIS ===")
    
    # Extract drive types
    df['DriveType'] = df['ArrayConfig'].str.extract(r'(WD|SKHynix)')
    
    for drive_type in ['WD', 'SKHynix']:
        print(f"\n{drive_type} Drives:")
        drive_data = df[df['DriveType'] == drive_type]
        
        for operation in ['read', 'write']:
            op_data = drive_data[drive_data['Operation'] == operation]
            
            if not op_data.empty:
                peak_2x2 = op_data[op_data['ArrayType'] == '2x2_Parallel']['Bandwidth_GBps'].max()
                peak_4drive = op_data[op_data['ArrayType'] == '4Drive_Single']['Bandwidth_GBps'].max()
                
                scaling_factor = peak_4drive / peak_2x2 if peak_2x2 > 0 else 0
                efficiency_percent = (scaling_factor / 2.0) * 100 if scaling_factor > 0 else 0
                
                print(f"  {operation.title()} Performance:")
                print(f"    2x2 Parallel:   {peak_2x2:.1f} GB/s")
                print(f"    4-Drive Single: {peak_4drive:.1f} GB/s")
                print(f"    Scaling Factor: {scaling_factor:.2f}x")
                print(f"    Efficiency:     {efficiency_percent:.1f}% of perfect scaling")
                
                if scaling_factor > 1.8:
                    print(f"    ✅ Excellent scaling")
                elif scaling_factor > 1.5:
                    print(f"    ⚠️  Good scaling") 
                else:
                    print(f"    ❌ Poor scaling")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 array_scaling_charts.py <scaling_csv_file>")
        print("Example: python3 array_scaling_charts.py /mnt/array_scaling_comparison.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please run the array scaling benchmark first.")
        sys.exit(1)
    
    print(f"Loading array scaling data from {csv_file}...")
    print(f"Loaded {len(df)} test results")
    print(f"Array configurations: {df['ArrayConfig'].unique()}")
    print(f"Array types: {df['ArrayType'].unique()}")
    
    # Create charts
    print("\nGenerating array scaling charts...")
    create_array_scaling_charts(df)
    
    # Print analysis
    print_scaling_analysis(df)

if __name__ == "__main__":
    main()