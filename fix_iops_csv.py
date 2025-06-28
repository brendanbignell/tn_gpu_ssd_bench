#!/usr/bin/env python3

import pandas as pd
import sys

def convert_block_size_to_mb(block_size_str):
    """Convert block size to MB"""
    size_str = block_size_str.lower()
    
    if size_str.endswith('k'):
        return float(size_str[:-1]) / 1024
    elif size_str.endswith('m'):
        return float(size_str[:-1])
    elif size_str.endswith('g'):
        return float(size_str[:-1]) * 1024
    else:
        # Assume bytes
        return float(size_str) / 1024 / 1024

def fix_iops_in_csv(input_file, output_file=None):
    """Fix IOPS calculations in existing CSV files"""
    
    if output_file is None:
        output_file = input_file.replace('.csv', '_fixed_iops.csv')
    
    # Load the CSV
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} rows from {input_file}")
    print(f"Columns: {list(df.columns)}")
    
    # Calculate IOPS from bandwidth and block size
    def calculate_iops(row):
        bandwidth_mbps = row['Bandwidth_MBps']
        block_size_str = row['BlockSize']
        
        if bandwidth_mbps == 0:
            return 0
        
        block_size_mb = convert_block_size_to_mb(block_size_str)
        iops = bandwidth_mbps / block_size_mb
        return int(iops)
    
    # Apply the IOPS calculation
    df['IOPS'] = df.apply(calculate_iops, axis=1)
    
    # Save the corrected CSV
    df.to_csv(output_file, index=False)
    
    print(f"\nFixed IOPS calculations and saved to: {output_file}")
    
    # Show some examples
    print(f"\nSample corrected results:")
    print(df[['DriveType', 'Operation', 'BlockSize', 'Bandwidth_GBps', 'IOPS']].head(10))
    
    # Show peak IOPS by drive type
    print(f"\n=== Peak IOPS by Drive Type ===")
    for drive_type in df['DriveType'].unique():
        drive_data = df[df['DriveType'] == drive_type]
        peak_iops = drive_data['IOPS'].max()
        peak_row = drive_data.loc[drive_data['IOPS'].idxmax()]
        print(f"{drive_type}: {peak_iops:,} IOPS ({peak_row['BlockSize']} blocks)")

def main():
    if len(sys.argv) not in [2, 3]:
        print("Usage: python3 fix_iops_csv.py <input_csv> [output_csv]")
        print("Example: python3 fix_iops_csv.py /mnt/drive_comparison_results.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else None
    
    fix_iops_in_csv(input_file, output_file)

if __name__ == "__main__":
    main()