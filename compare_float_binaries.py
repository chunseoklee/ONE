#!/usr/bin/env python3
"""
Script to compare two binary files containing float values and generate error statistics.
Analyzes relative errors in ranges from 0% to 50% and provides detailed statistics.
"""

import sys
import struct
import argparse
from collections import defaultdict


def read_float_binary_file(filename):
    """
    Read a binary file and return a list of float values.
    
    Args:
        filename (str): Path to the binary file
        
    Returns:
        list: List of float values read from the file
    """
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        
        # Each float is 4 bytes
        if len(data) % 4 != 0:
            raise ValueError(f"File size ({len(data)} bytes) is not multiple of 4 (float size)")
        
        # Unpack binary data as little-endian floats
        num_floats = len(data) // 4
        floats = struct.unpack(f'<{num_floats}f', data)
        
        return list(floats)
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except struct.error as e:
        print(f"Error reading binary file '{filename}': {e}")
        return None


def calculate_relative_error(val1, val2):
    """
    Calculate relative error between two float values.
    
    Args:
        val1 (float): First value (reference)
        val2 (float): Second value
        
    Returns:
        float: Relative error as a percentage
    """
    # Handle special cases
    if val1 == 0.0 and val2 == 0.0:
        return 0.0  # Both zero, no error
    elif val1 == 0.0:
        # Reference is zero, use absolute difference
        return abs(val2) * 100.0
    else:
        # Calculate relative error as percentage
        return abs((val1 - val2) / val1) * 100.0


def analyze_error_statistics(arr1, arr2, max_error_range=50.0):
    """
    Analyze error statistics between two float arrays.
    
    Args:
        arr1 (list): First array of floats (reference)
        arr2 (list): Second array of floats
        max_error_range (float): Maximum error range to analyze (default: 50.0%)
        
    Returns:
        dict: Comprehensive error statistics
    """
    if len(arr1) != len(arr2):
        return {
            'error': f"Arrays have different lengths: {len(arr1)} vs {len(arr2)}",
            'total_elements': max(len(arr1), len(arr2))
        }
    
    total_elements = len(arr1)
    errors = []
    exact_matches = 0
    zero_ref_matches = 0
    
    # Calculate errors for all elements
    for val1, val2 in zip(arr1, arr2):
        error = calculate_relative_error(val1, val2)
        errors.append(error)
        
        if error == 0.0:
            exact_matches += 1
        if val1 == 0.0:
            zero_ref_matches += 1
    
    # Sort errors for percentile calculations
    errors.sort()
    
    # Calculate basic statistics
    mean_error = sum(errors) / total_elements
    max_error = max(errors)
    min_error = min(errors)
    
    # Calculate percentiles
    percentiles = {}
    for p in [25, 50, 75, 90, 95, 99]:
        idx = int(total_elements * p / 100)
        percentiles[f'p{p}'] = errors[idx] if idx < total_elements else errors[-1]
    
    # Create error ranges (0-1%, 1-5%, 5-10%, 10-20%, 20-50%, >50%)
    error_ranges = [
        (0.0, 1.0, "0-1%"),
        (1.0, 5.0, "1-5%"),
        (5.0, 10.0, "5-10%"),
        (10.0, 20.0, "10-20%"),
        (20.0, 50.0, "20-50%"),
        (50.0, float('inf'), ">50%")
    ]
    
    range_counts = defaultdict(int)
    for error in errors:
        for min_err, max_err, label in error_ranges:
            if min_err <= error < max_err:
                range_counts[label] += 1
                break
    
    # Count elements with error > 1%, 5%, 10%, 20%, 50%
    threshold_counts = {}
    for threshold in [1.0, 5.0, 10.0, 20.0, 50.0]:
        count = sum(1 for error in errors if error > threshold)
        threshold_counts[f'>{threshold}%'] = count
    
    return {
        'total_elements': total_elements,
        'exact_matches': exact_matches,
        'zero_ref_matches': zero_ref_matches,
        'mean_error': mean_error,
        'max_error': max_error,
        'min_error': min_error,
        'percentiles': percentiles,
        'error_ranges': dict(range_counts),
        'threshold_counts': threshold_counts,
        'errors': errors
    }


def print_error_statistics(stats, tolerance=1.0):
    """
    Print comprehensive error statistics.
    
    Args:
        stats (dict): Error statistics from analyze_error_statistics
        tolerance (float): Tolerance threshold for pass/fail determination
    """
    if 'error' in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    total = stats['total_elements']
    print(f"📊 Error Statistics (Total Elements: {total:,})")
    print("=" * 60)
    
    # Basic statistics
    print(f"📈 Basic Statistics:")
    print(f"  Exact matches (0% error):     {stats['exact_matches']:,} ({stats['exact_matches']/total*100:.2f}%)")
    print(f"  Zero reference values:        {stats['zero_ref_matches']:,} ({stats['zero_ref_matches']/total*100:.2f}%)")
    print(f"  Mean error:                   {stats['mean_error']:.6f}%")
    print(f"  Max error:                    {stats['max_error']:.6f}%")
    print(f"  Min error:                    {stats['min_error']:.6f}%")
    print()
    
    # Percentiles
    print(f"📊 Error Percentiles:")
    for p_name, p_value in stats['percentiles'].items():
        print(f"  {p_name:>3}:                         {p_value:.6f}%")
    print()
    
    # Error ranges
    print(f"📈 Error Distribution (0-50% range):")
    for range_label, count in stats['error_ranges'].items():
        percentage = count / total * 100
        bar_length = int(percentage / 2)  # Scale bar to max 50 characters
        bar = "█" * bar_length
        print(f"  {range_label:>6}: {count:>6,} ({percentage:>5.1f}%) {bar}")
    print()
    
    # Threshold counts
    print(f"🎯 Elements Exceeding Thresholds:")
    for threshold, count in stats['threshold_counts'].items():
        percentage = count / total * 100
        print(f"  {threshold:>5}: {count:>6,} ({percentage:>5.1f}%)")
    print()
    
    # Pass/Fail determination based on tolerance
    elements_within_tolerance = sum(1 for error in stats['errors'] if error <= tolerance)
    elements_exceeding_tolerance = total - elements_within_tolerance
    
    print(f"✅ Pass/Fail Criteria (≤ {tolerance}% tolerance):")
    print(f"  Elements within tolerance:     {elements_within_tolerance:,} ({elements_within_tolerance/total*100:.2f}%)")
    print(f"  Elements exceeding tolerance:  {elements_exceeding_tolerance:,} ({elements_exceeding_tolerance/total*100:.2f}%)")
    
    if elements_exceeding_tolerance == 0:
        print(f"  🎉 RESULT: PASS (All elements within {tolerance}% tolerance)")
    else:
        print(f"  ❌ RESULT: FAIL ({elements_exceeding_tolerance} elements exceed {tolerance}% tolerance)")


def main():
    """Main function to compare two binary files and generate error statistics."""
    parser = argparse.ArgumentParser(
        description="Compare two binary files containing float values and generate comprehensive error statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compare_float_binaries.py file1.bin file2.bin
    python compare_float_binaries.py output.bin golden.bin --tolerance 5.0
    python compare_float_binaries.py test.bin reference.bin --max-error 100.0
        """
    )
    
    parser.add_argument('file1', help='First binary file containing float values')
    parser.add_argument('file2', help='Second binary file containing float values')
    parser.add_argument('--tolerance', type=float, default=1.0,
                       help='Tolerance threshold for pass/fail (default: 1.0%%)')
    parser.add_argument('--max-error', type=float, default=50.0,
                       help='Maximum error range to analyze (default: 50.0%%)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed error information for first 20 elements')
    
    args = parser.parse_args()
    
    print(f"🔍 Comparing binary files with error statistics:")
    print(f"  File 1: {args.file1}")
    print(f"  File 2: {args.file2}")
    print(f"  Tolerance: {args.tolerance}%")
    print(f"  Max Error Range: {args.max_error}%")
    print("=" * 60)
    
    # Read both files
    data1 = read_float_binary_file(args.file1)
    if data1 is None:
        sys.exit(1)
    
    data2 = read_float_binary_file(args.file2)
    if data2 is None:
        sys.exit(1)
    
    print(f"📁 File Information:")
    print(f"  File 1: {len(data1):,} float values")
    print(f"  File 2: {len(data2):,} float values")
    print()
    
    # Analyze error statistics
    stats = analyze_error_statistics(data1, data2, args.max_error)
    
    # Print comprehensive statistics
    print_error_statistics(stats, args.tolerance)
    
    # Verbose mode: show first 20 differences
    if args.verbose and 'errors' in stats:
        print()
        print(f"🔍 Detailed Error Information (First 20 elements):")
        print("-" * 80)
        print(f"{'Index':>6} {'Value 1':>12} {'Value 2':>12} {'Error (%)':>10} {'Status':>8}")
        print("-" * 80)
        
        for i, (val1, val2, error) in enumerate(zip(data1[:20], data2[:20], stats['errors'][:20])):
            status = "✓" if error <= args.tolerance else "✗"
            print(f"{i:>6} {val1:>12.6f} {val2:>12.6f} {error:>10.6f} {status:>8}")
    
    # Return exit code based on tolerance
    if 'error' in stats:
        return 1
    else:
        elements_exceeding_tolerance = stats['threshold_counts'][f'>{args.tolerance}%']
        return 1 if elements_exceeding_tolerance > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
