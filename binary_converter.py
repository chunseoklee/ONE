#!/usr/bin/env python3
"""
Binary File Converter Script

This script reads a binary file and converts quantized values to float32 values
using the formula: (x - zp) * scale
where x is the input value, zp is zero point, and scale is the scale factor.

Supported input types:
- uint8: 8-bit unsigned integers (0-255)
- int16: 16-bit signed integers (-32768 to 32767)

Usage:
    python binary_converter.py <input_file> <output_file> <scale> <zp> [--type uint8|int16]

Arguments:
    input_file  : Path to input binary file
    output_file : Path to output binary file (will contain float32 values)
    scale       : Scale factor (float)
    zp          : Zero point (integer)
    --type      : Input data type (uint8 or int16, default: uint8)
"""

import sys
import struct
import argparse


def convert_binary_file(input_path, output_path, scale, zp, data_type='uint8'):
    """
    Convert binary file using quantization parameters.
    
    Args:
        input_path (str): Path to input binary file
        output_path (str): Path to output binary file
        scale (float): Scale factor for conversion
        zp (int): Zero point for conversion
        data_type (str): Input data type ('uint8' or 'int16')
    """
    try:
        # Read input binary file
        with open(input_path, 'rb') as input_file:
            input_data = input_file.read()
        
        print(f"Read {len(input_data)} bytes from {input_path}")
        
        # Parse input data based on type
        parsed_values = []
        if data_type == 'uint8':
            # Each byte is a uint8 value
            parsed_values = list(input_data)
            element_size = 1
        elif data_type == 'int16':
            # Every 2 bytes is an int16 value (little endian)
            if len(input_data) % 2 != 0:
                print("Warning: Input file size is not multiple of 2, last byte will be ignored")
            element_size = 2
            for i in range(0, len(input_data) - 1, 2):
                # Unpack as little-endian int16
                int16_val = struct.unpack('<h', input_data[i:i+2])[0]
                parsed_values.append(int16_val)
        else:
            print(f"Error: Unsupported data type '{data_type}'")
            return False
        
        print(f"Parsed {len(parsed_values)} {data_type} values from {len(input_data)} bytes")
        
        # Convert each value using the formula: (x - zp) * scale
        float_values = []
        print(f"\nFirst 10 conversions (scale={scale}, zp={zp}, type={data_type}):")
        for i, val in enumerate(parsed_values):
            # Apply conversion formula
            converted_value = (val - zp) * scale
            float_values.append(converted_value)
            
            # Display first 10 conversions with details
            if i < 10:
                if data_type == 'uint8':
                    octal_val = oct(val)
                    print(f"  [{i+1}] uint8: {val:3d} (octal: {octal_val}) -> ({val} - {zp}) * {scale} = {converted_value:.6f}")
                else:  # int16
                    print(f"  [{i+1}] int16: {val:6d} -> ({val} - {zp}) * {scale} = {converted_value:.6f}")
        
        print(f"Converted {len(float_values)} values using formula: (x - {zp}) * {scale}")
        
        # Write float values to output binary file
        with open(output_path, 'wb') as output_file:
            for float_val in float_values:
                # Pack as 32-bit float (little endian)
                packed_float = struct.pack('<f', float_val)
                output_file.write(packed_float)
        
        print(f"Written {len(float_values)} float32 values to {output_path}")
        
        # Display some statistics
        if float_values:
            print(f"\nConversion Statistics:")
            print(f"  Input range: {min(parsed_values)} to {max(parsed_values)}")
            print(f"  Output range: {min(float_values):.6f} to {max(float_values):.6f}")
            print(f"  Output mean: {sum(float_values)/len(float_values):.6f}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return False
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert binary file using quantization parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python binary_converter.py input.bin output.bin 0.015625 128
    python binary_converter.py ws_input_q8.bin.0 ws_input_f32.bin.0 0.00392157 0
    python binary_converter.py input_int16.bin output.bin 0.01 0 --type int16
        """
    )
    
    parser.add_argument('input_file', help='Path to input binary file')
    parser.add_argument('output_file', help='Path to output binary file')
    parser.add_argument('scale', type=float, help='Scale factor (float)')
    parser.add_argument('zp', type=int, help='Zero point (integer)')
    parser.add_argument('--type', choices=['uint8', 'int16'], default='uint8',
                       help='Input data type (default: uint8)')
    
    args = parser.parse_args()
    
    # Validate zero point range based on data type
    if args.type == 'uint8':
        if not (0 <= args.zp <= 255):
            print("Error: Zero point (zp) should be in range 0-255 for uint8 values.")
            sys.exit(1)
    elif args.type == 'int16':
        if not (-32768 <= args.zp <= 32767):
            print("Error: Zero point (zp) should be in range -32768 to 32767 for int16 values.")
            sys.exit(1)
    
    print(f"Binary File Converter")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Scale: {args.scale}")
    print(f"Zero point: {args.zp}")
    print(f"Data type: {args.type}")
    print(f"Conversion formula: (x - {args.zp}) * {args.scale}")
    print("-" * 50)
    
    success = convert_binary_file(args.input_file, args.output_file, args.scale, args.zp, args.type)
    
    if success:
        print("\nConversion completed successfully!")
    else:
        print("\nConversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
