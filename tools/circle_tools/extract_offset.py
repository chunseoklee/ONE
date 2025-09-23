#!/usr/bin/env python3

import yaml
import struct
import sys
import copy
import os


def split_tiles(tiles, min_ch_size):
    """
    Split a tile, whose out_ch size is larger than min_ch_size,
    into min ch size slice
    """
    new_tiles = []
    for t in tiles:
        out_tile_size = t['shape']['K']
        tile_size = t['size']
        split_size = int(out_tile_size / min_ch_size)
        for i in range(split_size):
            new_tile = copy.deepcopy(t)
            new_tile['offset']['out_ch'] += i * min_ch_size
            new_tile['mem_offset'] = t['mem_offset'] + i*int(tile_size/split_size)
            new_tiles.append(new_tile)
    return new_tiles


def extract_offsets(yaml_file_path, output_prefix=None, min_ch_size_default=32):
    """
    Extract offset and stride information from YAML file containing MatMul operations.

    Args:
        yaml_file_path (str): Path to the input YAML file
        output_prefix (str, optional): Prefix for output files. If None, generates from filename
        min_ch_size_default (int): Default minimum channel size (default: 32)

    Returns:
        dict: Dictionary containing 'offset_result' and 'inch_stride_result'
    """
    # Generate output prefix if not provided
    if output_prefix is None:
        bname = os.path.basename(yaml_file_path)
        output_prefix = f'l.{bname.split(".")[1]}'

    # Load YAML data
    with open(yaml_file_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    offset_result = {}
    inch_stride_result = {}

    for key in data.keys():
        if "MatMul" in key:
            print(f'Generating oparam for {key}')
            matmul_type = data[key]['Type']
            if matmul_type == 'NConvW8A16':
                min_ch_size = 16
            else:
                min_ch_size = min_ch_size_default

            tiles = data[key]['Tiles']
            print(f'# of Tile is {len(tiles)}')

            in_tile_size = tiles[0]['shape']['D']
            for i in range(min(5, len(tiles))):
                if in_tile_size < tiles[i]['shape']['D']:
                    in_tile_size = tiles[i]['shape']['D']

            tiles = split_tiles(tiles, min_ch_size) # split along output channel dimension
            tiles = sorted(tiles, key=lambda item: (item['offset']['out_ch'], item['offset']['in_ch']))

            # Handle key extraction more safely
            key_parts = key.split('/')
            if "mlp" in os.path.basename(yaml_file_path):
                if len(key_parts) > 1:
                    new_key = key_parts[1]
                else:
                    new_key = key
            else:
                if len(key_parts) > 2:
                    new_key = key_parts[2]
                else:
                    new_key = key

            # Sanitize key for use in filename (replace slashes with underscores)
            new_key = new_key.replace('/', '_').replace('\\', '_')
            offset_result[new_key] = [item['mem_offset'] for item in tiles]
            inch_stride_result[new_key] = in_tile_size

    # Write output files
    for k, v in offset_result.items():
        print(f'key: {k}, nelements: {len(v)}')
        with open(f'{output_prefix}.{k}.op.bin', 'wb') as f:
            f.write(struct.pack(f"{len(v)}I", *v))

    for k, v in inch_stride_result.items():
        print(f'in channel tile size: {v}')
        with open(f'{output_prefix}.{k}.ics.bin', 'wb') as f:
            f.write(struct.pack("<I", v))

    return {
        'offset_result': offset_result,
        'inch_stride_result': inch_stride_result
    }


def main():
    """Command-line interface for extract_offsets"""
    if len(sys.argv) < 2:
        print("Usage: python extract_offset.py <yaml_file_path> [output_prefix] [min_ch_size_default]")
        sys.exit(1)

    yaml_file_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else None
    min_ch_size_default = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    bname = os.path.basename(yaml_file_path)
    print(bname)

    result = extract_offsets(yaml_file_path, output_prefix, min_ch_size_default)

    return result


if __name__ == "__main__":
    main()
