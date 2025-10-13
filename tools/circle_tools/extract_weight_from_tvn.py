#!/usr/bin/env python3
"""
TVN 파일에서 weight 데이터의 offset을 계산하는 Python 모듈

C++ 함수 load_tvn_weight_buf의 weight 위치 계산 부분만 Python으로 구현:
model_w + NPUBIN_META_SIZE + NPUBIN_META_EXTENDED_SIZE(meta->magiccode) + meta->program_size
"""

import os
import struct
from typing import Optional, Tuple


# Constants
NPUBIN_META_SIZE = 4096  # Base metadata size (4096 bytes)


def npubin_meta_extended_size(magiccode: int) -> int:
    """
    magiccode로부터 extended metadata 크기 계산
    
    Args:
        magiccode: NPU binary magic code
        
    Returns:
        Extended metadata size in bytes
    """
    # C++ 구현: ((magiccode >> 8) & 0xFFULL) * NPUBIN_META_SIZE
    num_extended = (magiccode >> 8) & 0xFF
    return num_extended * NPUBIN_META_SIZE


def parse_minimal_metadata(file_path: str) -> Optional[Tuple[int, int, int]]:
    """
    TVN 파일에서 필수 메타데이터만 추출 (magiccode, program_size, extended_metasize)
    
    Args:
        file_path: TVN 파일 경로
        
    Returns:
        (magiccode, program_size, extended_metasize) 튜플, 실패 시 None
    """
    try:
        with open(file_path, 'rb') as f:
            # Read base metadata (first 4096 bytes)
            metadata = f.read(NPUBIN_META_SIZE)
            
            if len(metadata) < NPUBIN_META_SIZE:
                print(f"Error: File too small, expected at least {NPUBIN_META_SIZE} bytes")
                return None
            
            # Parse magiccode (offset 0, 8 bytes, little-endian)
            magiccode = struct.unpack('<Q', metadata[0:8])[0]
            
            # Parse extended_metasize (offset 188, 4 bytes, little-endian)
            # npubin_meta 구조에서 extended_metasize는 offset 188에 위치
            extended_metasize = struct.unpack('<I', metadata[188:192])[0]
            
            # Parse program_size (offset 224, 8 bytes, little-endian)
            # npubin_meta 구조에서 program_size는 offset 224에 위치
            program_size = struct.unpack('<Q', metadata[224:232])[0]
            
            return magiccode, program_size, extended_metasize
            
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except struct.error as e:
        print(f"Error: Failed to parse metadata: {e}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error reading metadata: {e}")
        return None


def get_tvn_weight_offset(tvn_path) -> Optional[int]:
    """
    TVN 파일에서 weight 데이터의 offset을 계산하여 반환
    
    Args:
        tvn_path: TVN 파일 경로
        
    Returns:
        weight 데이터의 파일 offset (바이트 단위), 실패 시 None
    """
    fname = tvn_path
    
    # 2. 파일에서 메타데이터 읽기
    metadata_result = parse_minimal_metadata(fname)
    if metadata_result is None:
        return None
    
    magiccode, program_size, extended_metasize = metadata_result
    
    # 3. Extended metadata 크기 계산
    # Backward compatibility: extended_metasize가 0이면 legacy 방식 사용
    if extended_metasize == 0:
        extended_size = npubin_meta_extended_size(magiccode)
    else:
        extended_size = extended_metasize
    
    # 4. Weight offset 계산
    # NPUBIN_META_SIZE + extended_size + program_size
    weight_offset = NPUBIN_META_SIZE + extended_size + program_size
    
    print(f"File: {fname}")
    print(f"Magiccode: 0x{magiccode:016x}")
    print(f"Program size: {program_size} bytes")
    print(f"Extended metadata size: {extended_size} bytes")
    print(f"Weight offset: {weight_offset} bytes (0x{weight_offset:x})")
    
    return weight_offset


def get_tvn_weight_info(tvn_path) -> Optional[dict]:
    """
    TVN 파일의 weight 관련 전체 정보 반환
    
    Args:
        tvn_path: tvn file path
        
    Returns:
        Weight 관련 정보를 담은 딕셔너리, 실패 시 None
    """
    fname = tvn_path
    
    metadata_result = parse_minimal_metadata(fname)
    if metadata_result is None:
        return None
    
    magiccode, program_size, extended_metasize = metadata_result
    
    # Extended metadata 크기 계산
    if extended_metasize == 0:
        extended_size = npubin_meta_extended_size(magiccode)
    else:
        extended_size = extended_metasize
    
    # Weight offset 계산
    weight_offset = NPUBIN_META_SIZE + extended_size + program_size
    
    # 파일 크기 확인
    try:
        file_size = os.path.getsize(fname)
    except OSError:
        file_size = 0
    
    # Weight size 계산 (파일 끝에서 weight offset까지)
    weight_size = file_size - weight_offset if file_size > weight_offset else 0
    
    return {
        'filename': fname,
        'magiccode': f"0x{magiccode:016x}",
        'program_size': program_size,
        'extended_metasize': extended_metasize,
        'extended_size': extended_size,
        'weight_offset': weight_offset,
        'weight_size': weight_size,
        'file_size': file_size
    }


def extract_weight_data(tvn_path, output_path="weight.bin") -> bool:
    """
    TVN 파일에서 weight 데이터를 추출하여 별도 파일로 저장
    
    Args:
        tvn_path: TVN 파일 경로
        output_path: 출력 파일 경로 (기본값: "weight.bin")
        
    Returns:
        성공 시 True, 실패 시 False
    """
    # Weight 정보 가져오기
    weight_info = get_tvn_weight_info(tvn_path)
    if weight_info is None:
        print(f"Failed to get weight info from {tvn_path}")
        return False
    
    weight_offset = weight_info['weight_offset']
    weight_size = weight_info['weight_size']
    
    if weight_size == 0:
        print(f"No weight data found in {tvn_path}")
        return False
    
    print(f"Extracting weight data from {tvn_path}")
    print(f"Weight offset: {weight_offset} bytes (0x{weight_offset:x})")
    print(f"Weight size: {weight_size} bytes")
    print(f"Output file: {output_path}")
    
    try:
        # TVN 파일 열기
        with open(tvn_path, 'rb') as tvn_file:
            # Weight offset으로 이동
            tvn_file.seek(weight_offset)
            
            # Weight 데이터 읽기
            weight_data = tvn_file.read(weight_size)
            
            if len(weight_data) != weight_size:
                print(f"Error: Expected {weight_size} bytes, but read {len(weight_data)} bytes")
                return False
            
            # Weight 데이터를 출력 파일에 저장
            with open(output_path, 'wb') as output_file:
                output_file.write(weight_data)
            
            print(f"Successfully extracted {len(weight_data)} bytes to {output_path}")
            return True
            
    except FileNotFoundError:
        print(f"Error: File not found: {tvn_path}")
        return False
    except PermissionError:
        print(f"Error: Permission denied when accessing {tvn_path} or {output_path}")
        return False
    except Exception as e:
        print(f"Error: Unexpected error during extraction: {e}")
        return False


def get_tvn_weight_info_with_extraction(tvn_path, extract_weights=False, output_path="weight.bin") -> Optional[dict]:
    """
    TVN 파일의 weight 관련 전체 정보 반환 및 선택적 weight 데이터 추출
    
    Args:
        tvn_path: TVN 파일 경로
        extract_weights: weight 데이터 추출 여부 (기본값: False)
        output_path: 출력 파일 경로 (기본값: "weight.bin")
        
    Returns:
        Weight 관련 정보를 담은 딕셔너리, 실패 시 None
    """
    # Weight 정보 가져오기
    weight_info = get_tvn_weight_info(tvn_path)
    if weight_info is None:
        return None
    
    # Weight 정보 출력
    print(f"\n=== TVN Weight Information ===")
    print(f"File: {weight_info['filename']}")
    print(f"Magiccode: {weight_info['magiccode']}")
    print(f"Program size: {weight_info['program_size']:,} bytes")
    print(f"Extended metadata size: {weight_info['extended_size']:,} bytes")
    print(f"Weight offset: {weight_info['weight_offset']:,} bytes (0x{weight_info['weight_offset']:x})")
    print(f"Weight size: {weight_info['weight_size']:,} bytes")
    print(f"File size: {weight_info['file_size']:,} bytes")
    
    # Weight 데이터 추출 (요청된 경우)
    if extract_weights:
        print(f"\n=== Extracting Weight Data ===")
        success = extract_weight_data(tvn_path, output_path)
        if success:
            weight_info['extracted_file'] = output_path
        else:
            weight_info['extracted_file'] = None
    
    return weight_info


if __name__ == "__main__":
    # 테스트: 현재 디렉토리의 TVN 파일들로 테스트
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='TVN 파일 weight 정보 추출 도구')
    parser.add_argument('tvn_path', nargs='?', help='TVN 파일 경로')
    parser.add_argument('--extract', '-e', action='store_true', 
                       help='Weight 데이터를 별도 파일로 추출')
    parser.add_argument('--output', '-o', default='weight.bin',
                       help='출력 파일 경로 (기본값: weight.bin)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='모든 TVN 파일(layers.0-27.q.opt.tvn) 처리')
    
    args = parser.parse_args()
    
    if args.all:
        # 모든 TVN 파일 처리
        print("Processing all TVN files (layers.0.q.opt.tvn ~ layers.27.q.opt.tvn)...")
        print("=" * 70)
        
        for i in range(28):  # 0 to 27
            tvn_path = f"layers.{i}.q.opt.tvn"
            output_path = f"weight_layer_{i}.bin" if args.extract else args.output
            
            print(f"\nProcessing Layer {i}: {tvn_path}")
            info = get_tvn_weight_info_with_extraction(tvn_path, args.extract, output_path)
            
            if info is None:
                print(f"Failed to process layer {i}")
        
        print(f"\n{'='*70}")
        print("Processing completed!")
        
    elif args.tvn_path:
        # 특정 TVN 파일 처리
        info = get_tvn_weight_info_with_extraction(args.tvn_path, args.extract, args.output)
        
        if info is not None:
            print(f"\n{'='*50}")
            print("Processing completed successfully!")
            if args.extract and info.get('extracted_file'):
                print(f"Weight data saved to: {info['extracted_file']}")
        else:
            print("Failed to process TVN file")
            
    else:
        # 인자 없을 경우 사용법 출력
        parser.print_help()
        print("\nExamples:")
        print("  python3 tvn_weight_offset.py layers.0.q.opt.tvn")
        print("  python3 tvn_weight_offset.py layers.0.q.opt.tvn --extract")
        print("  python3 tvn_weight_offset.py layers.0.q.opt.tvn -e -o layer0_weights.bin")
        print("  python3 tvn_weight_offset.py --all")
        print("  python3 tvn_weight_offset.py --all --extract")
