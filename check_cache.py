#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

def check_session_cache(sessions_path, president_cache_path):
    """
    Check which Session folders have cached npy files in President folder
    
    Args:
        sessions_path: Session folder path
        president_cache_path: President cache folder path
    """
    # Get all Session folders
    session_folders = [f for f in os.listdir(sessions_path) 
                      if os.path.isdir(os.path.join(sessions_path, f))]
    
    print(f"Found {len(session_folders)} session folders")
    print("=" * 60)
    
    # Statistics
    total_sessions = len(session_folders)
    sessions_with_cache = 0
    sessions_without_cache = 0
    
    # Check each Session folder
    for session_folder in sorted(session_folders):
        session_id = session_folder
        
        # Look for corresponding npy files in President cache path
        cache_pattern = os.path.join(president_cache_path, f"*{session_id}*")
        cache_files = glob.glob(cache_pattern)
        
        # Find all npy files
        npy_files = [f for f in cache_files if f.endswith('.npy')]
        
        if npy_files:
            sessions_with_cache += 1
            status = "HAS CACHE"
            file_count = len(npy_files)
            file_examples = ", ".join([os.path.basename(f) for f in npy_files[:2]])  # Show first 2 files
            if len(npy_files) > 2:
                file_examples += f" ... (+{len(npy_files)-2} more)"
        else:
            sessions_without_cache += 1
            status = "NO CACHE"
            file_count = 0
            file_examples = ""
        if file_count < 20:
            print(f"Session {session_id:>4}: {status:12} | Files: {file_count:2} | {file_examples}")
    
    print("=" * 60)
    print(f"Summary:")
    print(f"Total sessions: {total_sessions}")
    print(f"Sessions with cache: {sessions_with_cache} ({sessions_with_cache/total_sessions*100:.1f}%)")
    print(f"Sessions without cache: {sessions_without_cache} ({sessions_without_cache/total_sessions*100:.1f}%)")
    
    return {
        'total_sessions': total_sessions,
        'sessions_with_cache': sessions_with_cache,
        'sessions_without_cache': sessions_without_cache,
        'cache_coverage': sessions_with_cache / total_sessions * 100
    }

def find_cache_files_by_pattern(president_cache_path, patterns):
    """
    Find cache files by specific patterns
    
    Args:
        president_cache_path: President cache folder path
        patterns: List of file patterns to search for
    """
    print(f"\nSearching for specific patterns in {president_cache_path}")
    print("=" * 60)
    
    all_cache_files = []
    
    for pattern in patterns:
        search_pattern = os.path.join(president_cache_path, pattern)
        files = glob.glob(search_pattern)
        all_cache_files.extend(files)
        
        # print(f"Pattern: {pattern}")
        # print(f"Found {len(files)} files")
        for file in files[:5]:  # Show first 5 files
            print(f"  - {os.path.basename(file)}")
        if len(files) > 5:
            print(f"  ... and {len(files)-5} more files")
        print()
    
    return all_cache_files

def get_session_ids_from_cache(president_cache_path):
    """
    Extract all session IDs from cache files
    """
    print(f"Extracting session IDs from cache files in {president_cache_path}")
    print("=" * 60)
    
    # Get all npy files
    npy_files = glob.glob(os.path.join(president_cache_path, "*.npy"))
    
    session_ids = set()
    
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        
        # Try to extract session ID from filename
        if 'session_' in filename:
            # Format: session_2_input0.npy
            parts = filename.split('_')
            if len(parts) >= 2:
                session_id = parts[1]  # Get number part
                session_ids.add(session_id)
        else:
            # Try to extract pure numbers as session ID
            import re
            numbers = re.findall(r'\d+', filename)
            if numbers:
                # Take the first longer number as session ID
                for num in numbers:
                    if len(num) >= 1:  # At least 1 digit
                        session_ids.add(num)
                        break
    
    # print(f"Found {len(session_ids)} unique session IDs in cache:")
    # for session_id in sorted(session_ids, key=int):
        # print(f"  Session {session_id}")
    
    return session_ids

def compare_sessions_vs_cache(sessions_path, president_cache_path):
    """
    Compare correspondence between Session folders and cache files
    """
    # Get all session IDs from Session folders
    session_folders = [f for f in os.listdir(sessions_path) 
                      if os.path.isdir(os.path.join(sessions_path, f))]
    session_ids_from_folders = set(session_folders)
    
    # Get all session IDs from cache files
    session_ids_from_cache = get_session_ids_from_cache(president_cache_path)
    
    print("\n" + "=" * 60)
    print("Session vs Cache Comparison")
    print("=" * 60)
    
    # In Session but not in cache
    only_in_sessions = session_ids_from_folders - session_ids_from_cache
    print(f"Sessions ONLY in folders (not in cache): {len(only_in_sessions)}")
    for session_id in sorted(only_in_sessions, key=int):
        print(f"  Session {session_id}")
    
    # In cache but not in Session
    only_in_cache = session_ids_from_cache - session_ids_from_folders
    # print(f"\nSessions ONLY in cache (not in folders): {len(only_in_cache)}")
    # for session_id in sorted(only_in_cache, key=int):
        # print(f"  Session {session_id}")
    
    # In both
    in_both = session_ids_from_folders & session_ids_from_cache
    # print(f"\nSessions in BOTH folders and cache: {len(in_both)}")
    # for session_id in sorted(in_both, key=int)[:10]:  # Only show first 10
        # print(f"  Session {session_id}")
    if len(in_both) > 10:
        print(f"  ... and {len(in_both)-10} more")
    
    return {
        'only_in_sessions': only_in_sessions,
        'only_in_cache': only_in_cache,
        'in_both': in_both
    }

def main():
    # Configure paths - please modify these paths according to your actual situation
    SESSIONS_PATH = "/dataset/MAHNOB-HCI/Sessions"  # Session文件夹路径
    PRESIDENT_CACHE_PATH = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"  # President缓存文件夹路径
    
    # Check if paths exist
    if not os.path.exists(SESSIONS_PATH):
        print(f"Error: Sessions path does not exist: {SESSIONS_PATH}")
        return
    
    if not os.path.exists(PRESIDENT_CACHE_PATH):
        print(f"Error: President cache path does not exist: {PRESIDENT_CACHE_PATH}")
        return
    
    print("MAHNOB-HCI Session Cache Checker")
    print("=" * 60)
    
    # 1. Basic check
    stats = check_session_cache(SESSIONS_PATH, PRESIDENT_CACHE_PATH)
    
    # 2. Detailed comparison
    comparison = compare_sessions_vs_cache(SESSIONS_PATH, PRESIDENT_CACHE_PATH)
    
    # 3. Find files by specific patterns
    patterns = [
        "session_*_input*.npy",  # Input files
        "session_*_label*.npy",  # Label files
    ]
    find_cache_files_by_pattern(PRESIDENT_CACHE_PATH, patterns)
    
    # Generate report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Cache Coverage: {stats['cache_coverage']:.1f}%")
    print(f"Missing cache for {len(comparison['only_in_sessions'])} sessions")
    
    # Recommendation for next steps
    if stats['cache_coverage'] < 100:
        print(f"\nRecommendation: Need to preprocess {len(comparison['only_in_sessions'])} sessions")
        missing_sessions = sorted(comparison['only_in_sessions'], key=int)
        print("Missing sessions:", ", ".join(missing_sessions))
    else:
        print("\nAll sessions have cache files!")

if __name__ == "__main__":
    main()