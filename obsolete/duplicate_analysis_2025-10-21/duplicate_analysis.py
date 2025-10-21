#!/usr/bin/env python3
"""
Comprehensive duplicate file analysis across meta insights folders and scripts/root
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import csv

def get_md5(filepath):
    """Calculate MD5 checksum of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_file_size(filepath):
    """Get file size in bytes"""
    try:
        return os.path.getsize(filepath)
    except:
        return 0

def scan_directory(base_path, max_depth=None):
    """Scan directory and return dict of files with their checksums and metadata"""
    files = {}
    base = Path(base_path)

    for filepath in base.rglob("*"):
        if filepath.is_file():
            rel_path = filepath.relative_to(base)
            checksum = get_md5(filepath)
            size = get_file_size(filepath)

            files[str(filepath)] = {
                'relative': str(rel_path),
                'filename': filepath.name,
                'checksum': checksum,
                'size': size,
                'path': str(filepath)
            }

    return files

def scan_root_only(base_path):
    """Scan only root level files (maxdepth=1)"""
    files = {}
    base = Path(base_path)

    for filepath in base.glob("*"):
        if filepath.is_file():
            checksum = get_md5(filepath)
            size = get_file_size(filepath)

            files[str(filepath)] = {
                'relative': filepath.name,
                'filename': filepath.name,
                'checksum': checksum,
                'size': size,
                'path': str(filepath)
            }

    return files

def main():
    base_dir = "/home/raimbetov/GitHub/ecm-atlas"

    print("Scanning directories...")
    print("=" * 80)

    # Scan all four locations
    meta13 = scan_directory(f"{base_dir}/13_meta_insights")
    meta13_1 = scan_directory(f"{base_dir}/13_1_meta_insights")
    scripts = scan_directory(f"{base_dir}/scripts")
    root = scan_root_only(base_dir)

    print(f"Files in 13_meta_insights: {len(meta13)}")
    print(f"Files in 13_1_meta_insights: {len(meta13_1)}")
    print(f"Files in scripts/: {len(scripts)}")
    print(f"Files in root: {len(root)}")
    print()

    # Build checksum and filename indexes for scripts and root
    scripts_by_checksum = defaultdict(list)
    scripts_by_filename = defaultdict(list)
    root_by_checksum = defaultdict(list)
    root_by_filename = defaultdict(list)

    for path, info in scripts.items():
        if info['checksum']:
            scripts_by_checksum[info['checksum']].append(info)
        scripts_by_filename[info['filename']].append(info)

    for path, info in root.items():
        if info['checksum']:
            root_by_checksum[info['checksum']].append(info)
        root_by_filename[info['filename']].append(info)

    # Find duplicates
    duplicates = []
    unique_meta_files = []

    # Check meta13 files
    for path, info in meta13.items():
        checksum = info['checksum']
        filename = info['filename']

        duplicate_found = False

        # Check for checksum match in scripts
        if checksum and checksum in scripts_by_checksum:
            for dup in scripts_by_checksum[checksum]:
                duplicates.append({
                    'meta_location': '13_meta_insights',
                    'meta_path': info['relative'],
                    'duplicate_location': 'scripts/',
                    'duplicate_path': dup['relative'],
                    'filename': filename,
                    'size': info['size'],
                    'match_type': 'MD5 checksum',
                    'checksum': checksum
                })
                duplicate_found = True

        # Check for checksum match in root
        if checksum and checksum in root_by_checksum:
            for dup in root_by_checksum[checksum]:
                duplicates.append({
                    'meta_location': '13_meta_insights',
                    'meta_path': info['relative'],
                    'duplicate_location': 'root',
                    'duplicate_path': dup['relative'],
                    'filename': filename,
                    'size': info['size'],
                    'match_type': 'MD5 checksum',
                    'checksum': checksum
                })
                duplicate_found = True

        # Check for filename match in scripts (no checksum match)
        if not duplicate_found and filename in scripts_by_filename:
            for dup in scripts_by_filename[filename]:
                if dup.get('checksum') != checksum:
                    duplicates.append({
                        'meta_location': '13_meta_insights',
                        'meta_path': info['relative'],
                        'duplicate_location': 'scripts/',
                        'duplicate_path': dup['relative'],
                        'filename': filename,
                        'size': info['size'],
                        'match_type': 'Filename only (different content)',
                        'checksum': f"meta:{checksum[:8]} vs scripts:{dup.get('checksum','N/A')[:8]}"
                    })
                    duplicate_found = True

        # Check for filename match in root (no checksum match)
        if not duplicate_found and filename in root_by_filename:
            for dup in root_by_filename[filename]:
                if dup.get('checksum') != checksum:
                    duplicates.append({
                        'meta_location': '13_meta_insights',
                        'meta_path': info['relative'],
                        'duplicate_location': 'root',
                        'duplicate_path': dup['relative'],
                        'filename': filename,
                        'size': info['size'],
                        'match_type': 'Filename only (different content)',
                        'checksum': f"meta:{checksum[:8]} vs root:{dup.get('checksum','N/A')[:8]}"
                    })
                    duplicate_found = True

        if not duplicate_found:
            unique_meta_files.append(info)

    # Check meta13_1 files
    for path, info in meta13_1.items():
        checksum = info['checksum']
        filename = info['filename']

        duplicate_found = False

        # Check for checksum match in scripts
        if checksum and checksum in scripts_by_checksum:
            for dup in scripts_by_checksum[checksum]:
                duplicates.append({
                    'meta_location': '13_1_meta_insights',
                    'meta_path': info['relative'],
                    'duplicate_location': 'scripts/',
                    'duplicate_path': dup['relative'],
                    'filename': filename,
                    'size': info['size'],
                    'match_type': 'MD5 checksum',
                    'checksum': checksum
                })
                duplicate_found = True

        # Check for checksum match in root
        if checksum and checksum in root_by_checksum:
            for dup in root_by_checksum[checksum]:
                duplicates.append({
                    'meta_location': '13_1_meta_insights',
                    'meta_path': info['relative'],
                    'duplicate_location': 'root',
                    'duplicate_path': dup['relative'],
                    'filename': filename,
                    'size': info['size'],
                    'match_type': 'MD5 checksum',
                    'checksum': checksum
                })
                duplicate_found = True

        # Check for filename match in scripts (no checksum match)
        if not duplicate_found and filename in scripts_by_filename:
            for dup in scripts_by_filename[filename]:
                if dup.get('checksum') != checksum:
                    duplicates.append({
                        'meta_location': '13_1_meta_insights',
                        'meta_path': info['relative'],
                        'duplicate_location': 'scripts/',
                        'duplicate_path': dup['relative'],
                        'filename': filename,
                        'size': info['size'],
                        'match_type': 'Filename only (different content)',
                        'checksum': f"meta:{checksum[:8]} vs scripts:{dup.get('checksum','N/A')[:8]}"
                    })
                    duplicate_found = True

        # Check for filename match in root (no checksum match)
        if not duplicate_found and filename in root_by_filename:
            for dup in root_by_filename[filename]:
                if dup.get('checksum') != checksum:
                    duplicates.append({
                        'meta_location': '13_1_meta_insights',
                        'meta_path': info['relative'],
                        'duplicate_location': 'root',
                        'duplicate_path': dup['relative'],
                        'filename': filename,
                        'size': info['size'],
                        'match_type': 'Filename only (different content)',
                        'checksum': f"meta:{checksum[:8]} vs root:{dup.get('checksum','N/A')[:8]}"
                    })
                    duplicate_found = True

        if not duplicate_found:
            unique_meta_files.append(info)

    # Calculate statistics
    total_duplicate_size = sum(d['size'] for d in duplicates)
    exact_checksum_matches = [d for d in duplicates if d['match_type'] == 'MD5 checksum']
    filename_only_matches = [d for d in duplicates if 'Filename only' in d['match_type']]

    # Print summary
    print("\n")
    print("=" * 80)
    print("DUPLICATE ANALYSIS REPORT")
    print("=" * 80)
    print()
    print("### Summary Statistics")
    print(f"- Total files in 13_meta_insights: {len(meta13)}")
    print(f"- Total files in 13_1_meta_insights: {len(meta13_1)}")
    print(f"- Total files in scripts/: {len(scripts)}")
    print(f"- Total files in root: {len(root)}")
    print(f"- **Duplicates found: {len(duplicates)} files**")
    print(f"  - Exact content matches (MD5): {len(exact_checksum_matches)}")
    print(f"  - Filename matches (different content): {len(filename_only_matches)}")
    print(f"- **Disk space to save: {total_duplicate_size:,} bytes ({total_duplicate_size/1024:.2f} KB)**")
    print(f"- Unique files in meta folders (no duplicates): {len(unique_meta_files)}")
    print()

    # Save duplicates to CSV
    if duplicates:
        csv_file = f"{base_dir}/duplicate_analysis_report.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['meta_location', 'meta_path', 'duplicate_location',
                                                    'duplicate_path', 'filename', 'size', 'match_type', 'checksum'])
            writer.writeheader()
            writer.writerows(duplicates)
        print(f"Detailed report saved to: {csv_file}")
        print()

    # Print sample duplicates
    print("### Sample Duplicates (first 20):")
    print()
    for i, dup in enumerate(duplicates[:20], 1):
        print(f"{i}. {dup['filename']}")
        print(f"   Meta: {dup['meta_location']}/{dup['meta_path']}")
        print(f"   Dup:  {dup['duplicate_location']}/{dup['duplicate_path']}")
        print(f"   Size: {dup['size']} bytes | Match: {dup['match_type']}")
        print()

    if len(duplicates) > 20:
        print(f"... and {len(duplicates) - 20} more duplicates")
        print()

    # Print unique files summary
    print("### Files Unique to Meta Folders (no duplicates in scripts/root):")
    print(f"Total: {len(unique_meta_files)} files")
    print()

    # Group by extension
    by_ext = defaultdict(int)
    for f in unique_meta_files:
        ext = Path(f['filename']).suffix or '(no extension)'
        by_ext[ext] += 1

    print("Breakdown by file type:")
    for ext, count in sorted(by_ext.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext}: {count} files")

    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
