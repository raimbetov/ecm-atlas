#!/usr/bin/env python3
"""
Audit all paths in processing scripts to ensure they work correctly.
"""

import os
from pathlib import Path
import re

def audit_script(filepath):
    """Audit a single script for path issues."""
    issues = []

    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    # Check for hardcoded absolute paths
    hardcoded_paths = re.findall(r'["\']/(home|Users|data|mnt)/[^"\']+["\']', content)
    if hardcoded_paths:
        issues.append(f"  ⚠️  Hardcoded absolute paths found: {len(hardcoded_paths)}")
        for path in hardcoded_paths[:3]:  # Show first 3
            issues.append(f"      - {path}")

    # Check for data_raw references
    data_raw_refs = re.findall(r'["\']data_raw/[^"\']+["\']', content)
    if data_raw_refs:
        issues.append(f"  ✓ data_raw references: {len(data_raw_refs)}")

    # Check for references directory access
    ref_access = re.findall(r'references/[^"\']+', content)
    if ref_access:
        issues.append(f"  ✓ references/ directory access: {len(ref_access)}")

    # Check for project root detection
    if 'project_root' in content.lower():
        issues.append(f"  ✓ Uses project_root variable")

    # Check for Path object usage
    if 'from pathlib import Path' in content or 'import pathlib' in content:
        issues.append(f"  ✓ Uses pathlib.Path")

    return issues

def main():
    project_root = Path("/home/raimbetov/GitHub/ecm-atlas")

    scripts_to_audit = [
        "11_subagent_for_LFQ_ingestion/autonomous_agent.py",
        "11_subagent_for_LFQ_ingestion/merge_to_unified.py",
        "11_subagent_for_LFQ_ingestion/universal_zscore_function.py",
        "11_subagent_for_LFQ_ingestion/study_config_template.py",
    ]

    print("=" * 80)
    print("PATH AUDIT REPORT")
    print("=" * 80)
    print(f"\nProject root: {project_root}")
    print(f"Exists: {project_root.exists()}")
    print()

    # Check key directories
    print("Key Directories:")
    key_dirs = [
        "data_raw",
        "references",
        "08_merged_ecm_dataset",
        "11_subagent_for_LFQ_ingestion"
    ]

    for dirname in key_dirs:
        dirpath = project_root / dirname
        status = "✓" if dirpath.exists() else "✗"
        print(f"  {status} {dirname}/")

    print()

    # Check matrisome references
    print("Matrisome References:")
    matrisome_files = [
        "references/mouse_matrisome_v2.csv",
        "references/human_matrisome_v2.csv"
    ]

    for filepath in matrisome_files:
        fullpath = project_root / filepath
        status = "✓" if fullpath.exists() else "✗"
        size = f"({fullpath.stat().st_size / 1024:.1f} KB)" if fullpath.exists() else ""
        print(f"  {status} {filepath} {size}")

    print()

    # Check Schuler data
    print("Schuler 2021 Data:")
    schuler_dir = project_root / "data_raw" / "Schuler et al. - 2021"
    if schuler_dir.exists():
        print(f"  ✓ {schuler_dir.name}/")
        for f in sorted(schuler_dir.glob("*.xls*")):
            print(f"      - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"  ✗ Directory not found: {schuler_dir}")

    print()
    print("-" * 80)
    print("SCRIPT AUDITS:")
    print("-" * 80)

    for script_path in scripts_to_audit:
        fullpath = project_root / script_path
        print(f"\n📄 {script_path}")

        if not fullpath.exists():
            print(f"  ✗ File not found!")
            continue

        issues = audit_script(fullpath)
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("  ✓ No issues found")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    recommendations = [
        "1. All scripts should use project root auto-detection",
        "2. Avoid hardcoded absolute paths",
        "3. Use pathlib.Path for path operations",
        "4. Use relative paths from project root",
        "5. Verify matrisome reference files exist before processing"
    ]

    for rec in recommendations:
        print(f"  {rec}")

    print()

if __name__ == '__main__':
    main()
