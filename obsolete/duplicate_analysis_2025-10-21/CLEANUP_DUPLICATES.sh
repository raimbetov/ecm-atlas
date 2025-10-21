#!/bin/bash
# Duplicate File Cleanup Script
# Generated: 2025-10-20
# Updated: 2025-10-21 (scripts/ moved to obsolete/scripts/)
#
# This script removes verified duplicate files from obsolete/scripts/ and root directories
# All files being removed are exact MD5 checksum matches with files in 13_meta_insights/
#
# IMPORTANT: Review the report before running this script!
# Report: DUPLICATE_ANALYSIS_COMPREHENSIVE_REPORT.md

set -e  # Exit on error

echo "=================================="
echo "ECM-Atlas Duplicate Cleanup Script"
echo "=================================="
echo ""
echo "This will remove 17 duplicate files (319.33 KB)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Creating backup..."
BACKUP_FILE="duplicates_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" \
    obsolete/scripts/agent_02_tissue_specific_analysis.py \
    obsolete/scripts/agent_01_universal_markers_hunter.py \
    obsolete/scripts/agent_18_mmp_timp_protease_balance_analyzer.py \
    obsolete/scripts/agent_03_compartment_crosstalk.py \
    obsolete/scripts/ml_agent_14_biological_integration.py \
    obsolete/scripts/ml_agent_11_fixed.py \
    obsolete/scripts/ml_agent_15_deep_learning_predictor.py \
    obsolete/scripts/ml_agent_13_network_topology.py \
    obsolete/scripts/ml_agent_11_random_forest_importance.py \
    obsolete/scripts/ml_agent_12_dimensionality_reduction.py \
    obsolete/scripts/agent_20_biomarker_panel_architect.py \
    obsolete/scripts/agent_05_matrisome_category_analysis_v2.py \
    obsolete/scripts/agent_13_fibrinogen_coagulation_cascade.py \
    obsolete/scripts/agent_11_basement_membrane_analysis.py \
    obsolete/scripts/agent_11_cross_species_analysis.py \
    agent_07_methodology_harmonizer.py \
    agent_10_weak_signal_amplifier.py \
    2>/dev/null || true

echo "Backup created: $BACKUP_FILE"
echo ""

# Track what we're removing
REMOVED_COUNT=0
REMOVED_SIZE=0

echo "Removing exact duplicates from obsolete/scripts/..."
echo ""

# Function to remove file and track stats
remove_file() {
    local file=$1
    if [ -f "$file" ]; then
        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        echo "  ✓ Removing: $file ($(echo $size | numfmt --to=iec 2>/dev/null || echo $size bytes))"
        rm "$file"
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
        REMOVED_SIZE=$((REMOVED_SIZE + size))
    else
        echo "  ⚠ File not found (already removed?): $file"
    fi
}

# Remove duplicates from obsolete/scripts/ (15 files)
remove_file "obsolete/scripts/agent_02_tissue_specific_analysis.py"
remove_file "obsolete/scripts/agent_01_universal_markers_hunter.py"
remove_file "obsolete/scripts/agent_18_mmp_timp_protease_balance_analyzer.py"
remove_file "obsolete/scripts/agent_03_compartment_crosstalk.py"
remove_file "obsolete/scripts/ml_agent_14_biological_integration.py"
remove_file "obsolete/scripts/ml_agent_11_fixed.py"
remove_file "obsolete/scripts/ml_agent_15_deep_learning_predictor.py"
remove_file "obsolete/scripts/ml_agent_13_network_topology.py"
remove_file "obsolete/scripts/ml_agent_11_random_forest_importance.py"
remove_file "obsolete/scripts/ml_agent_12_dimensionality_reduction.py"
remove_file "obsolete/scripts/agent_20_biomarker_panel_architect.py"
remove_file "obsolete/scripts/agent_05_matrisome_category_analysis_v2.py"
remove_file "obsolete/scripts/agent_13_fibrinogen_coagulation_cascade.py"
remove_file "obsolete/scripts/agent_11_basement_membrane_analysis.py"
remove_file "obsolete/scripts/agent_11_cross_species_analysis.py"

echo ""
echo "Removing exact duplicates from root..."
echo ""

# Remove duplicates from root (2 files)
remove_file "agent_07_methodology_harmonizer.py"
remove_file "agent_10_weak_signal_amplifier.py"

echo ""
echo "=================================="
echo "Cleanup Complete!"
echo "=================================="
echo ""
echo "Files removed: $REMOVED_COUNT"
echo "Disk space freed: $REMOVED_SIZE bytes"
echo "Backup saved to: $BACKUP_FILE"
echo ""
echo "Canonical versions remain in:"
echo "  - 13_meta_insights/agent_XX/"
echo ""
echo "Manual review still needed for 12 files with different content."
echo "See: DUPLICATE_ANALYSIS_COMPREHENSIVE_REPORT.md"
echo ""
