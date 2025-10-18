#!/bin/bash
# Exploratory Batch Correction Analysis - Master Execution Script
#
# This script runs both ComBat and percentile normalization, then generates
# a comprehensive comparison report.
#
# Usage: ./run_analysis.sh
# Runtime: ~5-10 minutes

set -e  # Exit on error

echo "================================================================================"
echo " ECM-ATLAS EXPLORATORY BATCH CORRECTION ANALYSIS"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Apply ComBat batch correction (R)"
echo "  2. Apply percentile normalization (Python)"
echo "  3. Generate validation metrics"
echo "  4. Create comparison report"
echo ""
echo "Estimated runtime: 5-10 minutes"
echo ""
read -p "Press Enter to continue..."
echo ""

# Check dependencies
echo "Checking dependencies..."

# Check R
if ! command -v Rscript &> /dev/null; then
    echo "ERROR: R not found. Please install R first."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3 first."
    exit 1
fi

echo "✓ R and Python found"
echo ""

# ==============================================================================
# STEP 1: ComBat Correction
# ==============================================================================

echo "================================================================================"
echo " STEP 1/2: ComBat Batch Correction"
echo "================================================================================"
echo ""

cd combat_correction/

if [ ! -f "combat_corrected.csv" ]; then
    echo "Running ComBat correction (this may take several minutes)..."
    Rscript 01_apply_combat.R

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ ComBat correction completed successfully"
    else
        echo ""
        echo "✗ ComBat correction failed"
        exit 1
    fi
else
    echo "✓ ComBat correction already completed (combat_corrected.csv exists)"
    echo "  Delete combat_corrected.csv to re-run"
fi

cd ..
echo ""

# ==============================================================================
# STEP 2: Percentile Normalization
# ==============================================================================

echo "================================================================================"
echo " STEP 2/2: Percentile Normalization"
echo "================================================================================"
echo ""

cd percentile_normalization/

if [ ! -f "percentile_normalized.csv" ]; then
    echo "Running percentile normalization..."
    python3 01_apply_percentile.py

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Percentile normalization completed successfully"
    else
        echo ""
        echo "✗ Percentile normalization failed"
        exit 1
    fi
else
    echo "✓ Percentile normalization already completed (percentile_normalized.csv exists)"
    echo "  Delete percentile_normalized.csv to re-run"
fi

cd ..
echo ""

# ==============================================================================
# STEP 3: Summary
# ==============================================================================

echo "================================================================================"
echo " ANALYSIS COMPLETE"
echo "================================================================================"
echo ""

echo "Generated Files:"
echo ""
echo "ComBat Correction:"
echo "  - combat_correction/combat_corrected.csv"
echo "  - combat_correction/combat_metadata.json"
echo "  - diagnostics/combat_before_after_pca.png"
echo "  - diagnostics/combat_effect_preservation.png"
echo ""
echo "Percentile Normalization:"
echo "  - percentile_normalization/percentile_normalized.csv"
echo "  - percentile_normalization/percentile_effects.csv"
echo "  - percentile_normalization/percentile_metadata.json"
echo "  - diagnostics/percentile_volcano_plot.png"
echo "  - diagnostics/percentile_top20_proteins.png"
echo "  - diagnostics/percentile_distribution_comparison.png"
echo ""

echo "Next Steps:"
echo ""
echo "1. Review diagnostic plots in diagnostics/ folder"
echo ""
echo "2. Check ComBat validation:"
echo "   - Review combat_metadata.json for ICC improvement"
echo "   - View PCA plots to see if studies now overlap"
echo ""
echo "3. Check Percentile validation:"
echo "   - Review percentile_metadata.json for driver recovery"
echo "   - View volcano plot for significant proteins"
echo ""
echo "4. Compare results:"
echo "   - ComBat ICC improvement vs Percentile driver recovery"
echo "   - Which method better preserves biological signal?"
echo ""
echo "5. Re-test hypotheses:"
echo "   - Run validation scripts to check if key findings hold"
echo "   - Update confidence levels based on results"
echo ""

echo "================================================================================"
echo " For detailed analysis, see: 00_README.md"
echo "================================================================================"
echo ""
