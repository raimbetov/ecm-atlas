# Duplicate Analysis Report: Meta Insights vs Scripts/Root

**Generated:** 2025-10-20
**Analysis:** Comparison of files between meta insights folders (13_meta_insights, 13_1_meta_insights) and scripts/ + root directories

---

## Executive Summary

A comprehensive duplicate file analysis was conducted across four locations in the ECM-Atlas repository:
- **Source folders:** `13_meta_insights/` and `13_1_meta_insights/`
- **Comparison folders:** `scripts/` and root folder `/`

**Key Findings:**
- 29 duplicate files identified between meta folders and scripts/root
- 513.82 KB of disk space can be reclaimed by removing duplicates
- 17 files are exact content matches (verified by MD5 checksum)
- 12 files share filenames but have different content (requires manual review)
- 378 files in meta folders are unique and have no duplicates

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total files in 13_meta_insights | 332 |
| Total files in 13_1_meta_insights | 75 |
| Total files in scripts/ | 33 |
| Total files in root | 25 |
| **Total duplicates found** | **29 files** |
| Exact content matches (MD5) | 17 |
| Filename matches (different content) | 12 |
| **Disk space to save** | **513.82 KB** |
| Unique files (keep in meta) | 378 |

---

## Duplicates to Remove (from scripts/ and root)

### Exact Content Matches (Safe to Remove - 17 files)

These files have identical MD5 checksums and can be safely removed from scripts/root:

| # | Filename | Size (bytes) | Meta Location | Duplicate Location | Checksum |
|---|----------|-------------|---------------|-------------------|----------|
| 1 | agent_02_tissue_specific_analysis.py | 25,332 | 13_meta_insights/agent_02_tissue_specific/ | scripts/ | 8495e58d... |
| 2 | agent_01_universal_markers_hunter.py | 32,211 | 13_meta_insights/agent_01_universal_markers/ | scripts/ | 3f97ae05... |
| 3 | agent_07_methodology_harmonizer.py | 36,149 | 13_meta_insights/agent_07_methodology/ | root | 54336770... |
| 4 | agent_10_weak_signal_amplifier.py | 8,112 | 13_meta_insights/agent_10_weak_signals/ | root | 4e3c8bee... |
| 5 | agent_18_mmp_timp_protease_balance_analyzer.py | 23,395 | 13_meta_insights/agent_18_protease/ | scripts/ | 08f4b4ca... |
| 6 | agent_03_compartment_crosstalk.py | 25,099 | 13_meta_insights/agent_03_nonlinear/ | scripts/ | cb53b575... |
| 7 | ml_agent_14_biological_integration.py | 9,692 | 13_meta_insights/ml_agents/ | scripts/ | 774b3430... |
| 8 | ml_agent_11_fixed.py | 3,228 | 13_meta_insights/ml_agents/ | scripts/ | 92880bd6... |
| 9 | ml_agent_15_deep_learning_predictor.py | 8,933 | 13_meta_insights/ml_agents/ | scripts/ | 834f88c9... |
| 10 | ml_agent_13_network_topology.py | 8,568 | 13_meta_insights/ml_agents/ | scripts/ | dd2d9527... |
| 11 | ml_agent_11_random_forest_importance.py | 6,817 | 13_meta_insights/ml_agents/ | scripts/ | a5c89676... |
| 12 | ml_agent_12_dimensionality_reduction.py | 8,364 | 13_meta_insights/ml_agents/ | scripts/ | b47d9890... |
| 13 | agent_20_biomarker_panel_architect.py | 34,665 | 13_meta_insights/agent_20_biomarkers/ | scripts/ | dfdf9add... |
| 14 | agent_05_matrisome_category_analysis_v2.py | 32,080 | 13_meta_insights/agent_05_matrisome/ | scripts/ | e1cb0a24... |
| 15 | agent_13_fibrinogen_coagulation_cascade.py | 36,124 | 13_meta_insights/agent_13_coagulation/ | scripts/ | 80c0e75f... |
| 16 | agent_11_basement_membrane_analysis.py | 14,801 | 13_meta_insights/agent_11_cross_species/ | scripts/ | 4a53e91d... |
| 17 | agent_11_cross_species_analysis.py | 13,425 | 13_meta_insights/agent_11_cross_species/ | scripts/ | e9678102... |

**Subtotal: 326,995 bytes (319.33 KB)**

---

### Filename Matches with Different Content (Requires Manual Review - 12 files)

These files share the same name but have different content. Manual review required to determine which version to keep:

| # | Filename | Size (bytes) | Meta Location | Duplicate Location | Checksum Comparison |
|---|----------|-------------|---------------|-------------------|---------------------|
| 1 | README.md | 8,883 | 13_meta_insights/ | root | meta:1e07b106 vs root:8f4b81ee |
| 2 | agent_15_timp3_therapeutic_evaluator.py | 22,833 | 13_meta_insights/agent_15_timp3/ | scripts/ | meta:5650a5ae vs scripts:314da37b |
| 3 | README.md | 18,986 | 13_meta_insights/synthesis/ | root | meta:7cfe7723 vs root:8f4b81ee |
| 4 | agent_19_species_conservation_analyzer.py | 22,686 | 13_meta_insights/agent_19_conservation/ | scripts/ | meta:bfc27718 vs scripts:aa85dc6c |
| 5 | agent_17_collagen_crosslinking_entropy.py | 20,453 | 13_meta_insights/agent_17_crosslinking/ | scripts/ | meta:6a8854c1 vs scripts:d831d4a9 |
| 6 | agent_10_summary_figure.py | 7,251 | 13_meta_insights/agent_10_weak_signals/ | root | meta:5b939f3a vs root:1b1ea8d1 |
| 7 | agent_10_visualizations.py | 11,405 | 13_meta_insights/agent_10_weak_signals/ | root | meta:65b48872 vs root:df372aed |
| 8 | agent_14_frzb_wnt_analysis.py | 16,970 | 13_meta_insights/agent_14_wnt_pathway/ | scripts/ | meta:76dcbe3c vs scripts:5e020741 |
| 9 | agent_12_versican_inflammatory_scaffold.py | 20,484 | 13_meta_insights/agent_12_versican/ | scripts/ | meta:73791423 vs scripts:31812966 |
| 10 | agent_16_tgfb_pathway_analysis.py | 34,947 | 13_meta_insights/agent_16_tgfb/ | scripts/ | meta:808627ec vs scripts:abd999f7 |
| 11 | README.md | 5,428 | 13_meta_insights/age_related_proteins/hypothesis_05_giants/ | root | meta:ec1ffdfa vs root:8f4b81ee |
| 12 | README.md | 8,833 | 13_meta_insights/age_related_proteins/hypothesis_01_inverse_paradox/ | root | meta:5fa105fb vs root:8f4b81ee |

**Subtotal: 199,159 bytes (194.49 KB)**

**Note:** The 4 README.md files all have different content in meta folders but match the same root README.md. This suggests these are different documents that happen to share a common name.

---

## Unique Files in Meta Folders (Keep in Both Locations)

**Total: 378 files** are unique to meta folders and have no duplicates in scripts/ or root.

### Breakdown by File Type

| File Extension | Count | Description |
|----------------|-------|-------------|
| .csv | 130 | Data files containing analysis results |
| .png | 126 | Visualization images |
| .md | 87 | Documentation files |
| .py | 27 | Python scripts unique to meta analysis |
| .txt | 4 | Text documents |
| .json | 3 | JSON data files |
| .sh | 1 | Shell scripts |

These files represent the unique analytical outputs and insights generated specifically within the meta folders and should be preserved.

---

## Impact Analysis

### Disk Space Recovery

| Category | Files | Size (KB) | Percentage |
|----------|-------|-----------|------------|
| Exact matches (safe to delete) | 17 | 319.33 | 62.1% |
| Different content (manual review) | 12 | 194.49 | 37.9% |
| **Total potential savings** | **29** | **513.82** | **100%** |

### File Distribution

```
Repository Structure:
├── 13_meta_insights/ (332 files)
├── 13_1_meta_insights/ (75 files)
├── scripts/ (33 files) ← 25 duplicates here
└── root (25 files) ← 4 duplicates here
```

---

## Recommendations

### Immediate Actions (Safe)

1. **Remove 17 exact duplicate files** from scripts/ and root:
   - All ML agent scripts (6 files) can be safely removed from scripts/
   - Agent analysis scripts with matching checksums (11 files) can be removed

2. **Commands to remove exact duplicates:**

```bash
cd /home/raimbetov/GitHub/ecm-atlas

# Remove exact duplicates from scripts/
rm scripts/agent_02_tissue_specific_analysis.py
rm scripts/agent_01_universal_markers_hunter.py
rm scripts/agent_18_mmp_timp_protease_balance_analyzer.py
rm scripts/agent_03_compartment_crosstalk.py
rm scripts/ml_agent_14_biological_integration.py
rm scripts/ml_agent_11_fixed.py
rm scripts/ml_agent_15_deep_learning_predictor.py
rm scripts/ml_agent_13_network_topology.py
rm scripts/ml_agent_11_random_forest_importance.py
rm scripts/ml_agent_12_dimensionality_reduction.py
rm scripts/agent_20_biomarker_panel_architect.py
rm scripts/agent_05_matrisome_category_analysis_v2.py
rm scripts/agent_13_fibrinogen_coagulation_cascade.py
rm scripts/agent_11_basement_membrane_analysis.py
rm scripts/agent_11_cross_species_analysis.py

# Remove exact duplicates from root
rm agent_07_methodology_harmonizer.py
rm agent_10_weak_signal_amplifier.py
```

**Disk space saved: 319.33 KB**

### Manual Review Required

1. **Compare 12 files with different content:**
   - 4 README.md files (likely different contexts, may want to rename)
   - 8 Python scripts with divergent versions (need diff analysis)

2. **For each file:**
   ```bash
   # Compare versions
   diff 13_meta_insights/path/to/file scripts/file

   # Or use a visual diff tool
   meld 13_meta_insights/path/to/file scripts/file
   ```

3. **Decision criteria:**
   - Which version is more recent?
   - Which version has better documentation?
   - Are changes significant or just formatting?
   - Can versions be merged?

### Long-term Organization

1. **Establish canonical locations:**
   - Analysis scripts → `13_meta_insights/agent_XX/`
   - Utility scripts → `scripts/`
   - Documentation → Root README only, others renamed

2. **Create symlinks** if scripts need to be accessible from multiple locations:
   ```bash
   ln -s ../13_meta_insights/agent_XX/script.py scripts/script.py
   ```

3. **Update documentation** to reflect single source of truth for each script

---

## Detailed Duplicate List

See complete list with checksums in: `duplicate_analysis_report.csv`

---

## Validation Steps

Before removing files, verify:

1. No active references in other scripts
2. No import dependencies
3. Files are not part of active development
4. Backup created (if needed)

```bash
# Check for imports/references
grep -r "agent_01_universal_markers_hunter" .
grep -r "ml_agent_14" .

# Create backup before deletion
tar -czf duplicates_backup_$(date +%Y%m%d).tar.gz scripts/ agent_*.py
```

---

## Appendix: Analysis Methodology

1. **File Discovery:** Used `find` to enumerate all files in each location
2. **Checksum Calculation:** MD5 hash computed for each file
3. **Comparison Logic:**
   - First: Match by MD5 checksum (exact content match)
   - Second: Match by filename (potential different versions)
4. **Size Calculation:** Summed sizes of duplicate files for impact analysis

**Script used:** `/home/raimbetov/GitHub/ecm-atlas/duplicate_analysis.py`

---

**Report End**
