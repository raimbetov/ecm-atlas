# Duplicate Analysis Quick Reference

## TL;DR

**Found:** 29 duplicate files  
**Can safely remove:** 17 files (319 KB)  
**Need manual review:** 12 files (195 KB)  

## Immediate Action

```bash
cd /home/raimbetov/GitHub/ecm-atlas
./CLEANUP_DUPLICATES.sh
```

This removes 17 exact duplicates and creates a backup automatically.

## Summary Table

| Location | Duplicates | Exact Matches | Manual Review | Space (KB) |
|----------|------------|---------------|---------------|------------|
| scripts/ | 21 | 15 | 6 | 411.24 |
| root | 8 | 2 | 6 | 102.58 |
| **Total** | **29** | **17** | **12** | **513.82** |

## Safe to Remove (Exact MD5 Matches)

### From scripts/ (15 files):
- agent_01_universal_markers_hunter.py
- agent_02_tissue_specific_analysis.py
- agent_03_compartment_crosstalk.py
- agent_05_matrisome_category_analysis_v2.py
- agent_11_basement_membrane_analysis.py
- agent_11_cross_species_analysis.py
- agent_13_fibrinogen_coagulation_cascade.py
- agent_18_mmp_timp_protease_balance_analyzer.py
- agent_20_biomarker_panel_architect.py
- ml_agent_11_fixed.py
- ml_agent_11_random_forest_importance.py
- ml_agent_12_dimensionality_reduction.py
- ml_agent_13_network_topology.py
- ml_agent_14_biological_integration.py
- ml_agent_15_deep_learning_predictor.py

### From root (2 files):
- agent_07_methodology_harmonizer.py
- agent_10_weak_signal_amplifier.py

## Need Manual Review (Different Content)

8 Python scripts with version differences:
- agent_10_summary_figure.py
- agent_10_visualizations.py
- agent_12_versican_inflammatory_scaffold.py
- agent_14_frzb_wnt_analysis.py
- agent_15_timp3_therapeutic_evaluator.py
- agent_16_tgfb_pathway_analysis.py
- agent_17_collagen_crosslinking_entropy.py
- agent_19_species_conservation_analyzer.py

4 README.md files (different contexts)

**To compare:**
```bash
diff 13_meta_insights/agent_XX/script.py scripts/script.py
# or
meld 13_meta_insights/agent_XX/script.py scripts/script.py
```

## Files Generated

| File | Purpose |
|------|---------|
| `duplicate_analysis.py` | Python script that performed the analysis |
| `duplicate_analysis_report.csv` | Complete list with checksums |
| `DUPLICATE_ANALYSIS_COMPREHENSIVE_REPORT.md` | Full detailed report |
| `DUPLICATE_ANALYSIS_SUMMARY.txt` | Text-based summary |
| `DUPLICATE_ANALYSIS_QUICK_REFERENCE.md` | This file |
| `CLEANUP_DUPLICATES.sh` | Automated cleanup script |

## Canonical Locations

After cleanup, files will exist only in:
- **Analysis scripts:** `13_meta_insights/agent_XX/`
- **ML scripts:** `13_meta_insights/ml_agents/`

## Impact

- **Before:** 465 total files
- **After (17 removed):** 446 files (4.1% reduction)
- **Space freed:** 319.33 KB immediately, up to 513.82 KB after manual review

## Safety

- Backup automatically created by cleanup script
- Only exact MD5 matches are removed automatically
- Different content requires manual decision

---

**Last updated:** 2025-10-20  
**Full report:** [DUPLICATE_ANALYSIS_COMPREHENSIVE_REPORT.md](/home/raimbetov/GitHub/ecm-atlas/DUPLICATE_ANALYSIS_COMPREHENSIVE_REPORT.md)
