# 🚀 START HERE: LFQ Dataset Processing

## What Is This?

This directory contains everything you need to process LFQ proteomics datasets and add them to the ECM-Atlas unified database.

---

## 🎯 Quick Start (Recommended)

### Use the Autonomous Agent

```bash
python autonomous_agent.py "data_raw/Author et al. - Year/"
```

That's it! The agent will:
- ✅ Find data files
- ✅ Create dedicated folder
- ✅ Log every step
- ✅ Process dataset
- ✅ Merge to unified CSV
- ✅ Calculate z-scores

**Track progress in real-time:**
```bash
tail -f XX_Author_Year_paper_to_csv/agent_log.md
```

**Full guide:** [`AUTONOMOUS_AGENT_GUIDE.md`](AUTONOMOUS_AGENT_GUIDE.md)

---

## 📚 File Structure

### For Users (Start Here)

1. **`00_START_HERE.md`** (this file) - Quick orientation
2. **`AUTONOMOUS_AGENT_GUIDE.md`** - How to use the autonomous agent
3. **`README.md`** - Complete overview with examples
4. **`IMPROVEMENTS_SUMMARY.md`** - What changed recently

### For Developers (Advanced)

5. **`01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md`** - PHASE 1 & 2 detailed algorithm
6. **`02_ZSCORE_CALCULATION_UNIVERSAL_FUNCTION.md`** - PHASE 3 algorithm
7. **`00_PIPELINE_FLOWCHART.md`** - Complete mermaid visualization

### Executable Scripts

8. **`autonomous_agent.py`** ⭐ - Main orchestrator (fully automated)
9. **`study_config_template.py`** - Configuration template
10. **`merge_to_unified.py`** - Merge script (PHASE 2)
11. **`universal_zscore_function.py`** - Z-score function (PHASE 3)

### Examples

12. **`EXAMPLE_agent_log.md`** - Sample log output
13. **`EXAMPLE_agent_state.json`** - Sample state file

---

## 🔄 The Pipeline (3 Phases)

```
PHASE 0: Reconnaissance
└─> Identify data files, generate config

PHASE 1: Data Normalization
└─> Excel → Long format → ECM filtering → Wide format

PHASE 2: Merge to Unified CSV
└─> Add study to 08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

PHASE 3: Z-Score Calculation
└─> Calculate z-scores for new study only
```

**Visual diagram:** [`00_PIPELINE_FLOWCHART.md`](00_PIPELINE_FLOWCHART.md)

---

## 🎬 Typical Workflow

### Option A: Fully Automated (⭐ Recommended)

```bash
# 1. Run agent
python autonomous_agent.py "data_raw/Randles et al. - 2021/"

# 2. Edit config (if prompted)
nano XX_Randles_2021_paper_to_csv/study_config.json

# 3. Re-run agent
python autonomous_agent.py "data_raw/Randles et al. - 2021/"

# Done! All phases completed.
```

### Option B: Manual Step-by-Step

```bash
# 1. Configure
cp study_config_template.py my_study_config.py
nano my_study_config.py

# 2. Process (PHASE 1) - follow manual instructions
# See: 01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md

# 3. Merge (PHASE 2)
python merge_to_unified.py 05_Randles_paper_to_csv/Randles_2021_wide_format.csv

# 4. Z-scores (PHASE 3)
python universal_zscore_function.py Randles_2021 Tissue
```

---

## 📦 Output

After processing, you get:

```
XX_Author_Year_paper_to_csv/
├── agent_log.md                      # ✅ Complete execution log
├── agent_state.json                  # ✅ Current state (for debugging)
├── study_config.json                 # ✅ Configuration
└── Author_Year_wide_format.csv       # ✅ Final dataset

08_merged_ecm_dataset/
├── merged_ecm_aging_zscore.csv       # ✅ UPDATED with new study
├── unified_metadata.json             # ✅ UPDATED metadata
└── backups/
    └── ECM_Atlas_Unified_*.csv       # ✅ Automatic backups
```

---

## 🆘 Need Help?

### If agent fails:

1. **Check log:** `less XX_Study/agent_log.md`
2. **Check state:** `cat XX_Study/agent_state.json`
3. **Look for ❌ ERROR markers** in log
4. **Fix issue and re-run**

### Common Issues:

| Error | Solution |
|-------|----------|
| Missing config fields | Edit `study_config.json`, fill in `young_ages`, `old_ages`, etc. |
| Data file not found | Check path, use absolute path if needed |
| Wide-format not found | PHASE 1 needs manual processing - see docs |

**Full troubleshooting:** [`AUTONOMOUS_AGENT_GUIDE.md`](AUTONOMOUS_AGENT_GUIDE.md#debugging-failed-runs)

---

## 🔍 Key Concepts

### Missing Values (NaN)
- **Normal:** 50-80% missing values in LFQ proteomics
- **Meaning:** Protein not detected (biological reality)
- **Handling:** Preserve NaN, exclude from statistics (`skipna=True`)
- **Never:** Impute or remove NaN values

### Compartments
- **Keep separate:** "Kidney_Glomerular" not "Kidney"
- **Z-scores:** Calculate per compartment
- **Examples:** Glomerular/Tubulointerstitial, NP/IAF/OAF

### ECM Filtering
- **Filter:** `Match_Confidence > 0` before wide-format
- **Typical:** ~10-20% of proteins are ECM
- **Categories:** Core matrisome, matrisome-associated

---

## 📖 Documentation Priority

**Read in this order:**

1. **`00_START_HERE.md`** (this file) ← You are here
2. **`AUTONOMOUS_AGENT_GUIDE.md`** ← How to use the agent
3. **`README.md`** ← Full overview and examples
4. **`01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md`** ← If you need manual processing
5. **`02_ZSCORE_CALCULATION_UNIVERSAL_FUNCTION.md`** ← For z-score details

---

## ✅ Examples

### Example 1: Randles 2021 (Human Kidney)

```bash
python autonomous_agent.py "data_raw/Randles et al. - 2021/"

# Output:
# - Study ID: Randles_2021
# - Tissue: Kidney (2 compartments)
# - Proteins: 2,610 → 229 ECM
# - Ages: Young (15,29,37) vs Old (61,67,69)
```

### Example 2: Tam 2020 (Human Disc)

```bash
python autonomous_agent.py "data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx"

# Output:
# - Study ID: Tam_2020
# - Tissue: Intervertebral disc (4 compartments)
# - Proteins: 3,101 → 426 ECM
# - Ages: Young (16) vs Aged (59)
```

---

## 🚀 Next Steps

1. **Try the autonomous agent** with example study
2. **Monitor log file** in real-time
3. **Review output** in unified CSV
4. **Process your own study** when ready

---

## 📞 Support

- **Documentation:** Check README.md and guides
- **Examples:** See `05_Randles_paper_to_csv/` and `07_Tam_2020_paper_to_csv/`
- **Issues:** Review `agent_log.md` and `agent_state.json`
- **Contact:** daniel@improvado.io

---

**Last updated:** 2025-10-13
**Version:** 1.0
**Quick start:** `python autonomous_agent.py "data_raw/Study et al. - Year/"`
