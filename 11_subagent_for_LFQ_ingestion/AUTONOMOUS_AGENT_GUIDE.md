# Autonomous Agent User Guide

## Overview

The `autonomous_agent.py` script is a fully automated orchestrator that processes LFQ proteomics datasets from start to finish with complete logging and debugging support.

---

## Quick Start

### Basic Usage

```bash
# Point to paper folder
python autonomous_agent.py "data_raw/Randles et al. - 2021/"

# Or specify exact data file
python autonomous_agent.py "data_raw/Randles et al. - 2021/ASN.2020101442-File027.xlsx"
```

### What Happens

The agent automatically:

1. **PHASE 0: Reconnaissance**
   - Identifies paper folder and extracts Study ID
   - Finds all data files (.xlsx, .xls, .csv, .tsv)
   - Selects largest file (or uses specified file)
   - Inspects Excel sheets and columns
   - Generates configuration template

2. **PHASE 1: Data Normalization**
   - Validates configuration
   - Executes full PHASE 1 pipeline
   - Creates wide-format CSV

3. **PHASE 2: Merge to Unified**
   - Calls `merge_to_unified.py`
   - Adds study to unified CSV
   - Creates backup

4. **PHASE 3: Z-Score Calculation**
   - Calls `universal_zscore_function.py`
   - Calculates z-scores for new study only
   - Updates unified CSV in-place

---

## Output Structure

After running, the agent creates:

```
XX_Author_Year_paper_to_csv/
├── agent_log.md              # Sequential log with timestamps
├── agent_state.json          # Current state (phase, status, errors)
├── study_config.json         # Generated configuration
└── Author_Year_wide_format.csv  # Final output (PHASE 1)
```

---

## Real-Time Monitoring

### Track Progress

Open a second terminal and run:

```bash
tail -f XX_Author_Year_paper_to_csv/agent_log.md
```

You'll see:
```
[2025-10-13 15:30:45] ✅ Workspace initialized
[2025-10-13 15:30:45] ## PHASE 0: Reconnaissance
[2025-10-13 15:30:45] ### Step 0.1: Identify paper folder
[2025-10-13 15:30:45] Paper folder: data_raw/Randles et al. - 2021
[2025-10-13 15:30:45] Detected Study ID: Randles_2021
[2025-10-13 15:30:45] ✅ Completed: Identify paper folder
```

### Check Current State

```bash
cat XX_Author_Year_paper_to_csv/agent_state.json
```

Example output:
```json
{
  "phase": "merge",
  "current_step": "Execute merge",
  "start_time": "2025-10-13T15:30:45.123456",
  "status": "running",
  "completed_steps": [
    {"step": "Identify paper folder", "timestamp": "2025-10-13T15:30:45.234567"},
    {"step": "Find data files", "timestamp": "2025-10-13T15:30:46.345678"},
    ...
  ]
}
```

---

## Debugging Failed Runs

### If Agent Stops with Error

1. **Check the log file:**
   ```bash
   less XX_Author_Year_paper_to_csv/agent_log.md
   ```

2. **Look for ❌ ERROR markers:**
   ```
   [2025-10-13 15:30:48] ❌ ERROR: Missing required config fields: ['young_ages', 'old_ages']
   ```

3. **Check the traceback:**
   ```
   Traceback (most recent call last):
     File "autonomous_agent.py", line 245, in _normalize_data
       if missing_fields:
   ValueError: Missing required config fields
   ```

4. **Review agent state:**
   ```bash
   cat XX_Author_Year_paper_to_csv/agent_state.json
   ```

### Common Errors and Solutions

#### Error: Missing required config fields

**Cause:** Configuration template needs manual review

**Solution:**
```bash
# Edit configuration
nano XX_Author_Year_paper_to_csv/study_config.json

# Fill in:
{
  "study_id": "Randles_2021",
  "species": "Homo sapiens",
  "tissue": "Kidney",
  "young_ages": [15, 29, 37],
  "old_ages": [61, 67, 69],
  "compartments": {"G": "Glomerular", "T": "Tubulointerstitial"}
}

# Re-run agent
python autonomous_agent.py "data_raw/Randles et al. - 2021/"
```

#### Error: Data file not found

**Cause:** Incorrect path or file doesn't exist

**Solution:**
```bash
# Check path
ls "data_raw/Randles et al. - 2021/"

# Use absolute path if needed
python autonomous_agent.py "/full/path/to/data_raw/Randles et al. - 2021/"
```

#### Error: Wide-format file not found

**Cause:** PHASE 1 processing not yet implemented or failed

**Solution:**
- Review PHASE 1 instructions: `01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md`
- Manually process study following documentation
- Create `Author_Year_wide_format.csv` in output directory
- Agent will continue from PHASE 2

---

## Advanced Usage

### Resume from Specific Phase

The agent can resume if you've manually completed earlier phases:

```bash
# If you already have wide_format.csv:
# 1. Create output folder: XX_Author_Year_paper_to_csv/
# 2. Place wide_format.csv inside
# 3. Create study_config.json with required fields
# 4. Run agent - it will skip PHASE 1 and continue with PHASE 2
python autonomous_agent.py "data_raw/Author et al. - Year/"
```

### Custom Project Root

```python
from autonomous_agent import LFQProcessingAgent
from pathlib import Path

agent = LFQProcessingAgent(
    input_path="data_raw/Study et al. - 2024/",
    project_root=Path("/custom/path/to/ecm-atlas")
)
agent.run()
```

---

## Log Format

### Timestamps

All log entries have timestamps:
```
[YYYY-MM-DD HH:MM:SS] Message
```

### Markers

- ✅ Completed step
- ❌ Error occurred
- ⚠️  Warning or manual action required
- 📊 Data insights
- 🔗 Links or references

### Structure

```markdown
# Autonomous LFQ Processing Agent Log

**Study ID:** Author_Year
**Start Time:** ISO timestamp
**Input Path:** ...
**Output Directory:** ...

---

## PHASE 0: Reconnaissance

### Step 0.1: Description
[timestamp] Log message
[timestamp] ✅ Completed: Step name

### Step 0.2: Description
...

---

## PHASE 1: Data Normalization

...

---

## PIPELINE COMPLETE

**Total Steps Completed:** N
**Total Time:** Xm Ys
```

---

## State File Format

```json
{
  "phase": "current_phase_name",
  "current_step": "current_step_description",
  "start_time": "ISO_timestamp",
  "status": "running|completed|error",
  "errors": [
    {
      "error": "error_message",
      "timestamp": "ISO_timestamp",
      "traceback": "full_python_traceback"
    }
  ],
  "completed_steps": [
    {
      "step": "step_name",
      "timestamp": "ISO_timestamp"
    }
  ],
  "last_updated": "ISO_timestamp"
}
```

---

## Integration with Existing Scripts

The autonomous agent calls:

1. **`merge_to_unified.py`** (PHASE 2)
   ```python
   from merge_to_unified import merge_study_to_unified
   df_merged = merge_study_to_unified(
       study_csv="XX_Study/Study_wide_format.csv",
       unified_csv="08_merged_ecm_dataset/ECM_Atlas_Unified.csv"
   )
   ```

2. **`universal_zscore_function.py`** (PHASE 3)
   ```python
   from universal_zscore_function import calculate_study_zscores
   df_updated = calculate_study_zscores(
       study_id="Study_2021",
       groupby_columns=["Tissue_Compartment"],
       backup=True
   )
   ```

---

## Configuration Template

Generated `study_config.json`:

```json
{
  "study_id": "Author_Year",
  "paper_folder": "data_raw/Author et al. - Year",
  "data_file": "data_raw/Author et al. - Year/data.xlsx",
  "data_sheet": "Sheet name",
  "species": "Homo sapiens",
  "tissue": "Kidney",
  "method": "Label-free LC-MS/MS",
  "young_ages": [15, 29, 37],
  "old_ages": [61, 67, 69],
  "compartments": {
    "G": "Glomerular",
    "T": "Tubulointerstitial"
  },
  "output_dir": "XX_Author_Year_paper_to_csv"
}
```

**Required fields:**
- `study_id`
- `species` (Homo sapiens | Mus musculus)
- `tissue`
- `young_ages` (list of ages)
- `old_ages` (list of ages)
- `data_file`
- `data_sheet` (for Excel files)

**Optional fields:**
- `compartments` (dict or null)
- `method` (descriptive string)
- `paper_pmid` (PubMed ID)
- `paper_doi` (DOI)

---

## Examples

### Example 1: Randles 2021

```bash
python autonomous_agent.py "data_raw/Randles et al. - 2021/"

# Agent detects:
# - Study ID: Randles_2021
# - Data file: ASN.2020101442-File027.xlsx (largest)
# - Sheet: Human data matrix fraction
# - Columns: Accession, Description, Gene name, G15, T15, ...

# Creates:
# XX_Randles_2021_paper_to_csv/
#   ├── agent_log.md
#   ├── agent_state.json
#   ├── study_config.json (needs manual review)
#   └── Randles_2021_wide_format.csv (after PHASE 1)

# Edit config, re-run, completes all phases
```

### Example 2: Tam 2020

```bash
python autonomous_agent.py "data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx"

# Agent detects:
# - Study ID: Tam_2020
# - Data file: elife-64940-supp1-v3.xlsx (specified)
# - Sheets: Raw data, Sample information
# - Columns: T: Majority protein IDs, T: Protein names, ...

# Processes with 4 compartments (NP, IAF, OAF, NP/IAF)
```

---

## Tips

### Parallel Processing

You can run multiple agents in parallel for different studies:

```bash
# Terminal 1
python autonomous_agent.py "data_raw/Study1 et al. - 2020/" &

# Terminal 2
python autonomous_agent.py "data_raw/Study2 et al. - 2021/" &

# Monitor both
tail -f XX_Study1_2020_paper_to_csv/agent_log.md
tail -f XX_Study2_2021_paper_to_csv/agent_log.md
```

### Dry Run

Check what the agent will do without running:

```python
from autonomous_agent import LFQProcessingAgent

agent = LFQProcessingAgent("data_raw/Study et al. - 2024/")
# Only run reconnaissance
agent._reconnaissance()
# Check generated config
import json
with open(agent.output_dir / "study_config.json") as f:
    print(json.dumps(json.load(f), indent=2))
```

---

## Troubleshooting

### Agent hangs at PHASE 1

**Cause:** PHASE 1 processing not yet implemented

**Solution:**
- Manually follow `01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md`
- Create wide-format CSV
- Agent will continue from PHASE 2

### Log file not updating in real-time

**Cause:** Output buffering

**Solution:**
```bash
# Use unbuffered mode
python -u autonomous_agent.py "data_raw/Study et al. - 2024/"

# Or use tail -f for real-time monitoring
```

### Study ID extraction fails

**Cause:** Folder name doesn't match "Author et al. - Year" pattern

**Solution:** Agent will use folder name as Study ID. Edit `study_config.json` to correct.

---

## Support

### If something goes wrong:

1. **Check log file:** `agent_log.md`
2. **Check state file:** `agent_state.json`
3. **Review configuration:** `study_config.json`
4. **Check documentation:** `01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md`

### Example output structures:

- **EXAMPLE_agent_log.md** - Sample log file
- **EXAMPLE_agent_state.json** - Sample state file

---

**Last updated:** 2025-10-13
**Version:** 1.0
**Author:** Claude Code
