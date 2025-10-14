# ECM-Atlas Repository Clean-Up Report

**Generated:** 2025-10-15
**Purpose:** Identify redundant, obsolete, and cluttered files/folders for potential archival or deletion

---

## Executive Summary

The repository has grown organically with 22 root-level items and multiple numbered experimental directories. Key issues identified:

- **163MB** of data in `05_paypers_to_csv/` with duplicate processing attempts
- **1.3GB** virtual environment tracked (should be ignored)
- Multiple dashboard versions (ver1, ver2) with unclear "active" version
- Duplicate documentation across `knowledge_base/` and `04_compilation_of_papers/`
- Scattered analysis reports across multiple directories
- Backup files not cleaned up

---

## 🔴 HIGH PRIORITY: Move to `obsolete/`

### 1. Virtual Environment ~~(CRITICAL)~~ ✅ **RESOLVED - NO ACTION NEEDED**
**Path:** `/env/`
**Size:** 1.3GB locally
**Status:** ✅ Already properly ignored by git (`.gitignore` line 20)
**Verification:** `git check-ignore` confirms it's not tracked
**Action:** **NONE - Keep it!** It's only on your local machine and won't be pushed to GitHub
**Impact:** No git repo size impact (already excluded)

### 2. Duplicate Angelidis Processing
**Paths:**
- `05_paypers_to_csv/09_Angelidis_2019_paper_to_csv/`
- `05_paypers_to_csv/09_01_Angelidis_2019_paper_to_csv/`

**Size:** ~50MB combined (163MB total for 05_paypers_to_csv)
**Reason:** Two nearly identical directories with different output formats (wide vs legacy). The `09_01_` appears to be a failed second attempt.
**Action:** Keep the one with better documentation (`09_Angelidis_2019_paper_to_csv`), move `09_01_` to obsolete
**Files differ:**
- agent_log.md (different logs)
- Output CSV formats differ (wide_format.csv vs LEGACY_format.csv)
- Only `09_` has LESSONS_LEARNED.md and SESSION_SUMMARY.md

### 3. Dashboard Version 1 (Superseded)
**Path:** `09_unified_dashboard_ver1/`
**Size:** 104KB
**Reason:** Superseded by `10_unified_dashboard_2_tabs/` which has more features (2 tabs, version tracking, better tests)
**Action:** Move to `obsolete/09_unified_dashboard_ver1/`
**Note:** Keep if you want version history for rollback

### 4. Experimental Multi-Agent Directories
**Paths:**
- `04_compilation_of_papers/02_multi_agent_parse_datasets/`
- `04_compilation_of_papers/03_age_bin_paper_focus/`
- `04_compilation_of_papers/ver_2_agent_ecm_parsing/`

**Reason:** Experimental/prototype directories with unclear production status. The comprehensive analysis MD files at root of `04_compilation_of_papers/` appear to be the final output.
**Action:** Move to `obsolete/04_compilation_experiments/`

### 5. Backup Files
**Path:** `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv.backup_20251013_160458`
**Size:** 489KB
**Reason:** Timestamped backup from Oct 13. Current version exists and is working.
**Action:** Move to `obsolete/backups/` or delete if git history is sufficient

---

## 🟡 MEDIUM PRIORITY: Consider for `obsolete/`

### 6. Duplicate Paper Analysis Documentation
**Paths:**
- `knowledge_base/01_paper_analysis/` (11 files, concise ~50 lines each)
- `04_compilation_of_papers/` (16 files, comprehensive ~250 lines each)

**Reason:** `knowledge_base/` files appear to be early drafts. The `04_compilation_of_papers/` versions are 5x more detailed and comprehensive.
**Action:**
- Keep `04_compilation_of_papers/` (comprehensive versions)
- Move `knowledge_base/01_paper_analysis/` to `obsolete/`
- Keep `knowledge_base/00_dataset_inventory.md`, `02_column_mapping_strategy.md`, `03_normalization_strategy.md`, `04_implementation_plan.md` (these are strategic docs, not duplicates)

### 7. Randles Paper Processing Subfolders
**Path:** `05_paypers_to_csv/05_Randles_paper_to_csv/`
**Contains:**
- `06_Randles_z_score_by_tissue_compartment/` (interactive dashboard - KEEP THIS!)
- `claude_code/`, `codex_cli/`, `gemini/` (tool-specific processing attempts)

**Reason:** The subdirectories `claude_code/`, `codex_cli/`, `gemini/` suggest experimental runs with different AI tools. Unclear which is canonical.
**Action:** Consolidate results, move tool-specific experiment folders to `obsolete/05_randles_experiments/`
**NOTE:** The `06_Randles_z_score_by_tissue_compartment/` is a WORKING DASHBOARD mentioned in CLAUDE.md - DO NOT MOVE

### 8. Calls Transcript Directory
**Path:** `calls_transcript/`
**Size:** 24KB
**Contains:** `download_calls.py`, README.md, .gitignore
**Reason:** Purpose unclear. Appears to be for downloading call transcripts (perhaps for meetings?). Not core to data pipeline.
**Action:** Move to `obsolete/calls_transcript/` if no longer actively used

### 9. Insights Directory
**Path:** `10_insights/`
**Size:** 144KB
**Contains:** GPT-generated analysis reports, a Word doc
**Reason:** Contains `01_gpt_pro_ver1*.md` files that appear to be early ChatGPT-generated analyses. The Word doc `ECM Aging Signature in Ovary.docx` might be a deliverable draft.
**Action:**
- Move GPT analysis drafts to `obsolete/10_early_insights/`
- Keep the Word doc if it's a working deliverable, otherwise move to `documentation/drafts/`

### 10. Empty/Minimal Directories
**Paths:**
- `data_processed/` (empty - intended for output)
- `documentation/` (7 files, 184KB)
- `12_crawl_datasets/` (1 file: research notes)

**Reason:**
- `data_processed/` is empty but is the intended output location per CLAUDE.md - KEEP
- `documentation/` has valuable pitch/vision docs - REORGANIZE but KEEP
- `12_crawl_datasets/` has only `00_gpt_pro_research.md` - move to `obsolete/` or merge into main docs

---

## 🟢 LOW PRIORITY: Reorganization Suggestions

### 11. Root-Level Task Documents
**Current:**
```
00_ECM_ATLAS_PIPELINE_OVERVIEW.md
01_TASK_DATA_STANDARDIZATION.md
02_TASK_PROTEIN_ANNOTATION_GUIDELINES.md
```

**Suggestion:** These are excellent documents but clutter root. Consider moving to:
```
documentation/tasks/
├── 00_ECM_ATLAS_PIPELINE_OVERVIEW.md
├── 01_TASK_DATA_STANDARDIZATION.md
└── 02_TASK_PROTEIN_ANNOTATION_GUIDELINES.md
```

**Rationale:** Root directory currently has 13 files. Best practice is to keep root minimal with only essential files (README, requirements, app.py, etc.)

### 12. Analysis Scripts at Root
**Current:**
```
analyze_aging_signatures.py (8.1KB)
calculate_missing_zscores.py (5.2KB)
create_inventory.py (3.4KB)
find_common_signatures.py (11KB)
read_excel.py (236 bytes - TRIVIAL, likely obsolete)
```

**Suggestion:** Move to `scripts/` or `analysis/` directory:
```
scripts/
├── analysis/
│   ├── analyze_aging_signatures.py
│   ├── find_common_signatures.py
│   └── calculate_missing_zscores.py
└── utilities/
    ├── create_inventory.py
    └── read_excel.py (or delete if obsolete)
```

### 13. Reports Directory Consolidation
**Path:** `reports/`
**Size:** 280KB, 14 files
**Contains:** Session summaries, analysis reports, UI/UX recommendations

**Suggestion:** This is well-organized but could be subdivided:
```
reports/
├── analysis/
│   ├── ECM_AGING_ANALYSIS_REPORT.md
│   ├── ECM_REMODELING_ENZYME_ANALYSIS.md
│   └── COMMON_AGING_SIGNATURES.csv
├── sessions/
│   └── 00_SESSION_2025-10-12_ECM_Atlas_MultiAgent.md
└── ui_ux/
    └── UI_UX_RECOMMENDATION.md
```

### 14. Documentation Directory Restructure
**Current state:** 7 files including pitch docs, scientific foundations, enzyme engineering vision

**Suggestion:**
```
documentation/
├── project/
│   ├── ECM_ATLAS_MASTER_OVERVIEW.md
│   └── README_NAVIGATION.md
├── scientific/
│   ├── 01_Scientific_Foundation.md
│   └── ECM_Enzyme_Engineering_Vision.md
├── business/
│   ├── ECM_ATLAS_LONGEVITY_PITCH.md
│   └── 04_Research_Insights.md
└── biomarkers/
    └── 04a_Biomarker_Framework.md
```

---

## 📊 Size Analysis Summary

| Directory/File | Size | Status |
|---------------|------|--------|
| `env/` | 1.3GB | **DELETE** |
| `05_paypers_to_csv/` | 163MB | Consolidate duplicates |
| `data_raw/` | ~500MB | Keep (source data) |
| `08_merged_ecm_dataset/` | 3.4MB | Keep (working data) |
| `pdf/` | Unknown | Keep (publications) |
| Numbered dirs (04,09,10,11,12) | ~1.5MB total | Mixed (see above) |

**Total space recoverable in git:** ~200MB (excluding `env/` which is already properly ignored)

---

## 🎯 Recommended Action Plan

### Phase 1: Immediate (Safety First)
1. ~~Move `env/`~~ ✅ **SKIP - Already properly ignored by git**
2. Create `obsolete/` directory at root
3. Move backup file → `obsolete/backups/`
4. Add `/obsolete/` to `.gitignore`

### Phase 2: Consolidation (Medium Risk)
5. Move `09_unified_dashboard_ver1/` → `obsolete/`
6. Move `05_paypers_to_csv/09_01_Angelidis_2019_paper_to_csv/` → `obsolete/`
7. Move `knowledge_base/01_paper_analysis/` → `obsolete/`
8. Move `04_compilation_of_papers/` experimental subdirs → `obsolete/`

### Phase 3: Reorganization (Optional, Low Risk)
9. Create `scripts/` directory and move analysis scripts
10. Create `documentation/tasks/` and move task documents
11. Restructure `documentation/` with subdirectories
12. Subdivide `reports/` for better organization

### Phase 4: Final Cleanup (After Testing)
13. Test that nothing breaks after moving files
14. If all tests pass for 1-2 weeks, delete `obsolete/` directory
15. Commit changes with message: "refactor: reorganize repository structure, remove obsolete files"

---

## ⚠️ Important Notes

### DO NOT MOVE/DELETE:
- `data_raw/` - Original source data (13 studies, 128 files)
- `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv` - Current working dataset
- `10_unified_dashboard_2_tabs/` - Active dashboard version
- `11_subagent_for_LFQ_ingestion/` - Active agent pipeline
- `05_paypers_to_csv/05_Randles_paper_to_csv/06_Randles_z_score_by_tissue_compartment/` - Working dashboard
- `CLAUDE.md` - Essential for Claude Code
- `README.md` - Essential documentation
- `requirements.txt` - Dependency specification
- `app.py` - Application entry point

### Git Ignore Check
Current `.gitignore` already excludes:
- `env/` ✅
- `*.backup` ✅
- `*.bak` ✅
- `.obsidian/` ✅
- `*.pyc` ✅

### Suggested `.gitignore` additions:
```
# Obsolete files archive
obsolete/

# Large data backups (use git history instead)
*.csv.backup*
08_merged_ecm_dataset/backups/
```

---

## 🔍 Questions to Resolve Before Cleanup

1. **Dashboard versions:** Is `09_unified_dashboard_ver1/` completely superseded or do you need to preserve it?
2. **Angelidis duplicates:** Which output format is canonical - wide or legacy?
3. **Randles tool experiments:** Which processing approach (claude_code/codex_cli/gemini) was final?
4. **Calls transcript:** Is this directory still actively used?
5. **References directory:** Is `Delocalized entropy aging theorem.pdf` and `vitalabs-ecm-proposal.pdf` needed for active work?

---

## 📁 Proposed Final Structure (After Cleanup)

```
ecm-atlas/
├── app.py
├── requirements.txt
├── README.md
├── CLAUDE.md
├── .gitignore
├── data_raw/          [13 study directories]
├── data_processed/    [empty, for output]
├── pdf/               [publications]
├── references/        [matrisome lists, proposals]
├── documentation/
│   ├── tasks/
│   ├── scientific/
│   ├── business/
│   └── biomarkers/
├── scripts/
│   ├── analysis/
│   └── utilities/
├── reports/
│   ├── analysis/
│   ├── sessions/
│   └── ui_ux/
├── 08_merged_ecm_dataset/     [active data]
├── 10_unified_dashboard_2_tabs/ [active dashboard]
├── 11_subagent_for_LFQ_ingestion/ [active pipeline]
└── obsolete/          [to be deleted later]
    ├── env/
    ├── 04_compilation_experiments/
    ├── 05_angelidis_duplicate/
    ├── 09_unified_dashboard_ver1/
    ├── knowledge_base_paper_analysis/
    └── backups/
```

---

## Next Steps

1. Review this report
2. Answer the questions in "Questions to Resolve"
3. Create a backup of entire repo before any changes
4. Execute Phase 1 (safest moves)
5. Test thoroughly
6. Proceed to Phase 2 if comfortable
7. Optional: Execute Phase 3 for better organization

---

**Total items to move to obsolete:** 8-12 directories/files
**Estimated time to execute:** 15-30 minutes
**Risk level:** Low (if using move instead of delete)
**Disk space recovery:** ~1.5GB
