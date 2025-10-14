## Task: Parse 13 ECM Proteomic Datasets into Unified Schema

**🔴 CRITICAL: NEVER MOCK DATA! Use real Excel/TSV files from data_raw/ directory. If file cannot be read - stop and document attempts.**

---

## Context

ECM-Atlas repository contains 13 proteomic studies (2017-2023) in `data_raw/` directory. Each study has Excel/TSV files with protein abundance data but different formats. Goal: parse ALL datasets into unified 12-column schema and create normalized database ready for Streamlit chatbot demo.

**Repository:** `/Users/Kravtsovd/projects/ecm-atlas/`

**Reference docs:**
- `00_REPO_OVERVIEW.md` - Repository structure (128 files, 13 studies)
- `01_TASK_DATA_STANDARDIZATION.md` - Detailed schema specification and normalization methods

---

## Unified Schema (12 Columns)

```
Protein_ID        - UniProt ID or Gene symbol (e.g., P02452, COL1A1)
Protein_Name      - Full protein name (e.g., "Collagen alpha-1(I) chain")
Gene_Symbol       - Gene nomenclature (e.g., COL1A1)
Tissue            - Organ type (lung, skin, heart, kidney, ovary, etc.)
Species           - Organism (Mus musculus, Homo sapiens, Bos taurus)
Age               - Numeric age value
Age_Unit          - Time unit (years, months, weeks)
Abundance         - Quantitative measurement value
Abundance_Unit    - Measurement unit (intensity, LFQ, ppm, spectral_counts)
Method            - Proteomic technique (LC-MS/MS, DIA, TMT, LFQ)
Study_ID          - Publication identifier (Author_Year format, e.g., Angelidis_2019)
Sample_ID         - Biological/technical replicate ID
```

---

## Success Criteria

### ✅ Data Parsing (5/5 required)
- [✅] **Parse 13 datasets**: Extract protein data from ALL 13 study directories in `data_raw/`
- [✅] **Map to schema**: Convert each dataset to unified 12-column format
- [✅] **Handle formats**: Successfully parse Excel (.xlsx, .xls), TSV, and mixed formats
- [✅] **Extract metadata**: Tissue, Species, Method from paper filenames or content
- [✅] **Preserve raw data**: Keep original abundance values (no normalization yet)

### ✅ Data Quality (4/4 required)
- [✅] **Protein IDs standardized**: Map all protein identifiers to UniProt or Gene symbols
- [✅] **No empty required fields**: Protein_ID, Abundance, Study_ID must be present for ALL rows
- [✅] **Age groups identified**: Correctly label "Young" vs "Old" from column headers/metadata
- [✅] **Abundance units documented**: Identify unit type (intensity/LFQ/ppm) per study from paper

### ✅ Output & Documentation (3/3 required)
- [✅] **Create data_processed/ directory**: Store parsed CSVs per study (e.g., `Angelidis_2019_parsed.csv`)
- [✅] **Generate metadata.json**: Document tissue, species, age_young, age_old, method, abundance_unit per study
- [✅] **Create validation report**: Show row counts, missing values, unique proteins per study

### ✅ Code Quality (2/2 required)
- [✅] **Reusable parsing functions**: Generic parser that works for ANY new study with minor config
- [✅] **Error handling**: Graceful failures with clear error messages for corrupt/missing files

---

## Dataset Inventory (13 Studies)

| Study | Files | Size | Tissue | Notes |
|-------|-------|------|--------|-------|
| Angelidis et al. 2019 | 11 | 59MB | Lung | MOESM format (Nature), main dataset: MOESM5_ESM.xlsx (12MB) |
| Ariosa-Morejon et al. 2021 | 5 | - | - | eLife format: fig*-data*.xlsx |
| Caldeira et al. 2017 | 7 | - | - | MOESM format, intensity sums |
| Chmelova et al. 2023 | 6 | - | - | Data Sheet format |
| Dipali et al. 2023 | 12 | 41MB | Ovary | DIA method, Candidates.tsv (5MB), Report_*.xls (29MB peptides) |
| Li et al. 2021 \| dermis | 5 | - | Skin | - |
| Li et al. 2021 \| pancreas | 5 | - | Pancreas | - |
| Lofaro et al. 2021 | - | - | Kidney | - |
| McCabe et al. 2020 | - | - | Lung | Matrisome focus |
| Ouni et al. 2022 | - | - | Adipose | - |
| Randles et al. 2021 | - | - | Kidney | Glomeruli-specific |
| Tam et al. 2020 | - | - | Heart | Cardiac tissue |
| Tsumagari et al. 2023 | - | - | - | Latest publication |

---

## Parsing Strategy

### Step 1: Per-Study Analysis
For EACH study in `data_raw/[StudyName]/`:
1. **Identify main data file**: Largest Excel/TSV file with protein IDs and abundances
2. **Extract metadata**:
   - Tissue: from folder name or README
   - Species: from paper or filenames
   - Method: from filenames (DIA, LC-MS, TMT) or default to "LC-MS/MS"
3. **Parse protein table**:
   - Find columns with UniProt IDs or Gene symbols
   - Find abundance columns (often multiple replicates: Young_1, Young_2, Old_1, Old_2)
   - Identify age groups from column headers
4. **Reshape to long format**: Each row = one protein in one sample

### Step 2: Standardization
1. **Protein ID mapping**: If only gene symbols → keep as-is, if mixed → prefer UniProt
2. **Abundance unit detection**:
   - Check column names: "Intensity" → intensity, "LFQ" → LFQ
   - Check value ranges: <1000 → LFQ, >100000 → raw intensity
3. **Age normalization**: Map "3mo"→3 months, "24mo"→24 months, etc.

### Step 3: Validation
1. **Count proteins per study**: Expect 200-5000 proteins per study
2. **Check coverage**: All 13 studies should produce at least one parsed CSV
3. **Spot check**: Manually verify 2-3 studies have correct Protein_IDs and Abundance values

---

## Example Output Structure

```
ecm-atlas/
├── data_processed/                          # NEW: Parsed datasets
│   ├── Angelidis_2019_parsed.csv           # ~3000 rows (proteins × samples)
│   ├── Dipali_2023_parsed.csv              # ~2000 rows
│   ├── ... [11 more CSVs]
│   ├── metadata.json                        # Study-level metadata
│   └── validation_report.md                 # Parsing statistics
├── data_raw/                                # UNCHANGED: Original data
│   ├── Angelidis et al. - 2019/
│   └── ...
└── 02_multi_agent_parse_datasets/           # THIS TASK
    ├── 01_task_multi_agent_ecm_parse.md     # This file
    ├── claude_code/
    │   ├── 01_ecm_parse_plan_claude_code.md
    │   ├── 90_ecm_parse_results_claude_code.md
    │   └── parse_datasets.py                # Code artifacts
    ├── codex_cli/
    │   ├── 01_ecm_parse_plan_codex.md
    │   ├── 90_ecm_parse_results_codex.md
    │   └── parse_datasets.py
    └── gemini/
        ├── 01_ecm_parse_plan_gemini.md
        ├── 90_ecm_parse_results_gemini.md
        └── parse_datasets.py
```

---

## Known Challenges

1. **Heterogeneous formats**: Nature journals use `MOESM[N]_ESM.xlsx`, eLife uses `fig[N]-data[N].xlsx`, Frontiers uses `Data Sheet [N].XLSX`
2. **Protein ID variety**: Some studies use UniProt (P02452), others Gene symbols (COL1A1), some both
3. **Missing metadata**: Tissue/Species may only be in paper title, not in data files
4. **Large files**: Angelidis MOESM8 is 22MB, Dipali peptide report is 29MB - need efficient pandas loading
5. **Age group naming**: "Young" vs "young" vs "Y", "Old" vs "old" vs "O" - case-insensitive matching required

---

## <Agents artefact requrement>

### Required Artifacts

Each agent MUST produce:

1. **01_ecm_parse_plan_[agent].md**
   ```markdown
   ## My Approach ([Agent Name])
   - [ ] Step 1: Analyze data_raw/ directory structure
   - [ ] Step 2: Create parsing script for Excel/TSV files
   - [ ] Step 3: Test on 2 example studies (Angelidis 2019, Dipali 2023)
   - [ ] Step 4: Parse all 13 studies
   - [ ] Step 5: Generate metadata.json and validation_report.md

   ## Progress Updates
   ✅ [timestamp] Step 1 complete: Found 13 study directories
   ✅ [timestamp] Step 2 complete: parse_datasets.py created
   ...
   ```

2. **90_ecm_parse_results_[agent].md**
   ```markdown
   ## Self-Evaluation ([Agent Name])

   ### Criterion 1: Parse 13 datasets
   **Status:** ✅/❌/⚠️
   **Evidence:** Found X Excel files, parsed Y studies successfully
   **Details:** [List files parsed, any failures]

   ### Criterion 2: Map to schema
   **Status:** ✅/❌/⚠️
   **Evidence:** All 12 columns present in output CSVs
   **Details:** [Show sample rows from parsed data]

   [... repeat for all 14 criteria]

   ## Overall Score
   **Criteria Met:** X/14
   **Grade:** ✅ (12+) / ⚠️ (8-11) / ❌ (<8)

   ## Code Location
   - Main script: `[agent]/parse_datasets.py`
   - Output: `../data_processed/`
   ```

3. **parse_datasets.py** (or similar code)
   - Parsing functions for Excel/TSV
   - Schema mapping logic
   - Metadata extraction
   - Validation checks

4. **Optional: sample_output.csv** (first 100 rows of parsed data for quick validation)

---

## Evaluation Rules

- **Winner = Most criteria met** (target: 14/14)
- **Tie-breaker 1:** Fewest parsing errors/warnings
- **Tie-breaker 2:** Most complete metadata.json (all fields populated)
- **Disqualification:** Mocking data instead of parsing real files

---

## Time Limit

- **Expected:** 30-60 minutes for full task
- **Checkpoint 1:** After 15 min - at least 2 studies parsed
- **Checkpoint 2:** After 30 min - at least 8 studies parsed

---

**Ready to launch agents!** Run from repository root:
```bash
cd /Users/Kravtsovd/projects/chrome-extension-tcs
./algorithms/product_div/Multi_agent_framework/run_parallel_agents.sh \
  ../ecm-atlas/02_multi_agent_parse_datasets/01_task_multi_agent_ecm_parse.md
```
