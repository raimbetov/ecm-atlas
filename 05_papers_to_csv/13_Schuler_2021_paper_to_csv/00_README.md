# Schüler et al. 2021 - Paper to CSV Processing

**Study ID:** Schuler_2021
**Status:** 📋 **Ready for Processing**
**Last Updated:** 2025-10-15

---

## 📚 Publication Details

- **Title:** Extensive remodeling of the extracellular matrix during aging contributes to age-dependent impairments of muscle stem cell functionality
- **Authors:** Svenja C. Schüler, Joanna M. Kirkpatrick, Manuel Schmidt, et al.
- **Journal:** Cell Reports
- **Year:** 2021
- **Volume/Issue:** 35(10):109223
- **DOI:** 10.1016/j.celrep.2021.109223
- **PMID:** 34107247
- **PRIDE:** PXD015728

---

## 🔬 Study Characteristics

### Biological Context
- **Species:** Mouse (Mus musculus)
- **Tissue:** Skeletal muscle - 4 different muscle types
- **Cell Type Focus:** Muscle stem cell (MuSC) niche and bulk muscle ECM
- **Method:** LFQ proteomics with Data-Independent Acquisition (DIA)
- **Age Groups:** Young (3 months) vs Old (18 months)

### Scientific Significance
- First comprehensive characterization of MuSC niche ECM changes with aging
- Identifies SMOC2 as age-accumulating protein contributing to aberrant signaling
- Provides both niche and bulk muscle ECM data for skeletal muscle aging

---

## 📁 Folder Contents

### 1. Assessment & Planning Documents

#### **SCHULER_2021_DATASET_ASSESSMENT.md** (Primary Overview)
- Complete dataset assessment and readiness check
- Prerequisites verification
- Processing workflow proposal
- Expected outputs and timeline
- Decision points requiring approval

#### **SCHULER_2021_DATA_FILE_SELECTION.md** (Data File Analysis)
- Comprehensive review of all 9 supplementary files (mmc1-mmc9)
- Identification of best file for ECM-Atlas (mmc4.xls)
- Comparison of different proteomics datasets
- Final recommendation: mmc4.xls with 4 muscle types

### 2. Technical Reports

#### **schuler_2021_data_inspection.md** (Data Structure Report)
- Automated inspection of all Excel/CSV files
- Sheet names and column structures
- Proteomics indicator detection
- Data preview for each file

#### **MATRISOME_ANNOTATION_CONFIRMATION.md** (Annotation Verification)
- Confirmation that Naba Lab matrisome annotations will be applied
- Explanation of 4-level annotation hierarchy
- Expected annotation coverage (>95% for pre-filtered ECM data)
- Matrisome category breakdown

#### **PATH_FIXES_REPORT.md** (Technical Validation)
- Documentation of all path fixes applied to processing scripts
- Database filename corrections
- Project root auto-detection verification
- Full validation checklist

### 3. Processing Scripts

#### **inspect_schuler_data.py**
- Automated data file inspection script
- Generates schuler_2021_data_inspection.md
- Checks Excel sheet structures and column names
- Identifies proteomics data indicators

#### **audit_paths.py**
- Path validation and audit script
- Checks project directory structure
- Verifies matrisome reference availability
- Validates script path configurations

---

## 🎯 Recommended Data File: mmc4.xls

### Why mmc4.xls?

**Advantages:**
- ✅ **Pre-filtered for ECM proteins** (compartment = "Extracellular")
- ✅ **4 muscle types** (comprehensive skeletal muscle coverage)
- ✅ **Clean data structure** (ready-to-use abundances)
- ✅ **Small file size** (0.27 MB - fast processing)
- ✅ **UniProt IDs provided**
- ✅ **Old vs Young comparison** (18m vs 3m)

### mmc4.xls Structure

**4 Sheets (4 muscle types):**
1. `1_S O vs. Y` - Soleus (Old vs Young)
2. `2_G O vs. Y` - Gastrocnemius (Old vs Young)
3. `3_TA O vs. Y` - Tibialis Anterior (Old vs Young)
4. `4_EDL O vs. Y` - Extensor Digitorum Longus (Old vs Young)

**Columns per sheet:**
- `uniprot` - UniProt protein ID
- `sample1_abundance` - Young abundance (3 months)
- `sample2_abundance` - Old abundance (18 months)
- `short.name` - Gene symbol
- `accession` - UniProt accession
- `compartment` - "Extracellular" (pre-filtered!)
- Statistical columns (residuals, CNV values, q-values)

**Example Data:**
```csv
uniprot  sample1_abundance  sample2_abundance  short.name  compartment
Q64739   15.023363          16.642141          Col11a2     Extracellular
Q8R1Q3   13.839816          15.308943          Angptl7     Extracellular
Q61646   14.301732          15.627979          Hp          Extracellular
```

---

## 🔄 Processing Workflow

### **Command to Run:**
```bash
cd /home/raimbetov/GitHub/ecm-atlas/11_subagent_for_LFQ_ingestion
python autonomous_agent.py "../data_raw/Schuler et al. - 2021/mmc4.xls"
```

### **Expected Phases:**

#### **PHASE 0: Reconnaissance**
- Identify study folder and extract Study_ID: `Schuler_2021`
- Find data file: `mmc4.xls`
- Create output workspace: `XX_Schuler_2021_paper_to_csv/`
- Generate configuration template

#### **PHASE 1: Data Normalization**
- Load mouse matrisome reference
- Process 4 sheets (Soleus, Gastrocnemius, TA, EDL)
- Annotate with matrisome categories (Naba Lab)
- Match proteins by Gene Symbol + UniProt ID
- Generate wide-format CSV

#### **PHASE 2: Merge to Unified Database**
- Load current database: `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- Create timestamped backup
- Merge Schuler_2021 data
- Validate schema consistency

#### **PHASE 3: Z-Score Calculation**
- Calculate z-scores per tissue compartment
- Group by: Tissue (4 muscle types)
- Update database with z-scores
- Generate metadata JSON

---

## 📊 Expected Outputs

### **Workspace Directory:**
```
XX_Schuler_2021_paper_to_csv/
├── agent_log.md                    # Sequential execution log
├── agent_state.json                # Current processing state
├── study_config.json               # Auto-generated configuration
└── Schuler_2021_wide_format.csv    # Processed wide-format data
```

### **Updated Database:**
```
08_merged_ecm_dataset/
├── merged_ecm_aging_zscore.csv     # UPDATED with Schuler_2021
├── zscore_metadata_Schuler_2021.json
└── backups/
    └── merged_ecm_aging_zscore_2025-10-15_XX-XX-XX.csv
```

### **Expected Data Volume:**
- **ECM proteins per muscle:** ~100-200
- **Total muscles:** 4 (Soleus, Gastrocnemius, TA, EDL)
- **Total new rows:** ~1,600-3,200
- **Matrisome match rate:** >95% (pre-filtered data)

---

## 🏷️ Database Integration Details

### **Study Metadata:**
```json
{
  "Study_ID": "Schuler_2021",
  "Species": "Mus musculus",
  "Tissue": "Skeletal muscle",
  "Method": "LFQ (DIA)",
  "Age_Young": 3,
  "Age_Old": 18,
  "Age_Unit": "months"
}
```

### **Tissue Compartments (4):**
1. `Skeletal_muscle_Soleus`
2. `Skeletal_muscle_Gastrocnemius`
3. `Skeletal_muscle_TA`
4. `Skeletal_muscle_EDL`

### **Matrisome Annotations:**
All proteins will be annotated with:
- `Matrisome_Division` (Core matrisome / Matrisome-associated)
- `Matrisome_Category` (Collagens, ECM Glycoproteins, etc.)
- `Canonical_Gene_Symbol` (Naba Lab standardized)
- `Match_Confidence` (100 = ECM, 0 = Non-ECM)
- `Match_Level` (exact_gene / Gene_Symbol_or_UniProt)

---

## ✅ Prerequisites Checklist

Before processing, verify:

- [x] **Data files present** - `data_raw/Schuler et al. - 2021/mmc4.xls`
- [x] **Matrisome references** - `references/mouse_matrisome_v2.csv`
- [x] **Database accessible** - `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- [x] **xlrd installed** - For reading .xls files
- [x] **Paths fixed** - All scripts use auto-detection
- [x] **Pipeline ready** - `11_subagent_for_LFQ_ingestion/autonomous_agent.py`

---

## 🔍 Quality Assurance

### **Expected Validation Points:**
1. All 4 muscle types processed successfully
2. Matrisome annotation rate >95%
3. Z-scores calculated per muscle compartment
4. No duplicate rows in final database
5. SMOC2 protein present (key finding from paper)
6. Backup created before database update

### **Post-Processing Verification:**
```bash
# Check Schuler_2021 in database
source env/bin/activate
python3 -c "
import pandas as pd
df = pd.read_csv('08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
schuler = df[df['Study_ID'] == 'Schuler_2021']
print(f'Schuler rows: {len(schuler)}')
print(f'Tissues: {schuler[\"Tissue_Compartment\"].unique()}')
print(f'Matrisome match rate: {(schuler[\"Match_Confidence\"]==100).sum() / len(schuler) * 100:.1f}%')
"
```

---

## 📚 Related Files

### **In data_raw/**
- `Schuler et al. - 2021/mmc4.xls` - Primary data file
- `Schuler et al. - 2021/mmc2.xls` - MuSC proteome (alternative)
- `Schuler et al. - 2021/mmc3.xls` - Bulk muscle TMT data

### **In pdf/**
- `Schuler et al. - 2021.pdf` - Full publication

### **In 04_compilation_of_papers/**
- `13_Schuler_2021_comprehensive_analysis.md` - Detailed paper analysis

### **In references/**
- `mouse_matrisome_v2.csv` - Matrisome reference for annotation

---

## 📝 Processing Notes

### **Key Considerations:**
1. **Pre-filtered Data:** mmc4.xls already contains only extracellular proteins
2. **Multiple Muscles:** Process all 4 sheets as separate tissue compartments
3. **Age Metadata:** Young = 3 months, Old = 18 months (standard MuSC study design)
4. **SMOC2 Validation:** Verify this key protein is present and shows age increase
5. **Compartment Naming:** Use format `Skeletal_muscle_[MuscleType]`

### **Potential Issues:**
- **Sheet Names:** Ensure autonomous agent handles space-separated sheet names
- **Abundance Units:** Verify log2-transformed abundances are handled correctly
- **UniProt Mapping:** Some proteins may need UniProt ID matching if gene symbol fails

---

## 🎯 Success Criteria

Processing is successful when:

1. ✅ All 4 muscle types processed without errors
2. ✅ Schuler_2021 appears in unified database
3. ✅ ~1,600-3,200 new rows added
4. ✅ Matrisome annotations applied (>95% match)
5. ✅ Z-scores calculated per muscle compartment
6. ✅ Backup created successfully
7. ✅ Dashboard shows Schuler_2021 in dropdown
8. ✅ SMOC2 protein present with age-related changes

---

## 🚀 Next Steps

1. **Review all documentation** in this folder
2. **Confirm processing approach** (all 4 muscles recommended)
3. **Run autonomous agent** with mmc4.xls
4. **Monitor progress** via `agent_log.md`
5. **Validate outputs** using verification script
6. **View in dashboard** at http://localhost:8080/dashboard.html

---

**Status:** 📋 **READY FOR PROCESSING**
**Estimated Time:** 15-25 minutes
**Risk Level:** Low (automatic backups, validated paths)
**Expected Outcome:** Schuler_2021 with 4 skeletal muscle compartments in ECM-Atlas database
