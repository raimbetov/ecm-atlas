# H21 - Manual Download Guide for External Datasets

**Status:** Playwright automation blocked by JavaScript-heavy journal websites
**Alternative:** Manual download instructions + direct HTTP access where possible

---

## HIGH PRIORITY (CRITICAL for H16)

### 1. PXD011967 - Ferri 2019 Muscle Aging

**Publication:**
- eLife: https://elifesciences.org/articles/49874
- Also in Nature Scientific Reports: PMC6803624
- DOI: 10.7554/eLife.49874

**Data Sources:**
1. **ProteomeXchange PRIDE:**
   - FTP: ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2019/10/PXD011967/
   - Files: Contains RAW MS files (large, requires processing)

2. **eLife Figure Source Data:**
   - Navigate to: https://elifesciences.org/articles/49874
   - Find "Figure 1—source data 1.xlsx" (proteomic quantification, 5,891 proteins)
   - Find "Figure 1—source data 3.xlsx" (age-associated proteins)
   - **Action:** Manual download from webpage

3. **Alternative - PMC Supplementary:**
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6803624/
   - PDF available: `/articles/instance/6803624/bin/41598_2019_51085_MOESM1_ESM.pdf`
   - Check for Excel supplements

**Expected Output:**
- File: `PXD011967_source_data.xlsx`
- Rows: 4,380 proteins × 58 samples
- Save to: `external_datasets/PXD011967/raw_data.xlsx`

---

### 2. PXD015982 - Richter 2021 Skin Matrisome

**Publication:**
- Journal: Matrix Biology Plus
- PMID: 33543036
- DOI: 10.1016/j.mbplus.2020.100039

**Data Sources:**
1. **PubMed Central:**
   - Europe PMC: https://europepmc.org/article/MED/33543036
   - Look for "Supplementary Table S1" (matrisome quantification)

2. **ScienceDirect:**
   - Direct URL: https://www.sciencedirect.com/science/article/pii/S2590028520300195
   - Navigate to "Supplementary Data" section
   - Download Excel files

3. **ProteomeXchange:**
   - PXD015982 PRIDE repository
   - FTP: ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2020/12/PXD015982/

**Expected Output:**
- File: `PXD015982_matrisome.xlsx`
- Rows: 229 matrisome proteins × 6 samples
- Save to: `external_datasets/PXD015982/raw_data.xlsx`

---

## MEDIUM PRIORITY (Optional)

### 3. PXD007048 - Bone Marrow Niche

**Access:**
- PRIDE: https://www.ebi.ac.uk/pride/archive/projects/PXD007048
- FTP: ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2017/09/PXD007048/
- **Note:** RAW files only, requires processing pipeline

**Alternative:** Contact authors for processed data

---

### 4. MSV000082958 - Lung Fibrosis

**Access:**
- MassIVE: https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=MSV000082958
- FTP: ftp://massive.ucsd.edu/MSV000082958/
- **Note:** May require MassIVE account

---

### 5. MSV000096508 - Brain Cognitive Aging (Mouse)

**Access:**
- MassIVE: https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=MSV000096508
- FTP: ftp://massive.ucsd.edu/MSV000096508/
- **Note:** Cross-species validation (mouse data)

---

### 6. PXD016440 - Skin Dermis Developmental

**Access:**
- PRIDE: https://www.ebi.ac.uk/pride/archive/projects/PXD016440
- FTP: ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2020/03/PXD016440/
- **Note:** Developmental (not aging), but useful baseline

---

## RECOMMENDED IMMEDIATE ACTIONS

### Option A: Manual Download (FASTEST - 30 minutes)

1. **Open browser**
2. **PXD011967 (eLife):**
   - Go to https://elifesciences.org/articles/49874
   - Click "Figures and data" tab
   - Download "Figure 1—source data 1.xlsx"
   - Save to `external_datasets/PXD011967/raw_data.xlsx`

3. **PXD015982 (ScienceDirect):**
   - Go to https://www.sciencedirect.com/science/article/pii/S2590028520300195
   - Click "Supplementary data" section
   - Download supplementary Excel files
   - Save to `external_datasets/PXD015982/raw_data.xlsx`

4. **Run validation:**
   ```bash
   python validate_downloads.py
   ```

### Option B: Contact Authors (RELIABLE - 2-3 days)

**For PXD011967:**
- Email: Luigi Ferrucci (ferruccilu@grc.nia.nih.gov)
- Request: "Processed proteomic abundance matrix from PXD011967 (eLife 2019)"

**For PXD015982:**
- Email: Corresponding authors from paper
- Request: "Supplementary Table S1 - Matrisome quantification data"

### Option C: Use ProteomeXchange RAW Files (SLOW - requires processing)

1. Install proteomics pipeline (MaxQuant, Fragpipe)
2. Download RAW MS files from PRIDE FTP
3. Process through LFQ pipeline (1-2 days compute time)
4. Generate protein abundance matrices

**NOT RECOMMENDED** - Too time-consuming for validation

---

## VALIDATION CHECKLIST

After downloading, verify each file:

```bash
python validate_downloads.py --dataset PXD011967
```

**Requirements:**
- ✅ File exists and not empty
- ✅ Valid Excel/CSV format
- ✅ ≥10 rows (proteins)
- ✅ ≥3 columns (samples)
- ✅ Contains protein identifiers (UniProt, Gene Symbol)
- ✅ Contains abundance values (LFQ, TMT, SILAC)

---

## SUCCESS CRITERIA

**Minimum for H16 Unblocking:**
- ✅ PXD011967 downloaded (CRITICAL - muscle aging, n=58)
- ✅ PXD015982 downloaded (CRITICAL - skin matrisome, n=6)
- 📊 2/6 datasets = 33% → **SUFFICIENT** for initial external validation

**Ideal:**
- 5/6 datasets (83%) → Robust meta-analysis

---

## FALLBACK: Simulated External Data

If downloads fail, we can:
1. Generate synthetic external data based on H16 framework
2. Use internal cross-validation (split dataset)
3. Report limitation: "External validation pending data access"

**NOT RECOMMENDED** - Would not constitute real validation

---

## NEXT STEPS AFTER DOWNLOAD

1. Preprocess datasets (UniProt mapping, z-score calculation)
2. Load into H16 validation framework
3. Run transfer learning (H08, H06, H03)
4. Generate external validation results
5. Create final H21 report

**Estimated time:** 4-6 hours after data acquisition

---

**Last Updated:** 2025-10-21
**Status:** Awaiting manual download completion
**Agent:** claude_code
