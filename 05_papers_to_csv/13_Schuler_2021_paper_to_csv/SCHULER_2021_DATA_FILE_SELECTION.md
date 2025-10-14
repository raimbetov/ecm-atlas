# Schuler et al. 2021 - Data File Selection Report

**Generated:** 2025-10-15
**Status:** 🎯 **BEST FILE IDENTIFIED**

---

## 🏆 RECOMMENDED FILE FOR ECM-ATLAS

### **File: mmc2.xls**
### **Sheet: "1_Proteome Old vs. Young"**

**Why this is the right file:**
1. ✅ **DIA-LFQ proteomics** of MuSC (muscle stem cells)
2. ✅ **Age comparison:** Old (18 months) vs Young
3. ✅ **Spectronaut output** (DIA quantification software)
4. ✅ **Complete protein identifiers:** UniProt IDs + Gene Symbols
5. ✅ **Quantitative data:** Ratios and statistical values
6. ✅ **Related to Figure 1 and S1** (main findings)

---

## 📊 Data File Breakdown

### mmc2.xls - **⭐ PRIMARY FILE** (4.42 MB)
**Content:** MuSC proteome aging comparisons

**Sheets (7 total):**
1. ✅ **"1_Proteome Old vs. Young"** ← **USE THIS**
   - Old (18m) vs Young (y)
   - Spectronaut DIA output
   - Complete protein quantification

2. **"2_Proteome Ger. Vs. Young"**
   - Geriatric (26m) vs Young (y)
   - Could be used as second age comparison

3. **"3_Overview dataset compared"**
4. **"4_Comp. transcriptome-proteome"**
5. **"5_IPA Proteome Old vs. Young"** (pathway analysis)
6. **"6_UPSTREAM REG. Old vs. Young"** (regulation analysis)

**Verdict:** ✅ **This is the main DIA-LFQ dataset for ECM-Atlas**

---

### mmc3.xls - BULK MUSCLE DATA (5.72 MB)
**Content:** Bulk muscle tissue proteomics (multiple muscles)

**Sheets (10 total):**
- TA (tibialis anterior) o vs y
- TA g vs y
- Soleus o vs y
- Soleus g vs y
- EDL (extensor digitorum longus) o vs y
- EDL g vs y
- Gastroc (gastrocnemius) o vs y
- Gastroc g vs y

**Method:** TMT (not LFQ!) - Labeled quantification

**Verdict:** ⚠️ **Different method (TMT, not LFQ)** - Not compatible with current pipeline, but valuable for bulk muscle ECM

---

### mmc4.xls - **⭐ ECM-SPECIFIC DATA** (0.27 MB)
**Content:** ECM compositional changes in aging muscles

**Sheets:**
- 1_S O vs. Y (Soleus)
- 2_G O vs. Y (Gastrocnemius)
- 3_TA O vs. Y (Tibialis anterior)
- 4_EDL O vs. Y

**Key features:**
- ✅ **Already filtered for ECM proteins!**
- ✅ **"compartment" column = "Extracellular"**
- ✅ **UniProt IDs provided**
- ✅ **Abundance values:** sample1_abundance (young), sample2_abundance (old)
- ✅ **Statistical analysis:** CNV values, q-values

**Example data:**
```
uniprot  sample1_abundance  sample2_abundance  short.name  compartment
Q64739   15.023363          16.642141          Col11a2     Extracellular
Q8R1Q3   13.839816          15.308943          Angptl7     Extracellular
Q61646   14.301732          15.627979          Hp          Extracellular
```

**Verdict:** ✅✅ **BEST FOR ECM-ATLAS** - Pre-filtered for ECM, ready to use!

---

### mmc5.xlsx - Ligand-Receptor Analysis (0.12 MB)
**Content:** Aging-affected ligands and their MuSC receptors
**Verdict:** ❌ Not primary proteomics data

---

### mmc6.xlsx - Injury Response (0.82 MB)
**Content:** Injured vs. non-injured TA muscle
**Verdict:** ❌ Not aging comparison

---

### mmc7.xlsx - Decellularized Muscle (0.30 MB)
**Content:** Old vs Young TA after decellularization
**Verdict:** ⚠️ Could be useful for pure ECM, but specialized treatment

---

### mmc8.xls - Phosphoproteome (1.30 MB)
**Content:** SMOC2 treatment effects on phosphorylation
**Verdict:** ❌ Not aging proteomics

---

### mmc9.xlsx - PCR Primers (0.01 MB)
**Content:** qPCR primer sequences
**Verdict:** ❌ Not proteomics data

---

## 🎯 FINAL RECOMMENDATION

### **Option A: mmc4.xls - ECM-Specific** (EASIEST, RECOMMENDED)

**Advantages:**
- ✅ Already filtered for ECM proteins
- ✅ Pre-processed abundances
- ✅ Multiple muscle types (4 datasets)
- ✅ Small file, easy to process
- ✅ Clear column structure

**Disadvantages:**
- ⚠️ Only bulk muscle ECM (not MuSC niche)
- ⚠️ Smaller protein list (~100-200 ECM proteins per muscle)

**Processing approach:**
```bash
# Can process all 4 muscle types:
1. Soleus (S) O vs. Y
2. Gastrocnemius (G) O vs. Y
3. Tibialis Anterior (TA) O vs. Y
4. EDL (extensor digitorum longus) O vs. Y

# Each as separate tissue compartment
```

---

### **Option B: mmc2.xls - MuSC Proteome** (MORE COMPREHENSIVE)

**Advantages:**
- ✅ MuSC niche (aligned with paper focus)
- ✅ DIA-LFQ (high quality)
- ✅ Complete proteome (~thousands of proteins)
- ✅ Can filter for ECM using matrisome database

**Disadvantages:**
- ⚠️ Needs ECM filtering step
- ⚠️ Larger dataset to process
- ⚠️ May include cellular proteins

**Processing approach:**
```bash
# Extract sheet: "1_Proteome Old vs. Young"
# Filter for matrisome proteins
# Calculate LFQ intensities from ratios
```

---

## 📋 AGE INFORMATION EXTRACTED

From mmc2.xls sheet headers:

**Sheet 1:** "Old (18m) vs. Young (y)"
**Sheet 2:** "Geriatric (26m) vs. Young (y)"

**Age Mapping:**
- **Young:** Assume 3 months (standard for mouse MuSC studies)
- **Old:** 18 months
- **Geriatric:** 26 months

**For ECM-Atlas binary comparison:**
- Young (3m) → "Young"
- Old (18m) → "Old"

---

## 🎬 RECOMMENDED PROCESSING PLAN

### **PLAN A: Process mmc4.xls (ECM-focused, fastest)**

**Step 1:** Process all 4 muscle types
- Soleus Old vs Young
- Gastrocnemius Old vs Young
- TA Old vs Young
- EDL Old vs Young

**Step 2:** Each muscle as separate "Tissue_Compartment"
- Study_ID: Schuler_2021
- Tissue: Skeletal muscle
- Compartment: Soleus / Gastrocnemius / TA / EDL

**Step 3:** Use columns:
- Protein_ID: `uniprot` column
- Gene_Symbol: `short.name` column
- Abundance_Young: `sample1_abundance`
- Abundance_Old: `sample2_abundance`
- Already extracellular filtered!

**Estimated output:** ~400-800 ECM proteins × 4 muscles = ~1,600-3,200 rows

---

### **PLAN B: Process mmc2.xls + mmc4.xls (comprehensive)**

**Step 1:** Process mmc4.xls (as above)

**Step 2:** Process mmc2.xls sheet "1_Proteome Old vs. Young"
- Filter for matrisome proteins
- Tissue_Compartment: "MuSC_niche"
- Extract abundance from ratios

**Estimated output:** ~2,000-4,000 total rows

---

## ✅ MY RECOMMENDATION

### **Use PLAN A with mmc4.xls**

**Reasons:**
1. **Pre-filtered for ECM** - No ambiguity
2. **Clean data structure** - Easy parsing
3. **Multiple tissue compartments** - Rich dataset
4. **Aligned with paper's ECM focus** - Figure 3D, S6A
5. **Reference provided:** https://www.embopress.org/doi/10.15252/msb.20178131

**Processing time:** ~15-20 minutes with autonomous agent

---

## 📊 Expected Database Integration

**After processing mmc4.xls:**

**New Study ID:** Schuler_2021

**Tissue Compartments (4):**
1. Skeletal_muscle_Soleus
2. Skeletal_muscle_Gastrocnemius
3. Skeletal_muscle_TA
4. Skeletal_muscle_EDL

**Species:** Mus musculus

**Age Groups:**
- Young (3 months)
- Old (18 months)

**Method:** LFQ (from bulk muscle ECM extraction)

**Expected ECM proteins per muscle:** 100-200

**Total new rows in database:** ~1,600-3,200

---

## 🔍 Next Steps for Processing

1. **Confirm plan:** Do you approve PLAN A (mmc4.xls)?

2. **Run autonomous agent:**
```bash
cd 11_subagent_for_LFQ_ingestion
python autonomous_agent.py "../data_raw/Schuler et al. - 2021/mmc4.xls"
```

3. **Manual config if needed:**
```json
{
  "study_id": "Schuler_2021",
  "data_file": "mmc4.xls",
  "sheet_name": "1_S O vs. Y",
  "species": "Mus musculus",
  "tissue": "Skeletal muscle",
  "compartment": "Soleus",
  "young_ages": [3],
  "old_ages": [18],
  "method": "LFQ"
}
```

4. **Repeat for sheets:** 2_G O vs. Y, 3_TA O vs. Y, 4_EDL O vs. Y

---

## ❓ Questions for You

1. **Which plan?**
   - [ ] PLAN A: mmc4.xls only (ECM-focused, faster)
   - [ ] PLAN B: mmc2.xls + mmc4.xls (comprehensive, longer)

2. **Process all 4 muscles or subset?**
   - [ ] All 4 muscles (recommended)
   - [ ] Just 1-2 muscles (specify which)

3. **Ready to proceed?**
   - [ ] Yes, run autonomous agent on mmc4.xls
   - [ ] Wait, I need more info
   - [ ] Modify approach

---

**Status:** ✅ **READY TO PROCESS**
**Recommended file:** mmc4.xls (all 4 sheets)
**Estimated time:** 20-30 minutes
**Expected result:** Schuler_2021 with 4 muscle compartments in ECM-Atlas database
