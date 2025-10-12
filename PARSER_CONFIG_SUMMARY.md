# Parser Configuration Quick Reference

## Dataset Summary Table

| # | Study | File | Rows | Skip | Protein ID | Gene | Protein Name | Abundance Columns | Age Encoding | Special Notes |
|---|-------|------|------|------|------------|------|--------------|-------------------|--------------|---------------|
| 1 | **Ariosa-Morejon 2021** | `elife-66635-fig2-data1-v1.xlsx` | 173-712 | 0 | `Majority protein IDs` | `Gene names` | `Protein names` | `iBAQ L A1-4`, `iBAQ H A1-4`, `Ratio H/L A1-4`, `Ratio H/L B1-4` | H/L ratio (H=old, L=young) | Multi-sheet (Plasma/Cartilage/Bone/Skin) |
| 2 | **Chmelova 2023** | `Data Sheet 1.XLSX` | 17 | 0 | N/A | Column headers | N/A | All columns (3830 genes) | Sample names: `X3m_*` vs `X18m_*` | **TRANSPOSED!** Samples=rows, Genes=cols |
| 3 | **Li 2021 dermis** | `Table 2.xlsx` | 264 | 3 | `Protein ID` | `Gene symbol` | N/A | Cols 4-18: `Toddler-Sample1-2`, `Teenager-Sample1-3`, `Adult-Sample1-3`, `Elderly-Sample1-5` | Sample names | Log2 FOT values, skip 3 for headers |
| 4 | **Li 2021 pancreas** | `41467_2021_21261_MOESM6_ESM.xlsx` | 2064 | 2 | `Accession` | N/A | `Description` | `F_7-12`, `J_8,22,46,59,67`, `Y_12,31,51,57,61,64`, `O_14,20,48,52,54,68` | Prefix: F=Fetal, J=Juvenile(wks), Y=Young(yrs), O=Old(yrs) | Age value in suffix |
| 5 | **Lofaro 2021** | PDFs only | - | - | - | - | - | - | - | **SKIP - No Excel** |
| 6 | **McCabe 2020** | DOCX only | - | - | - | - | - | - | - | **SKIP - No Excel** |
| 7 | **Ouni 2022** | `Supp Table 2.xlsx` | Various | 0 | N/A | N/A | N/A | N/A | N/A | **SKIP - Literature mining data** |
| 8 | **Randles 2021** | `ASN.2020101442-File027.xlsx` | 2610 | 1 | `Accession` | `Gene name` | `Description` | `G15,29,37,61,67,69`, `T15,29,37,61,67,69` | Age in col name (15-69 yrs); G=Glomerular, T=Tubular | Two tissue types |
| 9 | **Tam 2020** | `elife-64940-supp1-v3.xlsx` | 3157 | 1 | `Majority protein IDs` | `Gene names` | `Protein names` | 66 cols: `LFQ intensity {disc}_{age} {region} {tissue}` | "old" vs "Young" in col names | Complex: disc(L3/4,L4/5,L5/S1) × region(L/A/P/R) × tissue(OAF/IAF/NP) |
| 10 | **Tsumagari 2023** | `41598_2023_45570_MOESM3_ESM.xlsx` | 6821 | 0 | `UniProt accession` | `Gene name` | N/A | `Cx_3mo_1-6`, `Cx_15mo_1-6`, `Cx_24mo_1-6` | Age in col: 3mo/15mo/24mo | Protein info at end (cols 18-31) |

## Parser Implementation Priority

### ✅ Ready to Parse (7 datasets)

1. **Ariosa-Morejon 2021** - Standard multi-sheet, ratio-based
2. **Li 2021 dermis** - Simple after header skip
3. **Li 2021 pancreas** - Age prefix parsing
4. **Randles 2021** - Standard with dual tissue
5. **Tam 2020** - Complex but systematic column structure
6. **Tsumagari 2023** - Standard format
7. **Chmelova 2023** - Requires transpose operation

### ❌ Skip (3 datasets)

8. **Lofaro 2021** - PDF only
9. **McCabe 2020** - DOCX only
10. **Ouni 2022** - Not proteomics data

## Key Parsing Patterns

### Age Group Extraction Methods

1. **Column name patterns:**
   - Direct age in name: `G15`, `T29` → age = 15, 29
   - Age group words: `Toddler-Sample1`, `Elderly-Sample3`
   - Age with unit: `Cx_3mo_1`, `Cx_15mo_1` → age = 3, 15 months
   - Prefix codes: `F_7`, `J_22`, `Y_31`, `O_54` → F=Fetal, J=Juvenile, Y=Young, O=Old

2. **Sample name parsing:**
   - `X3m_ctrl_A` → 3 months
   - `X18m_MCAO_7d_B` → 18 months
   - `LFQ intensity L3/4 old L OAF` → old age group

3. **Ratio-based:**
   - `Ratio H/L` → Heavy/Light isotope (typically H=old, L=young)

### Column Identification

- **Protein IDs:** Look for: `Protein ID`, `Accession`, `Majority protein IDs`, `UniProt accession`
- **Genes:** Look for: `Gene name`, `Gene names`, `Gene symbol`
- **Protein names:** Look for: `Protein names`, `Description`
- **Abundance:** Look for: `LFQ intensity`, `iBAQ`, `Ratio`, `Intensity`, sample/age patterns

### Special Handling

1. **Transpose required:** Chmelova 2023 (samples as rows)
2. **Multi-sheet:** Ariosa-Morejon 2021 (4 tissue sheets)
3. **Multi-tissue:** Randles 2021 (Glomerular vs Tubular)
4. **Skip rows:** Li dermis (3), Li pancreas (2), Randles (1), Tam (1)
5. **Complex columns:** Tam 2020 (hierarchical: disc × age × region × tissue)

## Recommended Configuration Approach

```python
DATASET_CONFIGS = {
    'ariosa-morejon_2021': {
        'file': 'elife-66635-fig2-data1-v1.xlsx',
        'sheets': ['Plasma', 'Cartilage', 'Bone', 'Skin'],
        'skiprows': 0,
        'protein_id_col': 'Majority protein IDs',
        'gene_col': 'Gene names',
        'protein_name_col': 'Protein names',
        'abundance_pattern': r'(iBAQ|Ratio)',
        'age_extraction': 'ratio_based',  # H=old, L=young
    },
    'chmelova_2023': {
        'file': 'Data Sheet 1.XLSX',
        'transpose': True,
        'skiprows': 0,
        'gene_col': 'row_index_after_transpose',
        'abundance_pattern': r'(X3m_|X18m_)',
        'age_extraction': 'sample_name',  # Extract from X3m/X18m
        'filter_samples': 'ctrl',  # Only control samples
    },
    'li_2021_dermis': {
        'file': 'Table 2.xlsx',
        'sheet': 'Table S2',
        'skiprows': 3,
        'protein_id_col': 'Protein ID',
        'gene_col': 'Gene symbol',
        'abundance_pattern': r'Sample\d+$',
        'age_extraction': 'column_prefix',  # Toddler/Teenager/Adult/Elderly
    },
    # ... etc
}
```
