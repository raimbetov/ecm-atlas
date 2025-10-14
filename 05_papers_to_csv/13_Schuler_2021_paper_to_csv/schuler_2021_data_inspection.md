# Schuler et al. 2021 - Data File Inspection Report
Generated: 2025-10-15
Location: /home/raimbetov/GitHub/ecm-atlas/data_raw/Schuler et al. - 2021
================================================================================


## FILE: mmc2.xls
Size: 4.42 MB

**Sheet names:** ['Column Key', '1_Proteome Old vs. Young', '2_Proteome Ger. Vs. Young', '3_Overview dataset compared', '4_Comp. transcriptome-proteome', '5_IPA Proteome Old vs. Young', '6_UPSTREAM REG. Old vs. Young']
**Number of sheets:** 7

### Sheet: Column Key

**Shape:** 10 rows × 3 columns

**Columns (3 total):**
  1. Tab
  2. Column Header
  3. Meaning

**First 3 rows preview:**
```
   Tab               Column Header                             Meaning
0  1,2  Comparison (group1/group2)  indicates the compared ages groups
1  1,2                       Group                  Uniprot  Accession
2  1,2                       Genes                         Gene symbol
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ❌ Niche/MuSC
  ❌ Age values

### Sheet: 1_Proteome Old vs. Young

**Shape:** 10 rows × 11 columns

**Columns (11 total):**
  1. # Output table from Spectronaut filtered for the comparison of Old (18m) vs. Young (y) muscle stem cells. Related to Figure 1 and S1.
  2. Unnamed: 1
  3. Unnamed: 2
  4. Unnamed: 3
  5. Unnamed: 4
  6. Unnamed: 5
  7. Unnamed: 6
  8. Unnamed: 7
  9. Unnamed: 8
  10. Unnamed: 9
  11. Unnamed: 10

**First 3 rows preview:**
```
  # Output table from Spectronaut filtered for the comparison of Old (18m) vs. Young (y) muscle stem cells. Related to Figure 1 and S1. Unnamed: 1 Unnamed: 2           Unnamed: 3    Unnamed: 4               Unnamed: 5      Unnamed: 6               Unnamed: 7 Unnamed: 8 Unnamed: 9 Unnamed: 10
0                                                                                                                                   NaN        NaN        NaN                  NaN           NaN                      NaN             NaN                      NaN        NaN        NaN         NaN
1                                                                                                            Comparison (group1/group2)      Group      Genes  ProteinDescriptions  ProteinNames  # Unique Total Peptides  AVG Log2 Ratio  Absolute AVG Log2 Ratio   % Change      Ratio      Qvalue
2                                                                                                                                 o / y     A2ASS6        Ttn                Titin   TITIN_MOUSE                     1911        0.038614                 0.038614   2.712689   1.027127         0.0
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ✅ Young/Old
  ✅ Niche/MuSC
  ❌ Age values

### Sheet: 2_Proteome Ger. Vs. Young

**Shape:** 10 rows × 11 columns

**Columns (11 total):**
  1. # Output table from Spectronaut filtered for the comparison of Geriatric (26m) vs. Young (y) muscle stem cells. Related to Figure 1 and S1.
  2. Unnamed: 1
  3. Unnamed: 2
  4. Unnamed: 3
  5. Unnamed: 4
  6. Unnamed: 5
  7. Unnamed: 6
  8. Unnamed: 7
  9. Unnamed: 8
  10. Unnamed: 9
  11. Unnamed: 10

**First 3 rows preview:**
```
  # Output table from Spectronaut filtered for the comparison of Geriatric (26m) vs. Young (y) muscle stem cells. Related to Figure 1 and S1. Unnamed: 1 Unnamed: 2           Unnamed: 3    Unnamed: 4               Unnamed: 5      Unnamed: 6               Unnamed: 7  Unnamed: 8 Unnamed: 9 Unnamed: 10
0                                                                                                                                         NaN        NaN        NaN                  NaN           NaN                      NaN             NaN                      NaN         NaN        NaN         NaN
1                                                                                                                  Comparison (group1/group2)      Group      Genes  ProteinDescriptions  ProteinNames  # Unique Total Peptides  AVG Log2 Ratio  Absolute AVG Log2 Ratio    % Change      Ratio      Qvalue
2                                                                                                                                       g / y     O08638      Myh11            Myosin-11   MYH11_MOUSE                      193        1.063398                 1.063398  108.984833   2.089848         0.0
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ✅ Young/Old
  ✅ Niche/MuSC
  ❌ Age values

... and 4 more sheets

--------------------------------------------------------------------------------

## FILE: mmc3.xls
Size: 5.72 MB

**Sheet names:** ['Column Key', '1_TA o vs y', '2_TA g vs y', '3_Soleus o vs y', '4_Soleus g vs y', '5_EDL o vs y', '6_EDL g vs y', '7_Gastroc o vs y', '8_Gastroc g vs y', '9_GeneSetEnrichmentAnalysis']
**Number of sheets:** 10

### Sheet: Column Key

**Shape:** 10 rows × 3 columns

**Columns (3 total):**
  1. Tab
  2. Column Header
  3. Meaning

**First 3 rows preview:**
```
   Tab Column Header                                  Meaning
0  1-8         logFC                       log2 (fold change)
1  1-8       AveExpr  log2 (average protein abundance score )
2  1-8             t  moderated t-statistic computed by limma
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ❌ Niche/MuSC
  ❌ Age values

### Sheet: 1_TA o vs y

**Shape:** 10 rows × 10 columns

**Columns (10 total):**
  1. # Output table from limma for TMT proteomic data for the comparison of Old vs. Young TA muscle. Related to Figure 2, S3, S4 and S5.
  2. Unnamed: 1
  3. Unnamed: 2
  4. Unnamed: 3
  5. Unnamed: 4
  6. Unnamed: 5
  7. Unnamed: 6
  8. Unnamed: 7
  9. Unnamed: 8
  10. Unnamed: 9

**First 3 rows preview:**
```
  # Output table from limma for TMT proteomic data for the comparison of Old vs. Young TA muscle. Related to Figure 2, S3, S4 and S5. Unnamed: 1 Unnamed: 2 Unnamed: 3 Unnamed: 4 Unnamed: 5                                                Unnamed: 6  Unnamed: 7                                                                                                                                                                                                                                                                                                                                                                                          Unnamed: 8 Unnamed: 9
0                                                                                                                                 NaN        NaN        NaN        NaN        NaN        NaN                                                       NaN         NaN                                                                                                                                                                                                                                                                                                                                                                                                 NaN        NaN
1                                                                                                                               logFC    AveExpr          t    P.Value  adj.P.Val          B                                               description  short.name                                                                                                                                                                                                                                                                                                                                                                                            sub.cell         ID
2                                                                                                                            1.948299  14.676857   31.57163        0.0        0.0  18.936225  Cytoplasmic protein NCK1 (NCK adaptor protein 1) (Nck-1)  NCK1_MOUSE  SUBCELLULAR LOCATION: Cytoplasm {ECO:0000250}. Endoplasmic reticulum {ECO:0000250}. Nucleus {ECO:0000250}. Note=Mostly cytoplasmic, but shuttles between the cytoplasm and the nucleus. Import into the nucleus requires interaction with SOCS7. Predominantly nuclear following genotoxic stresses, such as UV irradiation, hydroxyurea or mitomycin C treatments (By similarity). {ECO:0000250}.     Q99M51
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ✅ Young/Old
  ✅ Niche/MuSC
  ❌ Age values

### Sheet: 2_TA g vs y

**Shape:** 10 rows × 10 columns

**Columns (10 total):**
  1. # Output table from limma for TMT proteomic data for the comparison of Geriatric vs. Young TA muscle. Related to Figure 2, S3, S4 and S5.
  2. Unnamed: 1
  3. Unnamed: 2
  4. Unnamed: 3
  5. Unnamed: 4
  6. Unnamed: 5
  7. Unnamed: 6
  8. Unnamed: 7
  9. Unnamed: 8
  10. Unnamed: 9

**First 3 rows preview:**
```
  # Output table from limma for TMT proteomic data for the comparison of Geriatric vs. Young TA muscle. Related to Figure 2, S3, S4 and S5. Unnamed: 1 Unnamed: 2 Unnamed: 3 Unnamed: 4 Unnamed: 5                                           Unnamed: 6  Unnamed: 7                                      Unnamed: 8 Unnamed: 9
0                                                                                                                                       NaN        NaN        NaN        NaN        NaN        NaN                                                  NaN         NaN                                             NaN        NaN
1                                                                                                                                     logFC    AveExpr          t    P.Value  adj.P.Val          B                                          description  short.name                                        sub.cell         ID
2                                                                                                                                 -1.696983  14.631915 -15.325969        0.0   0.000005  12.199066  Actin, alpha cardiac muscle 1 (Alpha-cardiac actin)  ACTC_MOUSE  SUBCELLULAR LOCATION: Cytoplasm, cytoskeleton.     P68033
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ✅ Young/Old
  ✅ Niche/MuSC
  ❌ Age values

... and 7 more sheets

--------------------------------------------------------------------------------

## FILE: mmc4.xls
Size: 0.27 MB

**Sheet names:** ['description', 'Column Key', '1_S O vs. Y', '2_G O vs. Y', '3_TA O vs. Y', '4_EDL O vs. Y']
**Number of sheets:** 6

### Sheet: description

**Shape:** 2 rows × 1 columns

**Columns (1 total):**
  1. Analysis of compositional changes of the extracellular matrix in aging skeletal muscles was performed as described in: 

**First 3 rows preview:**
```
  Analysis of compositional changes of the extracellular matrix in aging skeletal muscles was performed as described in: 
0                                                                     https://www.embopress.org/doi/10.15252/msb.20178131
1                                                                                           Related to Figure 3D and S6A.
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ✅ Niche/MuSC
  ❌ Age values

### Sheet: Column Key

**Shape:** 10 rows × 3 columns

**Columns (3 total):**
  1. Tab
  2. Column Header
  3. Meaning

**First 3 rows preview:**
```
   Tab      Column Header                                                      Meaning
0  1-4            uniprot                           indicates the compared ages groups
1  1-4  sample1_abundance             Average protein abundance in condition 1 (young)
2  1-4  sample2_abundance  Average protein abundance in condition 2 (old or geriatric)
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ❌ Niche/MuSC
  ❌ Age values

### Sheet: 1_S O vs. Y

**Shape:** 10 rows × 10 columns

**Columns (10 total):**
  1. uniprot
  2. sample1_abundance
  3. sample2_abundance
  4. short.name
  5. accession
  6. compartment
  7. residuals
  8. cnv_value
  9. cnv_fdrtool.pval
  10. cnv_fdrtool.qval

**First 3 rows preview:**
```
  uniprot  sample1_abundance  sample2_abundance short.name accession    compartment  residuals  cnv_value  cnv_fdrtool.pval  cnv_fdrtool.qval
0  Q64739          15.023363          16.642141    Col11a2    Q64739  Extracellular   1.699922   4.460667      1.393730e-11      4.473352e-09
1  Q8R1Q3          13.839816          15.308943    Angptl7    Q8R1Q3  Extracellular   1.463164   3.847771      5.545060e-09      8.898786e-07
2  Q61646          14.301732          15.627979         Hp    Q61646  Extracellular   1.354280   3.556328      7.110647e-08      7.607507e-06
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ❌ Niche/MuSC
  ❌ Age values

... and 3 more sheets

--------------------------------------------------------------------------------

## FILE: mmc5.xlsx
Size: 0.12 MB

**Sheet names:** ['Tab1']
**Number of sheets:** 1

### Sheet: Tab1

**Shape:** 10 rows × 25 columns

**Columns (25 total):**
  1. # List of ligands affected by aging in old or geriatric skeletal muscles and their receptors (according to Ramewloski et al. 2015) expressed by muscle stem cells. Related to Figure 4 and S7A.
  2. Unnamed: 1
  3. Unnamed: 2
  4. Unnamed: 3
  5. Unnamed: 4
  6. Unnamed: 5
  7. Unnamed: 6
  8. Unnamed: 7
  9. Unnamed: 8
  10. Unnamed: 9
  11. Unnamed: 10
  12. Unnamed: 11
  13. Unnamed: 12
  14. Unnamed: 13
  15. Unnamed: 14
  16. Unnamed: 15
  17. Unnamed: 16
  18. Unnamed: 17
  19. Unnamed: 18
  20. Unnamed: 19
  ... and 5 more columns

**First 3 rows preview:**
```
  # List of ligands affected by aging in old or geriatric skeletal muscles and their receptors (according to Ramewloski et al. 2015) expressed by muscle stem cells. Related to Figure 4 and S7A.             Unnamed: 1                                  Unnamed: 2    Unnamed: 3        Unnamed: 4    Unnamed: 5        Unnamed: 6   Unnamed: 7       Unnamed: 8   Unnamed: 9      Unnamed: 10  Unnamed: 11      Unnamed: 12  Unnamed: 13      Unnamed: 14    Unnamed: 15        Unnamed: 16    Unnamed: 17        Unnamed: 18              Unnamed: 19                                          Unnamed: 20              Unnamed: 21      Unnamed: 22              Unnamed: 23      Unnamed: 24
0                                                                                                                                                                                             NaN                    NaN                                         NaN           NaN               NaN           NaN               NaN          NaN              NaN          NaN              NaN          NaN              NaN          NaN              NaN            NaN                NaN            NaN                NaN                      NaN                                                  NaN                      NaN              NaN                      NaN              NaN
1                                                                                                                                                                                       Pair.Name  Ligand.ApprovedSymbol                                 Ligand.Name  TA.OvY.logFC  TA.OvY.adj.P.Val  TA.GvY.logFC  TA.GvY.adj.P.Val  S.OvY.logFC  S.OvY.adj.P.Val  S.GvY.logFC  S.GvY.adj.P.Val  G.OvY.logFC  G.OvY.adj.P.Val  G.GvY.logFC  G.GvY.adj.P.Val  EDL.OvY.logFC  EDL.OvY.adj.P.Val  EDL.GvY.logFC  EDL.GvY.adj.P.Val  Receptor.ApprovedSymbol                                        Receptor.Name  MuSC.GvY.AVG.Log2.Ratio  MuSC.GvY.Qvalue  MuSC.OvY.AVG.Log2.Ratio  MuSC.OvY.Qvalue
2                                                                                                                                                                                     CALM1_ABCA1                  CALM1  calmodulin 1 (phosphorylase kinase, delta)      0.330996          0.095977      0.239221          0.364906    -0.351022         0.146844    -0.114253         0.711118     0.033521         0.822593    -0.049157          0.68173       0.402553           0.002088       0.057106           0.773733                    ABCA1  ATP-binding cassette, sub-family A (ABC1), member 1                      NaN              NaN                      NaN              NaN
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ✅ Young/Old
  ✅ Niche/MuSC
  ❌ Age values

--------------------------------------------------------------------------------

## FILE: mmc6.xlsx
Size: 0.82 MB

**Sheet names:** ['Column Key', 'CTX TA']
**Number of sheets:** 2

### Sheet: Column Key

**Shape:** 10 rows × 2 columns

**Columns (2 total):**
  1. Column Header
  2. Meaning

**First 3 rows preview:**
```
  Column Header                                  Meaning
0         logFC                       log2 (fold change)
1       AveExpr  log2 (average protein abundance score )
2             t  moderated t-statistic computed by limma
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ❌ Niche/MuSC
  ❌ Age values

### Sheet: CTX TA

**Shape:** 10 rows × 10 columns

**Columns (10 total):**
  1. # Output table from limma for TMT proteomic data for the comparison of injured  vs. not-injured TA muscle at 7 days post-injury. Related to Figure 5.
  2. Unnamed: 1
  3. Unnamed: 2
  4. Unnamed: 3
  5. Unnamed: 4
  6. Unnamed: 5
  7. Unnamed: 6
  8. Unnamed: 7
  9. Unnamed: 8
  10. Unnamed: 9

**First 3 rows preview:**
```
  # Output table from limma for TMT proteomic data for the comparison of injured  vs. not-injured TA muscle at 7 days post-injury. Related to Figure 5. Unnamed: 1 Unnamed: 2 Unnamed: 3 Unnamed: 4 Unnamed: 5                                                                                          Unnamed: 6   Unnamed: 7                                                                 Unnamed: 8 Unnamed: 9
0                                                                                                                                                   NaN        NaN        NaN        NaN        NaN        NaN                                                                                                 NaN          NaN                                                                        NaN        NaN
1                                                                                                                                                 logFC    AveExpr          t    P.Value  adj.P.Val          B                                                                                         description   short.name                                                                   sub.cell         ID
2                                                                                                                                              4.099566  14.917182  26.000719        0.0        0.0  15.991322  CD180 antigen (Lymphocyte antigen 78) (Ly-78) (Radioprotective 105 kDa protein) (CD antigen CD180)  CD180_MOUSE  SUBCELLULAR LOCATION: Cell membrane; Single-pass type I membrane protein.     Q62192
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ✅ Niche/MuSC
  ❌ Age values

--------------------------------------------------------------------------------

## FILE: mmc7.xlsx
Size: 0.30 MB

**Sheet names:** ['Column Key', 'decell TA o vs. y']
**Number of sheets:** 2

### Sheet: Column Key

**Shape:** 10 rows × 2 columns

**Columns (2 total):**
  1. Column Header
  2. Meaning

**First 3 rows preview:**
```
  Column Header                                  Meaning
0         logFC                       log2 (fold change)
1       AveExpr  log2 (average protein abundance score )
2             t  moderated t-statistic computed by limma
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ❌ Niche/MuSC
  ❌ Age values

### Sheet: decell TA o vs. y

**Shape:** 10 rows × 10 columns

**Columns (10 total):**
  1. # Output table from limma for TMT proteomic data for the comparison of Old  vs. Young TA muscle after de-cellularization. Related to Figure S9E.
  2. Unnamed: 1
  3. Unnamed: 2
  4. Unnamed: 3
  5. Unnamed: 4
  6. Unnamed: 5
  7. Unnamed: 6
  8. Unnamed: 7
  9. Unnamed: 8
  10. Unnamed: 9

**First 3 rows preview:**
```
  # Output table from limma for TMT proteomic data for the comparison of Old  vs. Young TA muscle after de-cellularization. Related to Figure S9E. Unnamed: 1 Unnamed: 2 Unnamed: 3 Unnamed: 4 Unnamed: 5                                                                                           Unnamed: 6   Unnamed: 7 Unnamed: 8 Unnamed: 9
0                                                                                                                                              NaN        NaN        NaN        NaN        NaN        NaN                                                                                                  NaN          NaN        NaN        NaN
1                                                                                                                                            logFC    AveExpr          t    P.Value  adj.P.Val          B                                                                                          description   short.name   sub.cell         ID
2                                                                                                                                         1.534878  16.065255  15.986527        0.0   0.000007  11.248802  Ribose-phosphate pyrophosphokinase 1 (EC 2.7.6.1) (Phosphoribosyl pyrophosphate synthase I) (PRS-I)  PRPS1_MOUSE        NaN     Q9D7G0
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ✅ Young/Old
  ✅ Niche/MuSC
  ❌ Age values

--------------------------------------------------------------------------------

## FILE: mmc8.xls
Size: 1.30 MB

**Sheet names:** ['Column Key', 'Phosphoproteome C2C12 ± Smoc2']
**Number of sheets:** 2

### Sheet: Column Key

**Shape:** 10 rows × 2 columns

**Columns (2 total):**
  1. Column Header
  2. Meaning

**First 3 rows preview:**
```
  Column Header                                  Meaning
0         logFC                       log2 (fold change)
1       AveExpr  log2 (average protein abundance score )
2             t  moderated t-statistic computed by limma
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ❌ Niche/MuSC
  ❌ Age values

### Sheet: Phosphoproteome C2C12 ± Smoc2

**Shape:** 10 rows × 11 columns

**Columns (11 total):**
  1. # Output table from limma for phosphosite abundance comparison between C2C12 cells treate with Smoc2 or vehicle control. Related to Figure 6 and S10.
  2. Unnamed: 1
  3. Unnamed: 2
  4. Unnamed: 3
  5. Unnamed: 4
  6. Unnamed: 5
  7. Unnamed: 6
  8. Unnamed: 7
  9. Unnamed: 8
  10. Unnamed: 9
  11. Unnamed: 10

**First 3 rows preview:**
```
  # Output table from limma for phosphosite abundance comparison between C2C12 cells treate with Smoc2 or vehicle control. Related to Figure 6 and S10. Unnamed: 1 Unnamed: 2 Unnamed: 3 Unnamed: 4    Unnamed: 5 Unnamed: 6 Unnamed: 7                                                                                                                                                                                Unnamed: 8   Unnamed: 9 Unnamed: 10
0                                                                                                                                                   NaN        NaN        NaN        NaN        NaN           NaN        NaN        NaN                                                                                                                                                                                       NaN          NaN         NaN
1                                                                                                                                                 logFC    AveExpr          t    P.Value          B  fdrtool.qval         ID   Location                                                                                                                                                                               description   short.name   gene.name
2                                                                                                                                              0.940649  16.163491   8.315444    0.00062  -3.872189           0.0     Q03145       S893  Ephrin type-A receptor 2 (EC 2.7.10.1) (Epithelial cell kinase) (Tyrosine-protein kinase receptor ECK) (Tyrosine-protein kinase receptor MPK-5) (Tyrosine-protein kinase receptor SEK-2)  EPHA2_MOUSE       Epha2
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ❌ Niche/MuSC
  ❌ Age values

--------------------------------------------------------------------------------

## FILE: mmc9.xlsx
Size: 0.01 MB

**Sheet names:** ['Sheet1']
**Number of sheets:** 1

### Sheet: Sheet1

**Shape:** 10 rows × 3 columns

**Columns (3 total):**
  1. Name
  2. Forward sequence
  3. Reverse sequence

**First 3 rows preview:**
```
    Name       Forward sequence       Reverse sequence
0  Axin2  TGACTCTCCTTCCAGATCCCA    TGCCCACACTAGGCTGACA
1   Bcl2   GTCGCTACCGTCGTGACTTC   CAGACATGCACCTACCCAGC
2  Ccnd1  GCGTACCCTGACACCAATCTC  CTCCTCTTCGCACTTCTGCTC
```

**Proteomics Indicators:**
  ❌ Protein ID
  ❌ Gene Symbol
  ❌ LFQ/Intensity
  ❌ Young/Old
  ❌ Niche/MuSC
  ❌ Age values

--------------------------------------------------------------------------------
