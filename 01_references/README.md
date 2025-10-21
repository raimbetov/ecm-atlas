# Matrisome Reference Lists

This directory contains species-specific matrisome reference lists used for protein annotation in the ECM Atlas project.

## Files

1. [Delocalized entropy aging theorem](./Delocalized%20entropy%20aging%20theorem.pdf)
https://chatgpt.com/c/68ed24b6-d04c-8326-b462-5933fd725398

### Human Matrisome (Homo sapiens)
- **File:** `human_matrisome_v2.csv`
- **Source:** [Google Sheets](https://docs.google.com/spreadsheets/d/1GwwV3pFvsp7DKBbCgr8kLpf8Eh_xV8ks/edit)
- **Version:** Matrisome v2.0
- **Entries:** 1,026 genes
- **Download date:** 2024-10-12

### Mouse Matrisome (Mus musculus)
- **File:** `mouse_matrisome_v2.csv`
- **Source:** [Google Sheets](https://docs.google.com/spreadsheets/d/1Te6n2q_cisXeirzBClK-VzA6T-zioOB5/edit)
- **Version:** Matrisome v2.0
- **Entries:** 1,109 genes
- **Download date:** 2024-10-12

## Data Structure

Each CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `Matrisome Division` | High-level category (Core matrisome, Matrisome-associated) |
| `Matrisome Category` | Detailed classification (ECM Glycoproteins, Collagens, Proteoglycans, etc.) |
| `Gene Symbol` | Official gene symbol (HGNC for human, MGI for mouse) |
| `Gene Name` | Full gene name |
| `Synonyms` | Alternative gene names (pipe-separated) |
| `HGNC_IDs / MGI_IDs` | Gene database identifiers |
| `HGNC_IDs Links / MGI_IDs Links` | Links to gene database entries |
| `UniProt_IDs` | UniProt accession numbers (colon-separated for multiple isoforms) |
| `Refseq_IDs` | RefSeq identifiers (colon-separated) |
| `Notes` | Additional annotations |

## Matrisome Categories

### Core Matrisome (Structural ECM proteins)
- **ECM Glycoproteins:** Laminins, fibronectins, tenascins, etc.
- **Collagens:** Fibrillar and non-fibrillar collagens
- **Proteoglycans:** Heparan sulfate, chondroitin sulfate, dermatan sulfate proteoglycans

### Matrisome-associated (Regulatory proteins)
- **ECM Regulators:** MMPs, ADAMTSs, LOX family, etc.
- **ECM-affiliated Proteins:** Growth factors, cytokines binding ECM
- **Secreted Factors:** Proteins secreted to ECM but not structural components

## Usage

These reference lists are used by the protein annotation pipeline (see `02_TASK_PROTEIN_ANNOTATION_GUIDELINES.md`) to:
1. Harmonize protein identifiers across datasets
2. Classify proteins into matrisome categories
3. Validate annotation coverage and consistency

## Citation

If you use these matrisome lists, please cite:

**Matrisome AnalyzeR:**
> Naba Lab. Matrisome AnalyzeR: A suite of tools to annotate and quantify ECM molecules in big datasets across organisms. *Journal of Cell Science* (2023) 136(17):jcs261255.

**Matrisome Database:**
> Naba A, et al. MatrisomeDB: The ECM-protein knowledge database. *Nucleic Acids Research* (2020) 48(D1):D1136–D1144.

**Original Matrisome Definition:**
> Naba A, et al. The matrisome: in silico definition and in vivo characterization by proteomics of normal and tumor extracellular matrices. *Molecular & Cellular Proteomics* (2012) 11(4):M111.014647.

## Updates

To update these reference lists:
```bash
# Human matrisome
curl -L "https://docs.google.com/spreadsheets/d/1GwwV3pFvsp7DKBbCgr8kLpf8Eh_xV8ks/export?format=csv" -o references/human_matrisome_v2.csv

# Mouse matrisome
curl -L "https://docs.google.com/spreadsheets/d/1Te6n2q_cisXeirzBClK-VzA6T-zioOB5/export?format=csv" -o references/mouse_matrisome_v2.csv
```

## Maintainer

- **Matrisome Project:** Naba Lab, University of Illinois Chicago
- **Website:** https://sites.google.com/uic.edu/matrisome
- **Contact:** See matrisome website for contact information
