#!/usr/bin/env python3
"""
Extract tables from Lofaro 2021 PDF supplementary tables - FINAL VERSION
"""

import pandas as pd
import pdfplumber
import re
import os

# File paths
base_dir = "/Users/Kravtsovd/projects/ecm-atlas/data_raw/Lofaro et al. - 2021"
s1_pdf = os.path.join(base_dir, "Table S1.pdf")
s2_pdf = os.path.join(base_dir, "Table S2.pdf")
s1_csv = os.path.join(base_dir, "Table_S1_proteins.csv")
s2_csv = os.path.join(base_dir, "Table_S2_quantification.csv")

def extract_s1_proteins(pdf_path, csv_path):
    """Extract Table S1 - Protein list with proper column parsing"""
    print(f"\n{'='*60}")
    print(f"Extracting Table S1 - Protein List")
    print('='*60)

    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")

        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                all_text.append(text)

    full_text = "\n".join(all_text)
    lines = full_text.split('\n')

    proteins = []
    current_extraction = None

    for line in lines:
        line = line.strip()

        # Detect extraction method section
        if 'PBS extract' in line:
            current_extraction = 'PBS'
            continue
        elif 'Urea extract' in line:
            current_extraction = 'Urea'
            continue

        # Skip headers and empty lines
        if not line or 'Protein symbol' in line or 'Num. of significant' in line:
            continue
        if line.startswith('Table S1.') or 'LC-MS/MS raw data' in line:
            continue

        # Parse protein lines
        # Format: SYMBOL_MOUSE Score Mass Matches Sequences UniqueSeq Protein name
        if '_MOUSE' in line:
            parts = line.split()
            if len(parts) >= 7:
                protein_symbol = parts[0]
                score = parts[1]
                mass = parts[2]
                num_matches = parts[3]
                num_sequences = parts[4]
                num_unique_seq = parts[5]
                protein_name = ' '.join(parts[6:])

                # Extract gene symbol (part before _MOUSE)
                gene_symbol = protein_symbol.replace('_MOUSE', '')

                try:
                    proteins.append({
                        'Extraction_Method': current_extraction,
                        'Protein_Symbol': protein_symbol,
                        'Gene_Symbol': gene_symbol,
                        'Score': int(score),
                        'Mass': int(mass),
                        'Num_Significant_Matches': int(num_matches),
                        'Num_Significant_Sequences': int(num_sequences),
                        'Num_Unique_Sequences': int(num_unique_seq),
                        'Protein_Name': protein_name
                    })
                except ValueError:
                    # Skip lines that don't parse correctly
                    continue

    if proteins:
        df = pd.DataFrame(proteins)
        df.to_csv(csv_path, index=False)

        print(f"\n✓ Saved to: {os.path.basename(csv_path)}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())

        print(f"\nSummary by extraction method:")
        print(df.groupby('Extraction_Method').size())

        return df
    else:
        print("❌ No protein entries found")
        return None

def extract_s2_quantification(pdf_path, csv_path):
    """Extract Table S2 - Quantification with ALL categories"""
    print(f"\n{'='*60}")
    print(f"Extracting Table S2 - Quantification")
    print('='*60)

    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")

        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                all_text.append(text)

    full_text = "\n".join(all_text)
    lines = full_text.split('\n')

    proteins = []
    current_category = 'ECM regulators'  # Default category for first section
    current_extraction = None

    for line in lines:
        line = line.strip()

        # Check for extraction method headers
        if 'PBS extraction' in line:
            current_extraction = 'PBS'
            current_category = 'ECM regulators'  # Reset to default
            continue
        elif 'Urea extraction' in line:
            current_extraction = 'Urea'
            current_category = 'ECM regulators'  # Reset to default
            continue

        # Check for category headers (including all variations)
        category_patterns = [
            'ECM regulators',
            'ECM_Glycoproteins',
            'ECM Glycoproteins',
            'Proteoglycans',
            'Collagens',
            'ECM affiliated',
            'ECM-affiliated',
            'Secreted factors',
            'Secreted Factors'
        ]

        is_category_line = False
        for pattern in category_patterns:
            if pattern in line and not re.match(r'^[A-Z0-9]+\s+[-]?\d', line):
                # Normalize category names
                if 'Glycoprotein' in pattern:
                    current_category = 'ECM Glycoproteins'
                elif 'affiliated' in pattern.lower():
                    current_category = 'ECM-affiliated Proteins'
                elif 'Secreted' in pattern:
                    current_category = 'Secreted Factors'
                else:
                    current_category = pattern
                is_category_line = True
                break

        if is_category_line:
            continue

        # Skip header lines
        if 'Protein symbol' in line or 'Log Fold change' in line or 'p-value' in line:
            continue
        if line.startswith('Category') or not line:
            continue

        # Try to parse protein data
        # Pattern: GENE -0,123 0,4567 Protein name
        match = re.match(r'^([A-Z0-9]+)\s+([-]?\d+[,\.]\d+)\s+([-]?\d+[,\.]\d+)\s+(.+)$', line)
        if match and current_extraction:
            gene_symbol = match.group(1)
            log_fold_change = match.group(2).replace(',', '.')
            p_value = match.group(3).replace(',', '.')
            protein_name = match.group(4)

            try:
                proteins.append({
                    'Extraction_Method': current_extraction,
                    'Category': current_category,
                    'Gene_Symbol': gene_symbol,
                    'Log_Fold_Change': float(log_fold_change),
                    'P_Value': float(p_value),
                    'Protein_Name': protein_name
                })
            except ValueError:
                continue

    if proteins:
        df = pd.DataFrame(proteins)
        df.to_csv(csv_path, index=False)

        print(f"\n✓ Saved to: {os.path.basename(csv_path)}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())

        print(f"\nSummary by extraction method and category:")
        summary = df.groupby(['Extraction_Method', 'Category']).size()
        print(summary)

        print(f"\nAll unique categories:")
        print(df['Category'].unique())

        return df
    else:
        print("❌ No protein entries found")
        return None

def main():
    print("="*60)
    print("LOFARO 2021 - PDF TABLE EXTRACTION - FINAL")
    print("="*60)

    # Extract Table S1
    print("\n### TABLE S1 - PROTEIN LIST ###")
    df_s1 = extract_s1_proteins(s1_pdf, s1_csv)

    # Extract Table S2
    print("\n\n### TABLE S2 - QUANTIFICATION ###")
    df_s2 = extract_s2_quantification(s2_pdf, s2_csv)

    # Final summary
    print("\n\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)

    if df_s1 is not None:
        print(f"\n✓ Table S1: {len(df_s1)} rows, {len(df_s1.columns)} columns")
        print(f"  Output: {s1_csv}")
    else:
        print("\n❌ Table S1: Failed to extract")

    if df_s2 is not None:
        print(f"\n✓ Table S2: {len(df_s2)} rows, {len(df_s2.columns)} columns")
        print(f"  Output: {s2_csv}")
    else:
        print("\n❌ Table S2: Failed to extract")

if __name__ == "__main__":
    main()
