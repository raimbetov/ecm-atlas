"""
H20 - Cross-Species Conservation: Simple Ortholog Mapping
Creates ortholog map using direct gene symbol matching
Agent: claude_code
"""

import pandas as pd
import numpy as np

print("="*80)
print("SIMPLE ORTHOLOG MAPPING: HUMAN ↔ MOUSE")
print("="*80)

# Load main dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Get unique genes per species
human_genes = sorted(df[df['Species'] == 'Homo sapiens']['Gene_Symbol'].unique())
mouse_genes = sorted(df[df['Species'] == 'Mus musculus']['Gene_Symbol'].unique())

print(f"\nHuman genes: {len(human_genes)}")
print(f"Mouse genes: {len(mouse_genes)}")

# Create bidirectional mapping
# Most orthologs share the same gene symbol
orthologs_list = []

for h_gene in human_genes:
    # Direct match (case-sensitive)
    if h_gene in mouse_genes:
        orthologs_list.append({
            'Human_Gene': h_gene,
            'Mouse_Gene': h_gene,
            'Match_Type': 'exact',
            'In_Both_Datasets': True
        })
    else:
        # Try case-insensitive
        h_lower = h_gene.lower()
        matched = False
        for m_gene in mouse_genes:
            if m_gene.lower() == h_lower:
                orthologs_list.append({
                    'Human_Gene': h_gene,
                    'Mouse_Gene': m_gene,
                    'Match_Type': 'case_insensitive',
                    'In_Both_Datasets': True
                })
                matched = True
                break

        if not matched:
            # No mouse ortholog in our dataset
            orthologs_list.append({
                'Human_Gene': h_gene,
                'Mouse_Gene': None,
                'Match_Type': 'no_match',
                'In_Both_Datasets': False
            })

df_orthologs = pd.DataFrame(orthologs_list)

print(f"\n✓ Total orthologs mapped: {df_orthologs['In_Both_Datasets'].sum()}/{len(df_orthologs)}")
print(f"  - Exact matches: {(df_orthologs['Match_Type'] == 'exact').sum()}")
print(f"  - Case-insensitive: {(df_orthologs['Match_Type'] == 'case_insensitive').sum()}")
print(f"  - No match: {(df_orthologs['Match_Type'] == 'no_match').sum()}")

# Save
output_file = '13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_20_cross_species_conservation/claude_code/orthologs_human_mouse_simple_claude_code.csv'
df_orthologs.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# Key genes table
key_genes = ['S100A9', 'S100A10', 'S100B', 'TGM2', 'LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4',
             'COL1A1', 'COL1A2', 'COL3A1', 'SERPINE1', 'SERPINA1', 'FN1', 'VWF',
             'PLOD1', 'PLOD2', 'PLOD3', 'MMP2', 'MMP9', 'TIMP1']

print("\n" + "="*80)
print("KEY GENES ORTHOLOG STATUS")
print("="*80)
print(f"{'Human Gene':<15} {'Mouse Gene':<15} {'Match Type':<20} {'Available'}")
print("-"*80)

for gene in key_genes:
    row = df_orthologs[df_orthologs['Human_Gene'] == gene]
    if len(row) > 0:
        m_gene = row.iloc[0]['Mouse_Gene']
        match_type = row.iloc[0]['Match_Type']
        available = '✓' if row.iloc[0]['In_Both_Datasets'] else '✗'
        m_gene_str = m_gene if m_gene else 'N/A'
        print(f"{gene:<15} {m_gene_str:<15} {match_type:<20} {available}")
    else:
        print(f"{gene:<15} {'NOT IN HUMAN DATA':<15} {'-':<20} ✗")

print("\n" + "="*80)
print("SCRIPT COMPLETED")
print("="*80)
