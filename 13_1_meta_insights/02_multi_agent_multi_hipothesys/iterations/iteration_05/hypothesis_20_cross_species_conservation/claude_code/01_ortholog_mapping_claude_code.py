"""
H20 - Cross-Species Conservation: Ortholog Mapping
Maps human genes to mouse orthologs using Ensembl Compara API
Agent: claude_code
"""

import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, Optional
import json

print("="*80)
print("ORTHOLOG MAPPING: HUMAN → MOUSE")
print("="*80)

# Load main dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Get all unique human genes
human_genes = df[df['Species'] == 'Homo sapiens']['Gene_Symbol'].unique()
print(f"\nTotal unique human genes: {len(human_genes)}")

# Get all unique mouse genes
mouse_genes = df[df['Species'] == 'Mus musculus']['Gene_Symbol'].unique()
print(f"Total unique mouse genes: {len(mouse_genes)}")

def get_mouse_ortholog_ensembl(human_gene: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Query Ensembl REST API for mouse ortholog of a human gene

    Parameters:
    -----------
    human_gene : str
        Human gene symbol (e.g., 'S100A9')
    max_retries : int
        Number of retry attempts for failed requests

    Returns:
    --------
    dict or None
        {
            'mouse_gene': str,
            'orthology_type': str,
            'percent_identity': float,
            'dn': float,  # Non-synonymous substitutions
            'ds': float,  # Synonymous substitutions
            'dnds': float  # dN/dS ratio
        }
    """
    base_url = "https://rest.ensembl.org"

    for attempt in range(max_retries):
        try:
            # Query homology endpoint
            url = f"{base_url}/homology/symbol/human/{human_gene}"
            params = {
                'target_species': 'mouse',
                'format': 'full',
                'content-type': 'application/json'
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if 'data' in data and len(data['data']) > 0:
                    homologies = data['data'][0].get('homologies', [])

                    # Find best ortholog (1-to-1 preferred)
                    best_ortholog = None
                    for homology in homologies:
                        if homology['type'] == 'ortholog_one2one':
                            best_ortholog = homology
                            break

                    # If no 1-to-1, take first ortholog
                    if not best_ortholog and homologies:
                        best_ortholog = homologies[0]

                    if best_ortholog:
                        target = best_ortholog['target']

                        return {
                            'mouse_gene': target['id'],  # Gene symbol
                            'orthology_type': best_ortholog['type'],
                            'percent_identity': target.get('perc_id', np.nan),
                            'dn': best_ortholog.get('dn', np.nan),
                            'ds': best_ortholog.get('ds', np.nan),
                            'dnds': best_ortholog.get('dn') / best_ortholog.get('ds') if best_ortholog.get('ds', 0) > 0 else np.nan
                        }

            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            else:
                return None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None

    return None

# Map all human genes to mouse orthologs
print("\nQuerying Ensembl Compara API for orthologs...")
print("(This may take several minutes...)\n")

orthologs_map = {}
failed_genes = []

for i, gene in enumerate(human_genes):
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{len(human_genes)} genes processed...")

    ortholog = get_mouse_ortholog_ensembl(gene)

    if ortholog:
        orthologs_map[gene] = ortholog
    else:
        failed_genes.append(gene)

    # Rate limiting: 15 requests per second max
    time.sleep(0.07)

print(f"\n✓ Successfully mapped: {len(orthologs_map)}/{len(human_genes)} genes ({100*len(orthologs_map)/len(human_genes):.1f}%)")
print(f"✗ Failed to map: {len(failed_genes)} genes")

# Create DataFrame
df_orthologs = pd.DataFrame(orthologs_map).T
df_orthologs.index.name = 'Human_Gene'
df_orthologs.reset_index(inplace=True)

# Check which mouse genes from our dataset match
df_orthologs['in_our_dataset'] = df_orthologs['mouse_gene'].isin(mouse_genes)
n_in_dataset = df_orthologs['in_our_dataset'].sum()

print(f"\nOrthologs present in our mouse data: {n_in_dataset}/{len(df_orthologs)} ({100*n_in_dataset/len(df_orthologs):.1f}%)")

# Save results
output_file = '13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_20_cross_species_conservation/claude_code/orthologs_human_mouse_claude_code.csv'
df_orthologs.to_csv(output_file, index=False)
print(f"\n✓ Saved ortholog mapping to: {output_file}")

# Save failed genes
failed_file = '13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_20_cross_species_conservation/claude_code/failed_orthologs_claude_code.txt'
with open(failed_file, 'w') as f:
    f.write('\n'.join(failed_genes))
print(f"✓ Saved failed genes to: {failed_file}")

# Summary statistics
print("\n" + "="*80)
print("ORTHOLOG MAPPING SUMMARY")
print("="*80)
print(f"Orthology types:")
print(df_orthologs['orthology_type'].value_counts())
print(f"\nPercent identity statistics:")
print(df_orthologs['percent_identity'].describe())
print(f"\ndN/dS statistics:")
print(df_orthologs['dnds'].describe())

# Key genes from previous hypotheses
key_genes = ['S100A9', 'S100A10', 'S100B', 'TGM2', 'LOX', 'LOXL1', 'LOXL2',
             'COL1A1', 'COL1A2', 'SERPINE1', 'SERPINA1', 'FN1', 'VWF']

print(f"\n" + "="*80)
print("KEY GENES ORTHOLOG STATUS")
print("="*80)
for gene in key_genes:
    if gene in orthologs_map:
        info = orthologs_map[gene]
        in_data = "✓ IN DATASET" if info['mouse_gene'] in mouse_genes.tolist() else "✗ NOT IN DATASET"
        print(f"{gene:12} → {info['mouse_gene']:12} | dN/dS={info['dnds']:.3f} | {in_data}")
    else:
        print(f"{gene:12} → NOT FOUND")

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80)
