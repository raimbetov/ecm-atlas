#!/usr/bin/env python3
"""
Enrich missing protein metadata using UniProt REST API

This script fetches missing Protein_Name and Gene_Symbol values from UniProt
for proteins that have incomplete metadata in the merged dataset.

Usage:
    python enrich_missing_metadata.py

Output:
    - merged_ecm_enrichment.csv (enrichment lookup table)
    - merged_ecm_aging_zscore_enriched.csv (main dataset with enriched data)
"""

import pandas as pd
import requests
from time import sleep
from pathlib import Path

# Configuration
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{}.json"
RATE_LIMIT_DELAY = 0.1  # seconds between requests (10 req/sec)
REQUEST_TIMEOUT = 5  # seconds

def fetch_protein_metadata(uniprot_id):
    """
    Fetch protein metadata from UniProt REST API

    Args:
        uniprot_id (str): UniProt accession (e.g., 'P01635')

    Returns:
        dict or None: Protein metadata with keys:
            - name: Full protein name
            - gene: Gene symbol
            - organism: Organism name
            - source: 'UniProt'
            - confidence: 1.0 if successful
    """
    url = UNIPROT_API.format(uniprot_id)

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()

            # Extract protein name (try recommended name first, fallback to submitted)
            protein_desc = data.get('proteinDescription', {})
            name = None

            if 'recommendedName' in protein_desc:
                name = protein_desc['recommendedName'].get('fullName', {}).get('value')
            elif 'submittedName' in protein_desc:
                submitted = protein_desc['submittedName']
                if isinstance(submitted, list) and len(submitted) > 0:
                    name = submitted[0].get('fullName', {}).get('value')

            # Extract gene symbol
            genes = data.get('genes', [])
            gene_symbol = None
            if genes and len(genes) > 0:
                gene_symbol = genes[0].get('geneName', {}).get('value')

            # Extract organism
            organism = data.get('organism', {}).get('scientificName', 'Unknown')

            return {
                'name': name,
                'gene': gene_symbol,
                'organism': organism,
                'source': 'UniProt',
                'confidence': 1.0
            }
        elif response.status_code == 404:
            print(f"  ⚠️  {uniprot_id}: Not found in UniProt")
            return None
        else:
            print(f"  ❌ {uniprot_id}: HTTP {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        print(f"  ⏱️  {uniprot_id}: Request timeout")
        return None
    except Exception as e:
        print(f"  ❌ {uniprot_id}: {str(e)}")
        return None


def enrich_dataset():
    """Main enrichment workflow"""

    print("="*70)
    print("ECM Atlas - Protein Metadata Enrichment")
    print("="*70)

    # Load main dataset
    script_dir = Path(__file__).parent
    input_file = script_dir / 'merged_ecm_aging_zscore.csv'
    enrichment_file = script_dir / 'merged_ecm_enrichment.csv'
    output_file = script_dir / 'merged_ecm_aging_zscore_enriched.csv'

    print(f"\n📂 Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Total records: {len(df)}")
    print(f"   Unique proteins: {df['Protein_ID'].nunique()}")

    # Identify proteins with missing metadata
    print("\n🔍 Analyzing missing metadata...")

    missing_name = df['Protein_Name'].isna().sum()
    missing_gene = df['Gene_Symbol'].isna().sum()

    print(f"   Missing Protein_Name: {missing_name} / {len(df)} ({missing_name/len(df)*100:.1f}%)")
    print(f"   Missing Gene_Symbol:  {missing_gene} / {len(df)} ({missing_gene/len(df)*100:.1f}%)")

    # Get unique proteins that need enrichment
    needs_enrichment = df[
        df['Protein_Name'].isna() | df['Gene_Symbol'].isna()
    ]['Protein_ID'].unique()

    print(f"\n🌐 Fetching metadata from UniProt for {len(needs_enrichment)} proteins...")
    print("   (This may take a few minutes)")

    enrichment_data = []
    success_count = 0

    for i, protein_id in enumerate(needs_enrichment, 1):
        print(f"   [{i}/{len(needs_enrichment)}] {protein_id}...", end=' ')

        metadata = fetch_protein_metadata(protein_id)

        if metadata and (metadata['name'] or metadata['gene']):
            enrichment_data.append({
                'Protein_ID': protein_id,
                'Enriched_Name': metadata['name'],
                'Enriched_Gene': metadata['gene'],
                'Organism': metadata['organism'],
                'Source': metadata['source'],
                'Confidence': metadata['confidence']
            })
            success_count += 1
            print(f"✅ {metadata['gene'] or 'N/A'} | {(metadata['name'] or 'N/A')[:40]}")
        else:
            print("❌ Failed to fetch")

        # Rate limiting
        sleep(RATE_LIMIT_DELAY)

    print(f"\n📊 Enrichment Results:")
    print(f"   Successfully enriched: {success_count} / {len(needs_enrichment)} ({success_count/len(needs_enrichment)*100:.1f}%)")
    print(f"   Failed: {len(needs_enrichment) - success_count}")

    # Save enrichment table
    if enrichment_data:
        enrichment_df = pd.DataFrame(enrichment_data)
        enrichment_df.to_csv(enrichment_file, index=False)
        print(f"\n💾 Saved enrichment table: {enrichment_file}")
        print(f"   Columns: {list(enrichment_df.columns)}")
    else:
        print("\n⚠️  No enrichment data to save")
        return

    # Merge enrichment back to main dataset
    print(f"\n🔗 Merging enrichment data with main dataset...")
    df_enriched = df.merge(enrichment_df[['Protein_ID', 'Enriched_Name', 'Enriched_Gene']],
                           on='Protein_ID',
                           how='left')

    # Fill missing values with enriched data
    original_name_nulls = df_enriched['Protein_Name'].isna().sum()
    original_gene_nulls = df_enriched['Gene_Symbol'].isna().sum()

    df_enriched['Protein_Name'] = df_enriched['Protein_Name'].fillna(df_enriched['Enriched_Name'])
    df_enriched['Gene_Symbol'] = df_enriched['Gene_Symbol'].fillna(df_enriched['Enriched_Gene'])

    # Add data quality flag
    df_enriched['Data_Quality'] = 'Original'
    df_enriched.loc[df_enriched['Enriched_Name'].notna(), 'Data_Quality'] = 'Enriched_UniProt'

    # Remove temporary enrichment columns
    df_enriched = df_enriched.drop(columns=['Enriched_Name', 'Enriched_Gene'])

    # Save enriched dataset
    df_enriched.to_csv(output_file, index=False)

    # Summary
    new_name_nulls = df_enriched['Protein_Name'].isna().sum()
    new_gene_nulls = df_enriched['Gene_Symbol'].isna().sum()

    print(f"\n✅ Enrichment Complete!")
    print(f"\n📈 Before → After:")
    print(f"   Protein_Name NaN:  {original_name_nulls} → {new_name_nulls} (fixed {original_name_nulls - new_name_nulls})")
    print(f"   Gene_Symbol NaN:   {original_gene_nulls} → {new_gene_nulls} (fixed {original_gene_nulls - new_gene_nulls})")

    print(f"\n📁 Output files:")
    print(f"   Enrichment table:  {enrichment_file}")
    print(f"   Enriched dataset:  {output_file}")

    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Review enrichment_file for accuracy")
    print("  2. Update api_server.py to load enriched dataset:")
    print(f"     df = pd.read_csv('{output_file.name}')")
    print("  3. Restart API server")
    print("="*70)


if __name__ == '__main__':
    enrich_dataset()
