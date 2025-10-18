#!/usr/bin/env python3
"""
Create ECM-only version of dataset with compartment annotations
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

# Paths
PROJECT_ROOT = Path("/Users/Kravtsovd/projects/ecm-atlas")
INPUT_CSV = PROJECT_ROOT / "08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_CSV = PROJECT_ROOT / "08_merged_ecm_dataset/merged_ecm_aging_zscore_ECM_ONLY.csv"

print("="*70)
print("CREATE ECM-ONLY DATASET WITH COMPARTMENT ANNOTATIONS")
print("="*70)

# Load data
df = pd.read_csv(INPUT_CSV)
print(f"\nLoaded {len(df):,} rows, {df['Protein_ID'].nunique():,} unique proteins")

# Filter ECM proteins only
ecm_mask = df['Matrisome_Category'].notna()
df_ecm = df[ecm_mask].copy()

print(f"\n✓ Filtered to ECM proteins only:")
print(f"  Rows: {len(df_ecm):,} ({len(df_ecm)/len(df)*100:.1f}%)")
print(f"  Unique proteins: {df_ecm['Protein_ID'].nunique():,}")

# Add compartment annotations
print(f"\nAdding compartment annotations...")

# Mapping of organs to compartment types
COMPARTMENT_ANNOTATIONS = {
    'Kidney': {
        'Glomerular': 'Glomerular basement membrane (filtration barrier)',
        'Tubulointerstitial': 'Tubular and interstitial ECM'
    },
    'Intervertebral_disc': {
        'NP': 'Nucleus pulposus (gel-like core)',
        'IAF': 'Inner annulus fibrosus',
        'OAF': 'Outer annulus fibrosus'
    },
    'Heart': {
        'Native_Tissue': 'Native cardiac ECM',
        'Decellularized_Tissue': 'Decellularized cardiac scaffold'
    },
    'Skeletal muscle': {
        'EDL': 'Extensor digitorum longus (fast-twitch)',
        'Gastrocnemius': 'Gastrocnemius (mixed fiber)',
        'Soleus': 'Soleus (slow-twitch)',
        'TA': 'Tibialis anterior (fast-twitch)'
    },
    'Brain': {
        'Cortex': 'Cortical ECM',
        'Hippocampus': 'Hippocampal ECM'
    },
    'Ovary': {
        'Cortex': 'Ovarian cortical ECM'
    },
    'Lung': {
        'Lung': 'Pulmonary ECM'
    },
    'Skin dermis': {
        'Skin dermis': 'Dermal ECM'
    }
}

def annotate_compartment(row):
    """Add compartment annotation"""
    organ = row['Organ']
    compartment = row['Compartment']
    
    if pd.isna(organ) or pd.isna(compartment):
        return None
    
    organ_map = COMPARTMENT_ANNOTATIONS.get(organ, {})
    return organ_map.get(compartment, compartment)

df_ecm['Compartment_Annotation'] = df_ecm.apply(annotate_compartment, axis=1)

# Add matrisome category simplified
def simplify_matrisome(category):
    """Simplify matrisome category"""
    if pd.isna(category):
        return None
    if 'Core matrisome' in str(category):
        return 'Core Matrisome'
    elif 'Matrisome-associated' in str(category):
        return 'Matrisome-associated'
    return category

df_ecm['Matrisome_Category_Simplified'] = df_ecm['Matrisome_Category'].apply(simplify_matrisome)

# Statistics by organ and compartment
print(f"\nECM proteins by Organ and Compartment:")
print(f"{'Organ':<20s} {'Compartment':<30s} {'Proteins':>8s}")
print("-"*70)

for organ in sorted(df_ecm['Organ'].dropna().unique()):
    organ_df = df_ecm[df_ecm['Organ'] == organ]
    for comp in sorted(organ_df['Compartment'].dropna().unique()):
        comp_df = organ_df[organ_df['Compartment'] == comp]
        n_proteins = comp_df['Protein_ID'].nunique()
        print(f"{organ:<20s} {comp:<30s} {n_proteins:8,}")

# Save
df_ecm.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved ECM-only dataset to {OUTPUT_CSV}")
print(f"  Size: {OUTPUT_CSV.stat().st_size / 1024 / 1024:.2f} MB")

# Summary statistics
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Total ECM entries: {len(df_ecm):,}")
print(f"Unique ECM proteins: {df_ecm['Protein_ID'].nunique():,}")
print(f"Studies: {df_ecm['Study_ID'].nunique()}")
print(f"Organs: {df_ecm['Organ'].nunique()}")
print(f"Compartments: {df_ecm['Compartment'].nunique()}")

print(f"\nMatrisome categories:")
for cat in sorted(df_ecm['Matrisome_Category_Simplified'].dropna().unique()):
    count = (df_ecm['Matrisome_Category_Simplified'] == cat).sum()
    print(f"  {cat}: {count:,}")

print(f"\n✓ ECM-only dataset created successfully!")
