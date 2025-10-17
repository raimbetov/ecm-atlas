#!/usr/bin/env python3
"""
TIMP3 Therapeutic Potential Evaluator - Agent 15
Assesses TIMP3 as multi-functional ECM protector therapeutic target
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/'

def load_data():
    """Load and prepare dataset"""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['Canonical_Gene_Symbol'].nunique()} unique proteins")
    return df

def extract_timp_trajectories(df):
    """Extract all TIMP family members across tissues and ages"""

    timps = ['TIMP1', 'TIMP2', 'TIMP3', 'TIMP4']

    timp_data = df[df['Canonical_Gene_Symbol'].isin(timps)].copy()

    print(f"\nTIMP protein measurements:")
    for timp in timps:
        count = len(timp_data[timp_data['Canonical_Gene_Symbol'] == timp])
        tissues = timp_data[timp_data['Canonical_Gene_Symbol'] == timp]['Tissue'].nunique()
        studies = timp_data[timp_data['Canonical_Gene_Symbol'] == timp]['Study_ID'].nunique()
        print(f"  {timp}: {count} measurements, {tissues} tissues, {studies} studies")

    return timp_data

def extract_destructive_enzymes(df):
    """Extract MMPs, ADAMTS, and VEGF for correlation analysis"""

    # Matrix metalloproteinases
    mmp_pattern = df['Canonical_Gene_Symbol'].str.match(r'^MMP\d+', na=False)
    mmps = df[mmp_pattern].copy()

    # ADAMTS (aggrecanases)
    adamts_pattern = df['Canonical_Gene_Symbol'].str.match(r'^ADAMTS\d+', na=False)
    adamts = df[adamts_pattern].copy()

    # VEGF family
    vegf_pattern = df['Canonical_Gene_Symbol'].str.contains('VEGF', na=False)
    vegf = df[vegf_pattern].copy()

    print(f"\nDestructive enzyme measurements:")
    print(f"  MMPs: {mmps['Canonical_Gene_Symbol'].nunique()} unique, {len(mmps)} measurements")
    print(f"  ADAMTS: {adamts['Canonical_Gene_Symbol'].nunique()} unique, {len(adamts)} measurements")
    print(f"  VEGF family: {vegf['Canonical_Gene_Symbol'].nunique()} unique, {len(vegf)} measurements")

    return mmps, adamts, vegf

def calculate_timp3_correlations(df):
    """Calculate correlations between TIMP3 and destructive enzymes"""

    # Get all unique tissue-study combinations
    tissue_studies = df[['Tissue', 'Study_ID', 'Tissue_Compartment']].drop_duplicates()

    correlations = []

    for _, ts in tissue_studies.iterrows():
        tissue = ts['Tissue']
        study = ts['Study_ID']
        compartment = ts['Tissue_Compartment']

        # Get data for this tissue-study
        subset = df[(df['Tissue'] == tissue) &
                   (df['Study_ID'] == study) &
                   (df['Tissue_Compartment'] == compartment)]

        # Get TIMP3 data
        timp3_data = subset[subset['Canonical_Gene_Symbol'] == 'TIMP3']
        if len(timp3_data) == 0:
            continue

        timp3_zscore = timp3_data['Zscore_Delta'].mean()

        # Get MMPs
        mmp_pattern = subset['Canonical_Gene_Symbol'].str.match(r'^MMP\d+', na=False)
        mmps = subset[mmp_pattern]

        if len(mmps) > 0:
            mmp_mean_zscore = mmps['Zscore_Delta'].mean()

            # Check if we have paired measurements for correlation
            if len(mmps) >= 3:  # need at least 3 points
                # Try to correlate TIMP3 with individual MMPs
                for mmp in mmps['Canonical_Gene_Symbol'].unique():
                    mmp_data = mmps[mmps['Canonical_Gene_Symbol'] == mmp]
                    if len(mmp_data) > 0:
                        correlations.append({
                            'Tissue': tissue,
                            'Study': study,
                            'Compartment': compartment,
                            'TIMP3_Zscore': timp3_zscore,
                            'Target_Type': 'MMP',
                            'Target_Gene': mmp,
                            'Target_Zscore': mmp_data['Zscore_Delta'].mean(),
                            'Direction': 'Inverse' if timp3_zscore * mmp_data['Zscore_Delta'].mean() < 0 else 'Same'
                        })

        # Get ADAMTS
        adamts_pattern = subset['Canonical_Gene_Symbol'].str.match(r'^ADAMTS\d+', na=False)
        adamts = subset[adamts_pattern]

        if len(adamts) > 0:
            for adamts_gene in adamts['Canonical_Gene_Symbol'].unique():
                adamts_data = adamts[adamts['Canonical_Gene_Symbol'] == adamts_gene]
                correlations.append({
                    'Tissue': tissue,
                    'Study': study,
                    'Compartment': compartment,
                    'TIMP3_Zscore': timp3_zscore,
                    'Target_Type': 'ADAMTS',
                    'Target_Gene': adamts_gene,
                    'Target_Zscore': adamts_data['Zscore_Delta'].mean(),
                    'Direction': 'Inverse' if timp3_zscore * adamts_data['Zscore_Delta'].mean() < 0 else 'Same'
                })

        # Get VEGF
        vegf_pattern = subset['Canonical_Gene_Symbol'].str.contains('VEGF', na=False)
        vegf = subset[vegf_pattern]

        if len(vegf) > 0:
            for vegf_gene in vegf['Canonical_Gene_Symbol'].unique():
                vegf_data = vegf[vegf['Canonical_Gene_Symbol'] == vegf_gene]
                correlations.append({
                    'Tissue': tissue,
                    'Study': study,
                    'Compartment': compartment,
                    'TIMP3_Zscore': timp3_zscore,
                    'Target_Type': 'VEGF',
                    'Target_Gene': vegf_gene,
                    'Target_Zscore': vegf_data['Zscore_Delta'].mean(),
                    'Direction': 'Inverse' if timp3_zscore * vegf_data['Zscore_Delta'].mean() < 0 else 'Same'
                })

    return pd.DataFrame(correlations)

def calculate_tissue_deficiency_index(df):
    """Calculate TIMP3 deficiency score = (MMPs + VEGF) / TIMP3"""

    tissue_scores = []

    for tissue in df['Tissue'].unique():
        tissue_data = df[df['Tissue'] == tissue]

        # Get TIMP3
        timp3 = tissue_data[tissue_data['Canonical_Gene_Symbol'] == 'TIMP3']
        if len(timp3) == 0:
            continue

        timp3_mean = timp3['Zscore_Delta'].mean()

        # Get MMPs
        mmp_pattern = tissue_data['Canonical_Gene_Symbol'].str.match(r'^MMP\d+', na=False)
        mmps = tissue_data[mmp_pattern]
        mmp_mean = mmps['Zscore_Delta'].mean() if len(mmps) > 0 else 0

        # Get VEGF
        vegf_pattern = tissue_data['Canonical_Gene_Symbol'].str.contains('VEGF', na=False)
        vegf = tissue_data[vegf_pattern]
        vegf_mean = vegf['Zscore_Delta'].mean() if len(vegf) > 0 else 0

        # Calculate deficiency index
        # Higher = more destructive activity relative to TIMP3
        deficiency_score = (mmp_mean + vegf_mean) / (timp3_mean + 0.1)  # add 0.1 to avoid div by zero

        tissue_scores.append({
            'Tissue': tissue,
            'TIMP3_Zscore': timp3_mean,
            'MMP_Zscore_Mean': mmp_mean,
            'VEGF_Zscore_Mean': vegf_mean,
            'Deficiency_Index': deficiency_score,
            'N_TIMP3': len(timp3),
            'N_MMPs': len(mmps),
            'N_VEGF': len(vegf),
            'Priority_Rank': 0  # will fill later
        })

    df_scores = pd.DataFrame(tissue_scores)

    # Rank by deficiency index (higher = worse, needs intervention)
    df_scores = df_scores.sort_values('Deficiency_Index', ascending=False)
    df_scores['Priority_Rank'] = range(1, len(df_scores) + 1)

    return df_scores

def assess_ecm_protection(df):
    """Test if high TIMP3 tissues preserve ECM better"""

    # Calculate tissue-level ECM health metrics
    tissue_health = []

    for tissue in df['Tissue'].unique():
        tissue_data = df[df['Tissue'] == tissue]

        # Get TIMP3
        timp3 = tissue_data[tissue_data['Canonical_Gene_Symbol'] == 'TIMP3']
        if len(timp3) == 0:
            continue

        timp3_zscore = timp3['Zscore_Delta'].mean()

        # Get Core ECM proteins (structural)
        core_ecm = tissue_data[tissue_data['Matrisome_Division'] == 'Core matrisome']

        # Calculate ECM degradation score (how much core ECM decreases with age)
        ecm_degradation = core_ecm[core_ecm['Zscore_Delta'] < 0]['Zscore_Delta'].mean()
        ecm_degradation = abs(ecm_degradation) if pd.notna(ecm_degradation) else 0

        # Calculate ECM stability (% of core ECM with small changes)
        ecm_stable_pct = (abs(core_ecm['Zscore_Delta']) < 0.5).sum() / len(core_ecm) * 100 if len(core_ecm) > 0 else 0

        tissue_health.append({
            'Tissue': tissue,
            'TIMP3_Zscore': timp3_zscore,
            'ECM_Degradation_Score': ecm_degradation,
            'ECM_Stable_Pct': ecm_stable_pct,
            'N_Core_ECM': len(core_ecm),
            'Protective_Effect': -ecm_degradation if timp3_zscore > 0 else ecm_degradation
        })

    df_health = pd.DataFrame(tissue_health)

    # Calculate correlation between TIMP3 and ECM protection
    if len(df_health) > 2:
        corr, pval = pearsonr(df_health['TIMP3_Zscore'], df_health['ECM_Degradation_Score'])
        print(f"\nTIMP3 vs ECM Degradation correlation: r={corr:.3f}, p={pval:.3f}")

        corr2, pval2 = pearsonr(df_health['TIMP3_Zscore'], df_health['ECM_Stable_Pct'])
        print(f"TIMP3 vs ECM Stability correlation: r={corr2:.3f}, p={pval2:.3f}")

    return df_health

def compare_timp_family(timp_data):
    """Compare TIMP3 vs other TIMPs - is TIMP3 uniquely important?"""

    timp_comparison = []

    for timp in ['TIMP1', 'TIMP2', 'TIMP3', 'TIMP4']:
        timp_subset = timp_data[timp_data['Canonical_Gene_Symbol'] == timp]

        if len(timp_subset) == 0:
            continue

        timp_comparison.append({
            'TIMP': timp,
            'Mean_Zscore_Delta': timp_subset['Zscore_Delta'].mean(),
            'Median_Zscore_Delta': timp_subset['Zscore_Delta'].median(),
            'Std_Zscore_Delta': timp_subset['Zscore_Delta'].std(),
            'N_Measurements': len(timp_subset),
            'N_Tissues': timp_subset['Tissue'].nunique(),
            'N_Studies': timp_subset['Study_ID'].nunique(),
            'Pct_Decreasing': (timp_subset['Zscore_Delta'] < -0.2).sum() / len(timp_subset) * 100,
            'Pct_Increasing': (timp_subset['Zscore_Delta'] > 0.2).sum() / len(timp_subset) * 100,
            'Max_Increase': timp_subset['Zscore_Delta'].max(),
            'Max_Decrease': timp_subset['Zscore_Delta'].min()
        })

    return pd.DataFrame(timp_comparison)

def check_longevity_proteins(df):
    """Check if TIMP3 correlates with longevity-associated proteins"""

    longevity_keywords = ['FOXO', 'SIRT', 'KLOTHO', 'IGF', 'AMPK', 'mTOR']

    # Find longevity proteins in dataset
    longevity_proteins = []
    for keyword in longevity_keywords:
        matches = df[df['Canonical_Gene_Symbol'].str.contains(keyword, na=False, case=False)]
        if len(matches) > 0:
            longevity_proteins.extend(matches['Canonical_Gene_Symbol'].unique())

    print(f"\nLongevity-associated proteins found: {len(longevity_proteins)}")
    print(f"  {', '.join(longevity_proteins[:10])}")

    # Calculate correlations with TIMP3
    correlations = []

    for tissue in df['Tissue'].unique():
        tissue_data = df[df['Tissue'] == tissue]

        timp3 = tissue_data[tissue_data['Canonical_Gene_Symbol'] == 'TIMP3']
        if len(timp3) == 0:
            continue

        timp3_zscore = timp3['Zscore_Delta'].mean()

        for protein in longevity_proteins:
            protein_data = tissue_data[tissue_data['Canonical_Gene_Symbol'] == protein]
            if len(protein_data) > 0:
                correlations.append({
                    'Tissue': tissue,
                    'Longevity_Protein': protein,
                    'TIMP3_Zscore': timp3_zscore,
                    'Protein_Zscore': protein_data['Zscore_Delta'].mean(),
                    'Direction': 'Same' if timp3_zscore * protein_data['Zscore_Delta'].mean() > 0 else 'Opposite'
                })

    return pd.DataFrame(correlations) if correlations else None

def create_visualizations(timp_data, tissue_deficiency, ecm_health, timp_comparison, correlations):
    """Generate comprehensive visualization plots"""

    fig = plt.figure(figsize=(20, 16))

    # 1. TIMP3 trajectory across all tissues
    ax1 = plt.subplot(3, 3, 1)
    timp3_only = timp_data[timp_data['Canonical_Gene_Symbol'] == 'TIMP3']

    tissues = timp3_only['Tissue'].unique()
    for tissue in tissues[:10]:  # top 10 to avoid clutter
        tissue_data = timp3_only[timp3_only['Tissue'] == tissue]
        if len(tissue_data) > 0:
            plt.scatter([tissue] * len(tissue_data), tissue_data['Zscore_Delta'], alpha=0.6, s=50)

    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('TIMP3 Z-score Delta')
    plt.title('TIMP3 Changes Across Tissues')
    plt.grid(axis='y', alpha=0.3)

    # 2. TIMP family comparison
    ax2 = plt.subplot(3, 3, 2)
    if not timp_comparison.empty:
        x_pos = range(len(timp_comparison))
        plt.bar(x_pos, timp_comparison['Mean_Zscore_Delta'],
               yerr=timp_comparison['Std_Zscore_Delta'],
               color=['red' if t == 'TIMP3' else 'gray' for t in timp_comparison['TIMP']],
               alpha=0.7)
        plt.xticks(x_pos, timp_comparison['TIMP'])
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.ylabel('Mean Z-score Delta')
        plt.title('TIMP Family Comparison (TIMP3 in red)')
        plt.grid(axis='y', alpha=0.3)

    # 3. Tissue deficiency ranking
    ax3 = plt.subplot(3, 3, 3)
    top_10_deficient = tissue_deficiency.head(10)
    y_pos = range(len(top_10_deficient))
    plt.barh(y_pos, top_10_deficient['Deficiency_Index'], color='darkred', alpha=0.7)
    plt.yticks(y_pos, top_10_deficient['Tissue'])
    plt.xlabel('TIMP3 Deficiency Index (higher = worse)')
    plt.title('Top 10 Tissues Needing TIMP3 Augmentation')
    plt.grid(axis='x', alpha=0.3)

    # 4. TIMP3 vs MMP correlation
    ax4 = plt.subplot(3, 3, 4)
    if not correlations.empty:
        mmp_corr = correlations[correlations['Target_Type'] == 'MMP']
        if len(mmp_corr) > 0:
            plt.scatter(mmp_corr['TIMP3_Zscore'], mmp_corr['Target_Zscore'],
                       alpha=0.6, s=80, c='blue')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel('TIMP3 Z-score Delta')
            plt.ylabel('MMP Z-score Delta')
            plt.title('TIMP3 vs MMPs (expect inverse correlation)')
            plt.grid(alpha=0.3)

            # Add quadrant labels
            plt.text(0.7, 0.95, 'Both increase\n(problem)', transform=ax4.transAxes,
                    ha='center', va='top', fontsize=9, alpha=0.7)
            plt.text(0.05, 0.05, 'Both decrease\n(good)', transform=ax4.transAxes,
                    ha='left', va='bottom', fontsize=9, alpha=0.7)

    # 5. TIMP3 vs VEGF correlation
    ax5 = plt.subplot(3, 3, 5)
    if not correlations.empty:
        vegf_corr = correlations[correlations['Target_Type'] == 'VEGF']
        if len(vegf_corr) > 0:
            plt.scatter(vegf_corr['TIMP3_Zscore'], vegf_corr['Target_Zscore'],
                       alpha=0.6, s=80, c='green')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel('TIMP3 Z-score Delta')
            plt.ylabel('VEGF Z-score Delta')
            plt.title('TIMP3 vs VEGF (anti-angiogenic effect)')
            plt.grid(alpha=0.3)

    # 6. TIMP3 vs ADAMTS correlation
    ax6 = plt.subplot(3, 3, 6)
    if not correlations.empty:
        adamts_corr = correlations[correlations['Target_Type'] == 'ADAMTS']
        if len(adamts_corr) > 0:
            plt.scatter(adamts_corr['TIMP3_Zscore'], adamts_corr['Target_Zscore'],
                       alpha=0.6, s=80, c='orange')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel('TIMP3 Z-score Delta')
            plt.ylabel('ADAMTS Z-score Delta')
            plt.title('TIMP3 vs ADAMTS (aggrecanase inhibition)')
            plt.grid(alpha=0.3)

    # 7. ECM protection effect
    ax7 = plt.subplot(3, 3, 7)
    if not ecm_health.empty:
        plt.scatter(ecm_health['TIMP3_Zscore'], ecm_health['ECM_Degradation_Score'],
                   s=100, alpha=0.6, c='purple')
        plt.xlabel('TIMP3 Z-score Delta')
        plt.ylabel('ECM Degradation Score')
        plt.title('TIMP3 Protective Effect on ECM')
        plt.grid(alpha=0.3)

        # Add trend line
        if len(ecm_health) > 2:
            z = np.polyfit(ecm_health['TIMP3_Zscore'], ecm_health['ECM_Degradation_Score'], 1)
            p = np.poly1d(z)
            plt.plot(ecm_health['TIMP3_Zscore'], p(ecm_health['TIMP3_Zscore']),
                    "r--", alpha=0.5, label=f'Trend')
            plt.legend()

    # 8. TIMP3 tissue distribution
    ax8 = plt.subplot(3, 3, 8)
    tissue_counts = timp3_only.groupby('Tissue').size().sort_values(ascending=False).head(12)
    plt.barh(range(len(tissue_counts)), tissue_counts.values, color='teal', alpha=0.7)
    plt.yticks(range(len(tissue_counts)), tissue_counts.index)
    plt.xlabel('Number of Measurements')
    plt.title('TIMP3 Tissue Coverage')
    plt.grid(axis='x', alpha=0.3)

    # 9. Direction consistency for all TIMPs
    ax9 = plt.subplot(3, 3, 9)
    for timp in ['TIMP1', 'TIMP2', 'TIMP3', 'TIMP4']:
        timp_subset = timp_data[timp_data['Canonical_Gene_Symbol'] == timp]
        if len(timp_subset) > 0:
            values = timp_subset['Zscore_Delta'].values
            plt.hist(values, alpha=0.5, label=timp, bins=20)

    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Z-score Delta')
    plt.ylabel('Frequency')
    plt.title('TIMP Family Z-score Distributions')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}agent_15_timp3_therapeutic_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to {OUTPUT_DIR}agent_15_timp3_therapeutic_analysis.png")
    plt.close()

def main():
    """Main analysis pipeline"""

    print("=" * 80)
    print("AGENT 15: TIMP3 THERAPEUTIC POTENTIAL EVALUATOR")
    print("=" * 80)

    # Load data
    print("\n[1/9] Loading dataset...")
    df = load_data()

    # Extract TIMP trajectories
    print("\n[2/9] Extracting TIMP protein data...")
    timp_data = extract_timp_trajectories(df)

    # Extract destructive enzymes
    print("\n[3/9] Extracting destructive enzymes (MMPs, ADAMTS, VEGF)...")
    mmps, adamts, vegf = extract_destructive_enzymes(df)

    # Calculate correlations
    print("\n[4/9] Calculating TIMP3 correlations with destructive enzymes...")
    correlations = calculate_timp3_correlations(df)
    print(f"  Found {len(correlations)} correlation pairs")

    if not correlations.empty:
        inverse_pct = (correlations['Direction'] == 'Inverse').sum() / len(correlations) * 100
        print(f"  Inverse correlations: {inverse_pct:.1f}% (expected if TIMP3 protective)")

    # Calculate tissue deficiency index
    print("\n[5/9] Calculating tissue TIMP3 deficiency scores...")
    tissue_deficiency = calculate_tissue_deficiency_index(df)
    print(f"  Ranked {len(tissue_deficiency)} tissues by deficiency")
    print(f"\n  Top 5 tissues needing TIMP3 augmentation:")
    for _, row in tissue_deficiency.head(5).iterrows():
        print(f"    {row['Tissue']:30} | Deficiency: {row['Deficiency_Index']:6.2f}")

    # Assess ECM protection
    print("\n[6/9] Assessing TIMP3 protective effect on ECM...")
    ecm_health = assess_ecm_protection(df)

    # Compare TIMP family
    print("\n[7/9] Comparing TIMP3 vs other TIMP family members...")
    timp_comparison = compare_timp_family(timp_data)
    print("\n  TIMP Family Summary:")
    for _, row in timp_comparison.iterrows():
        print(f"    {row['TIMP']:6} | Mean Δz: {row['Mean_Zscore_Delta']:+6.3f} | "
              f"Tissues: {row['N_Tissues']:2d} | Studies: {row['N_Studies']:2d}")

    # Check longevity proteins
    print("\n[8/9] Checking correlations with longevity proteins...")
    longevity_corr = check_longevity_proteins(df)

    # Create visualizations
    print("\n[9/9] Generating visualizations...")
    create_visualizations(timp_data, tissue_deficiency, ecm_health, timp_comparison, correlations)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS...")
    print("=" * 80)

    timp_data.to_csv(f'{OUTPUT_DIR}agent_15_timp_trajectories.csv', index=False)
    correlations.to_csv(f'{OUTPUT_DIR}agent_15_timp3_correlations.csv', index=False)
    tissue_deficiency.to_csv(f'{OUTPUT_DIR}agent_15_tissue_deficiency_index.csv', index=False)
    ecm_health.to_csv(f'{OUTPUT_DIR}agent_15_ecm_protection_analysis.csv', index=False)
    timp_comparison.to_csv(f'{OUTPUT_DIR}agent_15_timp_family_comparison.csv', index=False)

    if longevity_corr is not None:
        longevity_corr.to_csv(f'{OUTPUT_DIR}agent_15_longevity_correlations.csv', index=False)

    print(f"✓ Saved TIMP trajectories ({len(timp_data)} measurements)")
    print(f"✓ Saved TIMP3 correlations ({len(correlations)} pairs)")
    print(f"✓ Saved tissue deficiency index ({len(tissue_deficiency)} tissues)")
    print(f"✓ Saved ECM protection analysis ({len(ecm_health)} tissues)")
    print(f"✓ Saved TIMP family comparison")
    print(f"✓ Saved visualization plot")

    # Return summary statistics for report generation
    return {
        'timp_data': timp_data,
        'correlations': correlations,
        'tissue_deficiency': tissue_deficiency,
        'ecm_health': ecm_health,
        'timp_comparison': timp_comparison,
        'longevity_corr': longevity_corr
    }

if __name__ == '__main__':
    results = main()
    print("\n✓ ANALYSIS COMPLETE")
