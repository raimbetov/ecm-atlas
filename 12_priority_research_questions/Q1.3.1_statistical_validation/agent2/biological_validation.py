#!/usr/bin/env python3
"""
Biological Validation of Alternative Normalization Methods
Compare method outputs against known ECM aging biology
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("BIOLOGICAL VALIDATION ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. KNOWN ECM AGING BIOMARKERS FROM LITERATURE (2023-2025)
# ============================================================================

known_aging_markers = {
    # Core collagens - known to change with aging
    'Collagens': ['COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL4A2', 'COL14A1',
                  'COL6A1', 'COL6A2', 'COL6A3', 'COL6A5'],

    # Laminins - known to decline with aging
    'Laminins': ['LAMA1', 'LAMA2', 'LAMA5', 'LAMB1', 'LAMB2', 'LAMC1', 'LAMC2'],

    # ECM Glycoproteins - known aging markers
    'ECM_Glycoproteins': ['FN1', 'TNXB', 'TNC', 'VTN', 'THBS1', 'THBS2', 'FBLN1', 'FBLN5'],

    # MMPs/TIMPs - balance shifts with aging
    'Proteases_Inhibitors': ['MMP1', 'MMP2', 'MMP3', 'MMP9', 'MMP13', 'TIMP1', 'TIMP2', 'TIMP3', 'TIMP4'],

    # Proteoglycans - known to change
    'Proteoglycans': ['DCN', 'BGN', 'LUM', 'VCAN', 'ACAN', 'HSPG2'],

    # Novel aging markers from recent studies
    'Novel_Markers': ['HAPLN1', 'HAPLN3', 'MATN2', 'MATN3', 'MATN4', 'CHAD',
                      'PRG4', 'CILP', 'COMP', 'ANGPTL7', 'ANGPTL2'],

    # Specific declining proteins (from literature FC < -2)
    'Strong_Decliners': ['COL14A1', 'LAMB1', 'TNXB', 'COL6A5', 'LAMC2']
}

# Flatten to single list with categories
all_known_markers = []
for category, genes in known_aging_markers.items():
    for gene in genes:
        all_known_markers.append({'Gene': gene, 'Category': category, 'Known_Marker': True})

known_df = pd.DataFrame(all_known_markers)

print("\n1. KNOWN AGING BIOMARKERS FROM LITERATURE:")
print("-" * 80)
for cat, genes in known_aging_markers.items():
    print(f"{cat}: {len(genes)} proteins")
print(f"\nTOTAL KNOWN MARKERS: {len(known_df)}")

# ============================================================================
# 2. LOAD METHOD RESULTS
# ============================================================================

print("\n2. LOADING METHOD RESULTS:")
print("-" * 80)

methods = {
    'Current_ZScore': 'method0_current_zscore.csv',
    'Percentile_Norm': 'method1_percentile_norm.csv',
    'Rank_Spearman': 'method2_rank_spearman.csv',
    'Mixed_Effects': 'method3_mixed_effects.csv',
    'Global_Standard': 'method4_global_standard.csv'
}

method_data = {}
for method_name, filename in methods.items():
    try:
        df = pd.read_csv(filename)
        # Filter to significant results
        df_sig = df[df['P_Value'] < 0.05].copy()
        method_data[method_name] = df_sig
        print(f"{method_name}: {len(df_sig)} significant proteins")
    except Exception as e:
        print(f"{method_name}: Error loading - {e}")

# ============================================================================
# 3. CALCULATE BIOLOGICAL VALIDITY SCORES
# ============================================================================

print("\n3. BIOLOGICAL VALIDITY SCORES:")
print("=" * 80)

validity_scores = []

for method_name, df_method in method_data.items():
    if len(df_method) == 0:
        continue

    # Get top 20 proteins
    df_sorted = df_method.sort_values('P_Value').head(20)
    top_genes = set(df_sorted['Gene'].tolist())

    # Count matches with known markers
    matches = top_genes & set(known_df['Gene'].tolist())

    # Calculate by category
    category_scores = {}
    for category, genes in known_aging_markers.items():
        category_set = set(genes)
        category_matches = top_genes & category_set
        category_scores[category] = {
            'matches': len(category_matches),
            'genes': list(category_matches)
        }

    # Overall score
    precision = len(matches) / len(top_genes) if len(top_genes) > 0 else 0
    recall = len(matches) / len(known_df) if len(known_df) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    validity_scores.append({
        'Method': method_name,
        'Top_20_Proteins': len(top_genes),
        'Known_Marker_Matches': len(matches),
        'Precision': f"{100 * precision:.1f}%",
        'Recall': f"{100 * recall:.1f}%",
        'F1_Score': f"{100 * f1_score:.1f}%",
        'Matched_Genes': sorted(list(matches))
    })

    print(f"\n{method_name}:")
    print(f"  Top 20 proteins: {len(top_genes)}")
    print(f"  Known marker matches: {len(matches)}/{len(top_genes)} ({100*precision:.1f}%)")
    print(f"  Matched genes: {', '.join(sorted(list(matches)))}")

    print("\n  By category:")
    for category, info in category_scores.items():
        if info['matches'] > 0:
            print(f"    {category}: {info['matches']} ({', '.join(info['genes'])})")

# ============================================================================
# 4. VALIDATE AGAINST KNOWN DRIVERS FROM Q1.1.1
# ============================================================================

print("\n\n4. VALIDATION AGAINST Q1.1.1 IDENTIFIED DRIVERS:")
print("=" * 80)

# Known drivers from Q1.1.1
q1_drivers = ['Col14a1', 'TNXB', 'LAMB1', 'COL14A1']  # Include case variations

print(f"\nQ1.1.1 Top Drivers: {', '.join(q1_drivers)}")

driver_recovery = []

for method_name, df_method in method_data.items():
    if len(df_method) == 0:
        continue

    # Get top 20 proteins
    df_sorted = df_method.sort_values('P_Value').head(20)
    top_genes = set(df_sorted['Gene'].tolist())

    # Check for driver recovery (case-insensitive)
    top_genes_upper = {g.upper() for g in top_genes}
    q1_upper = {d.upper() for d in q1_drivers}

    recovered = top_genes_upper & q1_upper
    recovered_list = [g for g in top_genes if g.upper() in recovered]

    driver_recovery.append({
        'Method': method_name,
        'Drivers_Recovered': len(recovered),
        'Total_Drivers': len(set(q1_upper)),
        'Recovery_Rate': f"{100 * len(recovered) / len(set(q1_upper)):.1f}%",
        'Recovered_Genes': sorted(recovered_list)
    })

    print(f"\n{method_name}:")
    print(f"  Recovered {len(recovered)}/{len(set(q1_upper))} drivers ({100*len(recovered)/len(set(q1_upper)):.1f}%)")
    if len(recovered_list) > 0:
        print(f"  Genes: {', '.join(sorted(recovered_list))}")

# ============================================================================
# 5. STRONG DECLINER RECOVERY
# ============================================================================

print("\n\n5. STRONG DECLINER RECOVERY (FC < -2 from literature):")
print("=" * 80)

strong_decliners = set(known_aging_markers['Strong_Decliners'])
print(f"\nKnown strong decliners: {', '.join(sorted(strong_decliners))}")

decliner_recovery = []

for method_name, df_method in method_data.items():
    if len(df_method) == 0:
        continue

    # Get all significant proteins
    all_sig = set(df_method['Gene'].tolist())

    # Check recovery
    recovered = all_sig & strong_decliners

    decliner_recovery.append({
        'Method': method_name,
        'Strong_Decliners_Found': len(recovered),
        'Total_Strong_Decliners': len(strong_decliners),
        'Recovery_Rate': f"{100 * len(recovered) / len(strong_decliners):.1f}%",
        'Recovered_Genes': sorted(list(recovered))
    })

    print(f"\n{method_name}:")
    print(f"  Found {len(recovered)}/{len(strong_decliners)} ({100*len(recovered)/len(strong_decliners):.1f}%)")
    if len(recovered) > 0:
        print(f"  Genes: {', '.join(sorted(list(recovered)))}")

# ============================================================================
# 6. CONSENSUS VALIDATION
# ============================================================================

print("\n\n6. CONSENSUS PROTEIN VALIDATION:")
print("=" * 80)

consensus = pd.read_csv('consensus_proteins.csv')
print(f"\nConsensus proteins (appear in ≥2 methods): {len(consensus)}")

# Check if consensus proteins are known markers
consensus_genes = set(consensus['Gene'].tolist())
known_genes = set(known_df['Gene'].tolist())
consensus_known = consensus_genes & known_genes

print(f"Known aging markers in consensus: {len(consensus_known)}/{len(consensus_genes)} ({100*len(consensus_known)/len(consensus_genes):.1f}%)")
print(f"Genes: {', '.join(sorted(list(consensus_known)))}")

# ============================================================================
# 7. FINAL RECOMMENDATION MATRIX
# ============================================================================

print("\n\n7. FINAL RECOMMENDATION MATRIX:")
print("=" * 80)

recommendation = []

for method_name in methods.keys():
    if method_name not in method_data:
        continue

    df_method = method_data[method_name]

    # Find matching validity and driver recovery
    validity = next((v for v in validity_scores if v['Method'] == method_name), None)
    driver_rec = next((d for d in driver_recovery if d['Method'] == method_name), None)
    decliner_rec = next((d for d in decliner_recovery if d['Method'] == method_name), None)

    if validity and driver_rec and decliner_rec:
        recommendation.append({
            'Method': method_name,
            'Significant_Proteins': len(df_method),
            'Known_Marker_Recovery': validity['Precision'],
            'Q1_Driver_Recovery': driver_rec['Recovery_Rate'],
            'Strong_Decliner_Recovery': decliner_rec['Recovery_Rate'],
            'Biological_Validity': 'High' if float(validity['Precision'].rstrip('%')) > 25 else 'Medium' if float(validity['Precision'].rstrip('%')) > 15 else 'Low'
        })

rec_df = pd.DataFrame(recommendation)
print(rec_df.to_string(index=False))

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

print("\n\n8. SAVING RESULTS:")
print("-" * 80)

# Save validity scores
pd.DataFrame(validity_scores).to_csv('biological_validity_scores.csv', index=False)
print("Saved: biological_validity_scores.csv")

# Save driver recovery
pd.DataFrame(driver_recovery).to_csv('driver_recovery_rates.csv', index=False)
print("Saved: driver_recovery_rates.csv")

# Save decliner recovery
pd.DataFrame(decliner_recovery).to_csv('decliner_recovery_rates.csv', index=False)
print("Saved: decliner_recovery_rates.csv")

# Save recommendation
rec_df.to_csv('method_recommendation_matrix.csv', index=False)
print("Saved: method_recommendation_matrix.csv")

print("\n" + "=" * 80)
print("BIOLOGICAL VALIDATION COMPLETE")
print("=" * 80)
