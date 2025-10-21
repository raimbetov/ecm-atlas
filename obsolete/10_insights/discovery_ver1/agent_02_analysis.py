#!/usr/bin/env python3
"""
AGENT 02: NON-LINEAR TRAJECTORY FINDER
Mission: Identify proteins with unexpected, non-monotonic aging patterns
"""

import pandas as pd
import numpy as np

def load_and_analyze():
    """Load existing nonlinear trajectories and prepare focused AGENT 02 output"""

    print("="*80)
    print("AGENT 02: NON-LINEAR TRAJECTORY FINDER")
    print("="*80)
    print("\nMission: Find proteins with UNEXPECTED, NON-MONOTONIC aging patterns")
    print("- U-shaped (decrease then increase)")
    print("- Inverted-U (increase then decrease)")
    print("- Threshold effects (sudden dramatic shifts)")
    print()

    # Load existing analysis
    df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/nonlinear_trajectories.csv')

    print(f"Total proteins with nonlinear patterns: {len(df)}")

    # Filter for most interesting cases
    # Priority criteria:
    # 1. High nonlinearity gain (polynomial fits much better than linear)
    # 2. Clear U-shaped or Inverted-U patterns
    # 3. Significant threshold scores
    # 4. Multiple studies (more reliable)

    # Score each protein
    df['interest_score'] = (
        df['Nonlinearity_Gain'] * 3 +  # Primary metric
        df['Threshold_Score'] +          # Sudden changes are interesting
        (df['N_Studies'] / 5) +          # More studies = more reliable
        (df['Sign_Changes'] * 0.5)       # Direction reversals
    )

    # Classify pattern clarity
    df['pattern_clarity'] = 'Unclear'
    df.loc[df['Shape'].str.contains('U-shaped', na=False), 'pattern_clarity'] = 'U-shaped'
    df.loc[df['Shape'].str.contains('Inverted-U', na=False), 'pattern_clarity'] = 'Inverted-U'

    # Filter out linear patterns and low-quality data
    interesting = df[
        (df['Shape'].str.contains('U-shaped|Inverted-U', na=False)) &
        (df['Nonlinearity_Gain'] > 0.2)  # Clear non-linearity
    ].copy()

    # Sort by interest score
    interesting = interesting.sort_values('interest_score', ascending=False)

    print(f"\nHigh-interest nonlinear proteins: {len(interesting)}")
    print(f"- U-shaped trajectories: {len(interesting[interesting['pattern_clarity'] == 'U-shaped'])}")
    print(f"- Inverted-U trajectories: {len(interesting[interesting['pattern_clarity'] == 'Inverted-U'])}")

    # Select top 15-30 for focused analysis
    top_proteins = interesting.head(30)

    print(f"\nSelected top {len(top_proteins)} proteins for detailed analysis")

    # Create output dataframe with key metrics
    output = pd.DataFrame({
        'Gene_Symbol': top_proteins['Gene'],
        'Tissue_Compartment': top_proteins['Tissue_Compartment'],
        'Matrisome_Category': top_proteins['Matrisome_Category'],
        'Pattern_Type': top_proteins['pattern_clarity'],
        'N_Studies': top_proteins['N_Studies'],
        'Nonlinearity_Gain': top_proteins['Nonlinearity_Gain'].round(4),
        'Threshold_Score': top_proteins['Threshold_Score'].round(2),
        'Sign_Changes': top_proteins['Sign_Changes'],
        'Poly_Coef_a': top_proteins['Poly_Coef_a'].round(4),
        'Poly_Coef_b': top_proteins['Poly_Coef_b'].round(4),
        'R2_Polynomial': top_proteins['R2_Polynomial'].round(4),
        'R2_Linear': top_proteins['R2_Linear'].round(4),
        'Vertex_Position': top_proteins['Vertex_X'].round(2),
        'Interest_Score': top_proteins['interest_score'].round(2),
        'Zscore_Trajectory': top_proteins['Deltas']
    })

    # Add biological interpretation hints
    output['Biological_Hypothesis'] = output.apply(lambda row:
        f"{'Compensatory response' if row['Pattern_Type'] == 'U-shaped' else 'Adaptive peak'} "
        f"in {row['Tissue_Compartment']} tissue. "
        f"{'High threshold suggests sudden phase transition.' if row['Threshold_Score'] > 2 else 'Gradual transition.'}"
    , axis=1)

    # Save output
    output_path = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/discovery_ver1/agent_02_nonlinear_trajectories.csv'
    output.to_csv(output_path, index=False)
    print(f"\n✅ Saved: {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("TOP 10 MOST INTERESTING NON-LINEAR PROTEINS")
    print("="*80)

    for idx, row in output.head(10).iterrows():
        print(f"\n{idx+1}. {row['Gene_Symbol']} ({row['Tissue_Compartment']})")
        print(f"   Pattern: {row['Pattern_Type']}")
        print(f"   Category: {row['Matrisome_Category']}")
        print(f"   Nonlinearity Gain: {row['Nonlinearity_Gain']:.3f} (polynomial fits {row['Nonlinearity_Gain']*100:.1f}% better)")
        print(f"   Threshold Score: {row['Threshold_Score']:.2f}")
        print(f"   Studies: {row['N_Studies']}")
        print(f"   Hypothesis: {row['Biological_Hypothesis']}")

    # Summary by pattern type
    print("\n" + "="*80)
    print("PATTERN DISTRIBUTION")
    print("="*80)

    pattern_summary = output.groupby('Pattern_Type').agg({
        'Gene_Symbol': 'count',
        'Nonlinearity_Gain': 'mean',
        'Threshold_Score': 'mean'
    }).round(3)
    pattern_summary.columns = ['Count', 'Avg_Nonlinearity_Gain', 'Avg_Threshold_Score']
    print(pattern_summary)

    # Tissue distribution
    print("\n" + "="*80)
    print("TISSUE DISTRIBUTION")
    print("="*80)

    tissue_summary = output['Tissue_Compartment'].value_counts().head(10)
    for tissue, count in tissue_summary.items():
        print(f"{tissue}: {count} proteins")

    # Matrisome category distribution
    print("\n" + "="*80)
    print("MATRISOME CATEGORY DISTRIBUTION")
    print("="*80)

    category_summary = output['Matrisome_Category'].value_counts()
    for category, count in category_summary.items():
        print(f"{category}: {count} proteins")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    print("\n1. NON-MONOTONIC PATTERNS SUGGEST:")
    print("   - Compensatory mechanisms (cell fights back)")
    print("   - Phase transitions (qualitative change at specific age)")
    print("   - Biphasic responses (different mechanisms at different ages)")

    print("\n2. U-SHAPED TRAJECTORIES:")
    u_shaped = output[output['Pattern_Type'] == 'U-shaped']
    if len(u_shaped) > 0:
        print(f"   - {len(u_shaped)} proteins show initial decline then recovery")
        print(f"   - Suggests protective/compensatory upregulation in late aging")
        print(f"   - Top examples: {', '.join(u_shaped.head(5)['Gene_Symbol'].tolist())}")

    print("\n3. INVERTED-U TRAJECTORIES:")
    inverted_u = output[output['Pattern_Type'] == 'Inverted-U']
    if len(inverted_u) > 0:
        print(f"   - {len(inverted_u)} proteins show peak then decline")
        print(f"   - Suggests adaptive peak response followed by exhaustion")
        print(f"   - Top examples: {', '.join(inverted_u.head(5)['Gene_Symbol'].tolist())}")

    print("\n4. THRESHOLD EFFECTS:")
    high_threshold = output[output['Threshold_Score'] > 3]
    if len(high_threshold) > 0:
        print(f"   - {len(high_threshold)} proteins show dramatic sudden shifts")
        print(f"   - Indicates critical age thresholds")
        print(f"   - Examples: {', '.join(high_threshold.head(5)['Gene_Symbol'].tolist())}")

    print("\n5. THERAPEUTIC IMPLICATIONS:")
    print("   - U-shaped proteins: Intervene before decline starts")
    print("   - Inverted-U proteins: Support during peak, prevent decline")
    print("   - Threshold proteins: Target intervention before critical age")

    return output

if __name__ == "__main__":
    results = load_and_analyze()
    print("\n✅ Analysis complete!")
