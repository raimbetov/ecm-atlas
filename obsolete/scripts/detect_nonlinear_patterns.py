#!/usr/bin/env python3
"""
AGENT 3: Non-linear Pattern Detector
Mission: Find non-linear, threshold, and phase-transition aging patterns
that linear analysis would miss in ECM-Atlas aging data.

Analyses:
1. Non-monotonic trajectories (U-shaped, inverted-U, threshold effects)
2. Protein interaction networks (synergistic/antagonistic pairs)
3. Switch points (simultaneous multi-protein flips)
4. Bimodal distributions (distinct aging trajectories)
5. Complex polynomial/kernel patterns
6. Hidden variables (protein combinations predicting aging)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def load_unified_dataset(file_path):
    """Load merged ECM dataset with all studies"""
    print("="*80)
    print("LOADING ECM-ATLAS UNIFIED DATASET")
    print("="*80)

    df = pd.read_csv(file_path)

    print(f"\nTotal records: {len(df):,}")
    print(f"Unique proteins: {df['Gene_Symbol'].nunique():,}")
    print(f"Studies: {df['Study_ID'].nunique()}")
    print(f"Tissues: {df['Tissue'].nunique()}")
    print(f"Species: {df['Species'].nunique()}")

    # Filter valid z-score data
    valid = df.dropna(subset=['Zscore_Delta'])
    print(f"\nRecords with valid z-score data: {len(valid):,}")

    return df, valid


def detect_nonmonotonic_trajectories(df):
    """
    Task 1: Find proteins with non-monotonic aging patterns
    - U-shaped (decrease then increase)
    - Inverted-U (increase then decrease)
    - Threshold effects (stable until sudden jump)
    """
    print("\n" + "="*80)
    print("TASK 1: NON-MONOTONIC AGING TRAJECTORIES")
    print("="*80)

    results = []

    # Group by protein and tissue compartment
    for (gene, tissue_comp), group in df.groupby(['Gene_Symbol', 'Tissue_Compartment']):
        if len(group) < 3:  # Need at least 3 points
            continue

        # Sort by study (proxy for time/methodology diversity)
        group = group.sort_values('Study_ID')

        # Get z-score trajectory
        z_young = group['Zscore_Young'].dropna().values
        z_old = group['Zscore_Old'].dropna().values

        if len(z_young) < 2 or len(z_old) < 2:
            continue

        # Calculate direction changes (sign flips)
        deltas = group['Zscore_Delta'].dropna().values
        if len(deltas) < 3:
            continue

        # Detect sign changes in deltas
        sign_changes = np.sum(np.diff(np.sign(deltas)) != 0)

        # Polynomial fit (quadratic) to detect curvature
        if len(deltas) >= 3:
            x = np.arange(len(deltas))
            try:
                # Fit quadratic: y = ax^2 + bx + c
                poly_coefs = np.polyfit(x, deltas, 2)
                a, b, c = poly_coefs

                # Calculate R-squared for polynomial vs linear
                poly_fit = np.polyval(poly_coefs, x)
                linear_coefs = np.polyfit(x, deltas, 1)
                linear_fit = np.polyval(linear_coefs, x)

                ss_res_poly = np.sum((deltas - poly_fit)**2)
                ss_res_linear = np.sum((deltas - linear_fit)**2)
                ss_tot = np.sum((deltas - np.mean(deltas))**2)

                r2_poly = 1 - (ss_res_poly / ss_tot) if ss_tot > 0 else 0
                r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0

                # Non-linearity score: how much better is polynomial?
                nonlinearity_gain = r2_poly - r2_linear

                # Classify trajectory shape
                if a > 0.01:  # U-shaped (concave up)
                    shape = "U-shaped (decrease then increase)"
                    vertex_x = -b / (2*a)
                    is_nonlinear = True
                elif a < -0.01:  # Inverted-U (concave down)
                    shape = "Inverted-U (increase then decrease)"
                    vertex_x = -b / (2*a)
                    is_nonlinear = True
                else:
                    shape = "Linear/monotonic"
                    vertex_x = None
                    is_nonlinear = False

                # Threshold detection: sudden jump in delta
                if len(deltas) >= 3:
                    delta_diffs = np.abs(np.diff(deltas))
                    max_jump = np.max(delta_diffs)
                    mean_jump = np.mean(delta_diffs)
                    threshold_score = max_jump / (mean_jump + 0.001)
                else:
                    threshold_score = 0

                if is_nonlinear or sign_changes > 0 or threshold_score > 3:
                    results.append({
                        'Gene': gene,
                        'Tissue_Compartment': tissue_comp,
                        'N_Studies': len(deltas),
                        'Shape': shape,
                        'Sign_Changes': sign_changes,
                        'Poly_Coef_a': a,
                        'Poly_Coef_b': b,
                        'Poly_Coef_c': c,
                        'R2_Polynomial': r2_poly,
                        'R2_Linear': r2_linear,
                        'Nonlinearity_Gain': nonlinearity_gain,
                        'Threshold_Score': threshold_score,
                        'Vertex_X': vertex_x,
                        'Deltas': deltas.tolist(),
                        'Matrisome_Category': group['Matrisome_Category'].iloc[0] if 'Matrisome_Category' in group.columns else 'Unknown'
                    })

            except Exception as e:
                continue

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # Sort by nonlinearity gain
        results_df = results_df.sort_values('Nonlinearity_Gain', ascending=False)

        print(f"\nFound {len(results_df)} proteins with non-linear patterns")
        print(f"\nU-shaped trajectories: {len(results_df[results_df['Shape'].str.contains('U-shaped', na=False)])}")
        print(f"Inverted-U trajectories: {len(results_df[results_df['Shape'].str.contains('Inverted-U', na=False)])}")
        print(f"Proteins with sign changes: {len(results_df[results_df['Sign_Changes'] > 0])}")
        print(f"Threshold effects (score > 3): {len(results_df[results_df['Threshold_Score'] > 3])}")

        print("\nTop 10 most non-linear proteins:")
        for idx, row in results_df.head(10).iterrows():
            print(f"\n{row['Gene']} ({row['Tissue_Compartment']})")
            print(f"  Shape: {row['Shape']}")
            print(f"  Nonlinearity gain: {row['Nonlinearity_Gain']:.3f}")
            print(f"  Threshold score: {row['Threshold_Score']:.2f}")
            print(f"  Category: {row['Matrisome_Category']}")
    else:
        print("\nNo significant non-linear patterns detected")
        results_df = pd.DataFrame()

    return results_df


def detect_protein_interactions(df):
    """
    Task 2: Identify protein pairs with interaction effects
    - Protein A↑ only when Protein B↓ (conditional dependencies)
    - Synergistic changes (both change together more than expected)
    """
    print("\n" + "="*80)
    print("TASK 2: PROTEIN INTERACTION NETWORKS")
    print("="*80)

    # Build protein-tissue matrix
    pivot = df.pivot_table(
        index=['Study_ID', 'Tissue_Compartment'],
        columns='Gene_Symbol',
        values='Zscore_Delta',
        aggfunc='mean'
    )

    print(f"\nAnalyzing {pivot.shape[1]} proteins across {pivot.shape[0]} tissue-study combinations")

    # Calculate pairwise correlations
    corr_matrix = pivot.corr(method='pearson')

    interactions = []

    # Find strong correlations
    for i, gene_a in enumerate(corr_matrix.columns):
        for j, gene_b in enumerate(corr_matrix.columns):
            if i >= j:  # Skip self and duplicates
                continue

            corr = corr_matrix.loc[gene_a, gene_b]

            if pd.isna(corr):
                continue

            # Strong positive (synergistic) or negative (antagonistic) correlation
            if abs(corr) > 0.6:
                # Calculate co-occurrence
                gene_a_data = pivot[gene_a].dropna()
                gene_b_data = pivot[gene_b].dropna()

                # Get overlapping indices
                common_idx = gene_a_data.index.intersection(gene_b_data.index)

                if len(common_idx) < 3:
                    continue

                a_vals = pivot.loc[common_idx, gene_a].values
                b_vals = pivot.loc[common_idx, gene_b].values

                # Statistical significance
                _, p_value = stats.pearsonr(a_vals, b_vals)

                # Classify interaction type
                if corr > 0.6:
                    interaction_type = "Synergistic (both change together)"
                elif corr < -0.6:
                    interaction_type = "Antagonistic (opposite changes)"
                else:
                    interaction_type = "Unknown"

                interactions.append({
                    'Protein_A': gene_a,
                    'Protein_B': gene_b,
                    'Correlation': corr,
                    'P_value': p_value,
                    'N_Observations': len(common_idx),
                    'Interaction_Type': interaction_type,
                    'Mean_Delta_A': np.mean(a_vals),
                    'Mean_Delta_B': np.mean(b_vals)
                })

    interactions_df = pd.DataFrame(interactions)

    if len(interactions_df) > 0:
        interactions_df = interactions_df.sort_values('Correlation', ascending=False, key=abs)

        # Filter by significance
        significant = interactions_df[interactions_df['P_value'] < 0.05]

        print(f"\nFound {len(significant)} significant protein interactions (p < 0.05)")
        print(f"Synergistic pairs: {len(significant[significant['Correlation'] > 0])}")
        print(f"Antagonistic pairs: {len(significant[significant['Correlation'] < 0])}")

        print("\nTop 10 strongest interactions:")
        for idx, row in significant.head(10).iterrows():
            print(f"\n{row['Protein_A']} <-> {row['Protein_B']}")
            print(f"  Type: {row['Interaction_Type']}")
            print(f"  Correlation: {row['Correlation']:.3f} (p={row['P_value']:.2e})")
            print(f"  Mean changes: Δz_A={row['Mean_Delta_A']:.3f}, Δz_B={row['Mean_Delta_B']:.3f}")
    else:
        print("\nNo significant protein interactions detected")
        significant = pd.DataFrame()

    return significant


def detect_switch_points(df):
    """
    Task 3: Find switch points - tissues/ages where multiple proteins
    flip direction simultaneously
    """
    print("\n" + "="*80)
    print("TASK 3: SWITCH POINT DETECTION")
    print("="*80)

    switch_points = []

    # Analyze by tissue compartment
    for tissue_comp, group in df.groupby('Tissue_Compartment'):
        if len(group) < 10:  # Need enough proteins
            continue

        # Count proteins increasing vs decreasing
        increasing = len(group[group['Zscore_Delta'] > 0.5])
        decreasing = len(group[group['Zscore_Delta'] < -0.5])
        stable = len(group[abs(group['Zscore_Delta']) <= 0.5])
        total = len(group)

        # Calculate dominance ratio
        if increasing > decreasing:
            dominant = "Upregulation"
            ratio = increasing / total
        elif decreasing > increasing:
            dominant = "Downregulation"
            ratio = decreasing / total
        else:
            dominant = "Balanced"
            ratio = max(increasing, decreasing) / total

        # Detect if this is a switch point (high ratio)
        if ratio > 0.6:  # 60% of proteins moving in same direction
            switch_points.append({
                'Tissue_Compartment': tissue_comp,
                'Dominant_Direction': dominant,
                'Dominance_Ratio': ratio,
                'N_Increasing': increasing,
                'N_Decreasing': decreasing,
                'N_Stable': stable,
                'Total_Proteins': total
            })

    switch_df = pd.DataFrame(switch_points)

    if len(switch_df) > 0:
        switch_df = switch_df.sort_values('Dominance_Ratio', ascending=False)

        print(f"\nFound {len(switch_df)} potential switch points")
        print("\nTop switch points (coordinated protein changes):")
        for idx, row in switch_df.head(10).iterrows():
            print(f"\n{row['Tissue_Compartment']}")
            print(f"  Direction: {row['Dominant_Direction']}")
            print(f"  Dominance: {row['Dominance_Ratio']*100:.1f}%")
            print(f"  Proteins: ↑{row['N_Increasing']} ↓{row['N_Decreasing']} →{row['N_Stable']}")
    else:
        print("\nNo clear switch points detected")
        switch_df = pd.DataFrame()

    return switch_df


def detect_bimodal_distributions(df):
    """
    Task 4: Detect bimodal distributions - proteins with two distinct
    aging trajectories across studies/tissues
    """
    print("\n" + "="*80)
    print("TASK 4: BIMODAL DISTRIBUTION DETECTION")
    print("="*80)

    bimodal_proteins = []

    # Analyze each protein across all studies
    for gene, group in df.groupby('Gene_Symbol'):
        deltas = group['Zscore_Delta'].dropna().values

        if len(deltas) < 10:  # Need enough data points
            continue

        # Test for bimodality using Hartigan's dip test approximation
        # Simple approach: fit 2-component Gaussian mixture
        from scipy.stats import norm

        # Standardize
        deltas_std = (deltas - np.mean(deltas)) / (np.std(deltas) + 0.001)

        # K-means with k=2 to find potential modes
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(deltas_std.reshape(-1, 1))

        cluster0 = deltas[clusters == 0]
        cluster1 = deltas[clusters == 1]

        # Check if clusters are well-separated
        mean0 = np.mean(cluster0)
        mean1 = np.mean(cluster1)
        std0 = np.std(cluster0)
        std1 = np.std(cluster1)

        # Separation metric (distance between means relative to stds)
        separation = abs(mean1 - mean0) / (std0 + std1 + 0.001)

        # Bimodality coefficient (BC)
        # BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)(n-3)))
        skewness = stats.skew(deltas)
        kurtosis = stats.kurtosis(deltas)
        n = len(deltas)
        bc = (skewness**2 + 1) / (kurtosis + 3 * (n-1)**2 / ((n-2)*(n-3))) if n > 3 else 0

        # BC > 0.555 suggests bimodality
        if separation > 1.5 or bc > 0.555:
            bimodal_proteins.append({
                'Gene': gene,
                'N_Observations': len(deltas),
                'Separation_Score': separation,
                'Bimodality_Coefficient': bc,
                'Cluster0_Mean': mean0,
                'Cluster0_Std': std0,
                'Cluster0_N': len(cluster0),
                'Cluster1_Mean': mean1,
                'Cluster1_Std': std1,
                'Cluster1_N': len(cluster1),
                'Overall_Mean': np.mean(deltas),
                'Overall_Std': np.std(deltas)
            })

    bimodal_df = pd.DataFrame(bimodal_proteins)

    if len(bimodal_df) > 0:
        bimodal_df = bimodal_df.sort_values('Separation_Score', ascending=False)

        print(f"\nFound {len(bimodal_df)} proteins with bimodal aging patterns")
        print("\nTop 10 most bimodal proteins:")
        for idx, row in bimodal_df.head(10).iterrows():
            print(f"\n{row['Gene']}")
            print(f"  Separation score: {row['Separation_Score']:.2f}")
            print(f"  Bimodality coefficient: {row['Bimodality_Coefficient']:.3f}")
            print(f"  Cluster 1: Δz={row['Cluster0_Mean']:.3f}±{row['Cluster0_Std']:.3f} (n={row['Cluster0_N']})")
            print(f"  Cluster 2: Δz={row['Cluster1_Mean']:.3f}±{row['Cluster1_Std']:.3f} (n={row['Cluster1_N']})")
    else:
        print("\nNo significant bimodal distributions detected")
        bimodal_df = pd.DataFrame()

    return bimodal_df


def ml_feature_importance(df):
    """
    Task 6: Use Random Forest to find hidden variables - combinations
    of proteins that predict aging better than individual ones
    """
    print("\n" + "="*80)
    print("TASK 6: MACHINE LEARNING - HIDDEN VARIABLE DETECTION")
    print("="*80)

    # Build feature matrix: proteins as features, studies as samples
    pivot = df.pivot_table(
        index=['Study_ID', 'Tissue_Compartment'],
        columns='Gene_Symbol',
        values='Zscore_Delta',
        aggfunc='mean'
    )

    # Drop columns/rows with too many NaNs
    pivot = pivot.dropna(axis=1, thresh=int(0.3 * len(pivot)))  # Keep proteins present in >30% studies
    pivot = pivot.fillna(0)  # Impute remaining NaNs with 0

    if pivot.shape[1] < 5:
        print("\nInsufficient data for ML analysis")
        return pd.DataFrame()

    print(f"\nFeature matrix: {pivot.shape[0]} samples × {pivot.shape[1]} protein features")

    # Create synthetic "aging intensity" target: mean absolute z-score change
    y = pivot.abs().mean(axis=1).values
    X = pivot.values

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Protein': pivot.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\nRandom Forest R² score: {rf.score(X, y):.3f}")
    print("\nTop 20 most important proteins for predicting aging intensity:")

    for idx, row in feature_importance.head(20).iterrows():
        print(f"{row['Protein']:20s} | Importance: {row['Importance']:.4f}")

    # Identify protein pairs with high combined importance
    top_proteins = feature_importance.head(30)['Protein'].tolist()

    print("\n\nAnalyzing protein combinations (pairwise interactions)...")

    # Test top pairs
    pair_scores = []
    for p1, p2 in list(combinations(top_proteins[:15], 2))[:50]:  # Test top 50 pairs
        if p1 in pivot.columns and p2 in pivot.columns:
            # Create interaction term
            X_pair = pivot[[p1, p2]].values
            X_interact = np.column_stack([X_pair, X_pair[:, 0] * X_pair[:, 1]])

            # Fit simple model
            lr = LinearRegression()
            lr.fit(X_interact, y)
            score = lr.score(X_interact, y)

            pair_scores.append({
                'Protein_1': p1,
                'Protein_2': p2,
                'R2_Score': score,
                'Importance_1': feature_importance[feature_importance['Protein'] == p1]['Importance'].iloc[0],
                'Importance_2': feature_importance[feature_importance['Protein'] == p2]['Importance'].iloc[0]
            })

    pairs_df = pd.DataFrame(pair_scores).sort_values('R2_Score', ascending=False)

    print("\nTop 10 protein pairs (synergistic predictors):")
    for idx, row in pairs_df.head(10).iterrows():
        print(f"\n{row['Protein_1']} + {row['Protein_2']}")
        print(f"  Combined R²: {row['R2_Score']:.3f}")
        print(f"  Individual importance: {row['Importance_1']:.4f}, {row['Importance_2']:.4f}")

    return feature_importance, pairs_df


def main():
    """Main analysis pipeline"""

    print("\n" + "="*80)
    print("AGENT 3: NON-LINEAR PATTERN DETECTOR")
    print("="*80)
    print("\nMission: Find non-linear, threshold, and phase-transition patterns")
    print("that linear analysis would miss.\n")

    # Load data
    dataset_path = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
    df, valid_df = load_unified_dataset(dataset_path)

    # Task 1: Non-monotonic trajectories
    nonmonotonic_df = detect_nonmonotonic_trajectories(valid_df)

    # Task 2: Protein interactions
    interactions_df = detect_protein_interactions(valid_df)

    # Task 3: Switch points
    switch_df = detect_switch_points(valid_df)

    # Task 4: Bimodal distributions
    bimodal_df = detect_bimodal_distributions(valid_df)

    # Task 6: ML feature importance
    feature_importance_df, pairs_df = ml_feature_importance(valid_df)

    # Save results
    output_dir = "/Users/Kravtsovd/projects/ecm-atlas/10_insights"

    if len(nonmonotonic_df) > 0:
        nonmonotonic_df.to_csv(f"{output_dir}/nonlinear_trajectories.csv", index=False)
        print(f"\n✅ Saved: {output_dir}/nonlinear_trajectories.csv")

    if len(interactions_df) > 0:
        interactions_df.to_csv(f"{output_dir}/protein_interactions.csv", index=False)
        print(f"✅ Saved: {output_dir}/protein_interactions.csv")

    if len(switch_df) > 0:
        switch_df.to_csv(f"{output_dir}/switch_points.csv", index=False)
        print(f"✅ Saved: {output_dir}/switch_points.csv")

    if len(bimodal_df) > 0:
        bimodal_df.to_csv(f"{output_dir}/bimodal_proteins.csv", index=False)
        print(f"✅ Saved: {output_dir}/bimodal_proteins.csv")

    if len(feature_importance_df) > 0:
        feature_importance_df.to_csv(f"{output_dir}/ml_feature_importance.csv", index=False)
        print(f"✅ Saved: {output_dir}/ml_feature_importance.csv")

    if len(pairs_df) > 0:
        pairs_df.to_csv(f"{output_dir}/synergistic_protein_pairs.csv", index=False)
        print(f"✅ Saved: {output_dir}/synergistic_protein_pairs.csv")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext step: Run script to generate markdown report")
    print("Command: python /Users/Kravtsovd/projects/ecm-atlas/scripts/detect_nonlinear_patterns.py")

    return {
        'nonmonotonic': nonmonotonic_df,
        'interactions': interactions_df,
        'switch_points': switch_df,
        'bimodal': bimodal_df,
        'feature_importance': feature_importance_df,
        'pairs': pairs_df
    }


if __name__ == "__main__":
    results = main()
