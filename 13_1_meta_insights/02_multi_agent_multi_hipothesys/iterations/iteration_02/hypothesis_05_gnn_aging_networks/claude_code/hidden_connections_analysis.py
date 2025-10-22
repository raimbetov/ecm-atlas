#!/usr/bin/env python3
"""
Hidden Connections and Therapeutic Target Analysis
Identifies non-obvious protein relationships discovered by GNN
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine, euclidean
from pathlib import Path

# Paths
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_02/hypothesis_05_gnn_aging_networks/claude_code")
DATA_PATH = Path("/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv")

print("=" * 80)
print("HIDDEN CONNECTIONS & THERAPEUTIC TARGET ANALYSIS")
print("=" * 80)

# Load data
print("\n[1/4] Loading data and embeddings...")
df = pd.read_csv(DATA_PATH)
embeddings_df = pd.read_csv(OUTPUT_DIR / 'protein_embeddings_gnn_claude_code.csv', index_col=0)
master_regs = pd.read_csv(OUTPUT_DIR / 'master_regulators_claude_code.csv')

proteins = embeddings_df.index.tolist()
embeddings = embeddings_df.values

# Rebuild correlation matrix
pivot = df.pivot_table(values='Zscore_Delta', index='Gene_Symbol', columns='Tissue', aggfunc='mean')
pivot = pivot.loc[proteins]  # Align with embeddings
pivot_filled = pivot.fillna(0)
X_raw = pivot_filled.values

corr_matrix, _ = spearmanr(X_raw, axis=1, nan_policy='omit')
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

# ============================================================================
# 1. HIDDEN CONNECTIONS (Low correlation, High GNN similarity)
# ============================================================================
print("\n[2/4] Identifying hidden connections...")
hidden_connections = []

# Compute GNN embedding similarity (cosine similarity)
gnn_similarity = np.zeros_like(corr_matrix)
for i in range(len(proteins)):
    for j in range(i+1, len(proteins)):
        sim = 1 - cosine(embeddings[i], embeddings[j])
        gnn_similarity[i, j] = sim
        gnn_similarity[j, i] = sim

# Find pairs with LOW correlation but HIGH GNN similarity
correlation_threshold = 0.3  # Low correlation
gnn_sim_threshold = 0.85  # High GNN similarity

for i in range(len(proteins)):
    for j in range(i+1, len(proteins)):
        corr_val = abs(corr_matrix[i, j])
        gnn_sim = gnn_similarity[i, j]

        if corr_val < correlation_threshold and gnn_sim > gnn_sim_threshold:
            # Get Matrisome categories
            cat_i = master_regs[master_regs['Gene_Symbol'] == proteins[i]]['Matrisome_Category'].values
            cat_j = master_regs[master_regs['Gene_Symbol'] == proteins[j]]['Matrisome_Category'].values
            cat_i = cat_i[0] if len(cat_i) > 0 else 'Unknown'
            cat_j = cat_j[0] if len(cat_j) > 0 else 'Unknown'

            hidden_connections.append({
                'Protein_A': proteins[i],
                'Protein_B': proteins[j],
                'Correlation': corr_val,
                'GNN_Similarity': gnn_sim,
                'Similarity_Gap': gnn_sim - corr_val,
                'Category_A': cat_i,
                'Category_B': cat_j
            })

hidden_df = pd.DataFrame(hidden_connections)
if len(hidden_df) > 0:
    hidden_df = hidden_df.sort_values('Similarity_Gap', ascending=False)
    hidden_df.to_csv(OUTPUT_DIR / 'hidden_connections_claude_code.csv', index=False)
    print(f"Found {len(hidden_df)} hidden connections")
    print("\nTop 5 Hidden Connections:")
    print(hidden_df.head()[['Protein_A', 'Protein_B', 'Correlation', 'GNN_Similarity']])
else:
    print("No hidden connections found with current thresholds")
    # Create empty file
    pd.DataFrame(columns=['Protein_A', 'Protein_B', 'Correlation', 'GNN_Similarity']).to_csv(
        OUTPUT_DIR / 'hidden_connections_claude_code.csv', index=False
    )

# ============================================================================
# 2. LINK PREDICTION (Future co-dysregulation)
# ============================================================================
print("\n[3/4] Predicting future co-dysregulation...")
link_predictions = []

# Predict links based on high GNN similarity but currently not strongly correlated
link_pred_corr_threshold = 0.4
link_pred_gnn_threshold = 0.80

for i in range(len(proteins)):
    for j in range(i+1, len(proteins)):
        corr_val = abs(corr_matrix[i, j])
        gnn_sim = gnn_similarity[i, j]

        if corr_val < link_pred_corr_threshold and gnn_sim > link_pred_gnn_threshold:
            # Get delta z values
            delta_z_i = X_raw[i].mean()
            delta_z_j = X_raw[j].mean()

            link_predictions.append({
                'Protein_A': proteins[i],
                'Protein_B': proteins[j],
                'Current_Correlation': corr_val,
                'GNN_Similarity': gnn_sim,
                'Prediction_Confidence': gnn_sim - corr_val,
                'Delta_Z_A': delta_z_i,
                'Delta_Z_B': delta_z_j,
                'Both_Upregulated': (delta_z_i > 0.5 and delta_z_j > 0.5),
                'Both_Downregulated': (delta_z_i < -0.5 and delta_z_j < -0.5)
            })

link_pred_df = pd.DataFrame(link_predictions)
if len(link_pred_df) > 0:
    link_pred_df = link_pred_df.sort_values('Prediction_Confidence', ascending=False)
    link_pred_df.to_csv(OUTPUT_DIR / 'link_prediction_claude_code.csv', index=False)
    print(f"Generated {len(link_pred_df)} link predictions")
    print("\nTop 5 Link Predictions:")
    print(link_pred_df.head()[['Protein_A', 'Protein_B', 'Current_Correlation', 'GNN_Similarity']])
else:
    print("No link predictions with current thresholds")
    pd.DataFrame(columns=['Protein_A', 'Protein_B', 'Current_Correlation', 'GNN_Similarity']).to_csv(
        OUTPUT_DIR / 'link_prediction_claude_code.csv', index=False
    )

# ============================================================================
# 3. THERAPEUTIC TARGET RANKING
# ============================================================================
print("\n[4/4] Ranking therapeutic targets...")

# Load perturbation analysis
perturbation_df = pd.read_csv(OUTPUT_DIR / 'perturbation_analysis_claude_code.csv')

# Known druggable protein families (simplified)
druggable_keywords = [
    'SERPIN', 'MMP', 'TIMP', 'ADAM', 'LOX', 'PLOD', 'P4H',
    'COL', 'FN', 'VTN', 'THBS', 'TNC', 'SPARC'
]

therapeutic_targets = []
for idx, row in master_regs.head(20).iterrows():  # Top 20 master regulators
    gene = row['Gene_Symbol']
    score = row['Combined_Score']

    # Check if druggable
    is_druggable = any(keyword in gene.upper() for keyword in druggable_keywords)

    # Get perturbation impact
    pert_row = perturbation_df[perturbation_df['Master_Regulator'] == gene]
    if len(pert_row) > 0:
        impact = pert_row['Affected_Percentage'].values[0]
        max_shift = pert_row['Max_Embedding_Shift'].values[0]
    else:
        impact = 0
        max_shift = 0

    # Therapeutic potential score
    # Higher = more druggable + higher impact
    therapeutic_score = (
        score * 0.4 +  # Master regulator score
        (impact / 100) * 0.3 +  # Network impact
        (1.0 if is_druggable else 0.5) * 0.3  # Druggability
    )

    therapeutic_targets.append({
        'Gene_Symbol': gene,
        'Master_Regulator_Score': score,
        'Network_Impact_Pct': impact,
        'Max_Embedding_Shift': max_shift,
        'Is_Druggable': is_druggable,
        'Therapeutic_Score': therapeutic_score,
        'Matrisome_Category': row['Matrisome_Category'],
        'Delta_Z_Mean': row['Delta_Z_Mean'],
        'Recommendation': 'High Priority' if therapeutic_score > 0.7 else 'Medium Priority' if therapeutic_score > 0.5 else 'Low Priority'
    })

therapeutic_df = pd.DataFrame(therapeutic_targets)
therapeutic_df = therapeutic_df.sort_values('Therapeutic_Score', ascending=False)
therapeutic_df.to_csv(OUTPUT_DIR / 'therapeutic_ranking_claude_code.csv', index=False)

print("\nTop 10 Therapeutic Targets:")
print(therapeutic_df.head(10)[['Gene_Symbol', 'Therapeutic_Score', 'Is_Druggable', 'Recommendation']])

print("\n" + "=" * 80)
print("HIDDEN CONNECTIONS ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nOutputs:")
print(f"  - hidden_connections_claude_code.csv: {len(hidden_df) if len(hidden_df) > 0 else 0} connections")
print(f"  - link_prediction_claude_code.csv: {len(link_pred_df) if len(link_pred_df) > 0 else 0} predictions")
print(f"  - therapeutic_ranking_claude_code.csv: {len(therapeutic_df)} targets")
