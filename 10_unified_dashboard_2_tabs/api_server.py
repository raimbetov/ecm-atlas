#!/usr/bin/env python3
"""
Unified Flask API Server for ECM Atlas Dashboard
Supports both individual dataset analysis and cross-dataset comparison
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Helper function to convert pandas values to JSON-safe format
def to_json_safe(value):
    """Convert pandas/numpy values to JSON-safe format (replace NaN with None)"""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value) if isinstance(value, np.floating) else int(value)
    return value

def series_to_json_safe(series):
    """Convert pandas Series to JSON-safe list"""
    return [to_json_safe(x) for x in series]

# Load merged data on startup
print("Loading ECM Atlas data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} entries ({df['Protein_ID'].nunique()} unique proteins)")

# ===== GLOBAL ENDPOINTS =====

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "ECM Atlas Unified API Server",
        "port": 5004
    })

@app.route('/api/global_stats')
def get_global_stats():
    """Get overall statistics across all datasets"""
    return jsonify({
        "total_proteins": int(df['Protein_ID'].nunique()),
        "total_entries": len(df),
        "datasets": int(df['Dataset_Name'].nunique()),
        "organs": int(df['Organ'].nunique()),
        "compartments": int(df['Compartment'].nunique()),
        "ecm_proteins": int(df[df['Matrisome_Category'].notna()]['Protein_ID'].nunique())
    })

@app.route('/api/datasets')
def get_datasets():
    """Get list of available datasets with metadata"""
    datasets = []

    for dataset_name in df['Dataset_Name'].unique():
        dataset_df = df[df['Dataset_Name'] == dataset_name]
        organ = dataset_df['Organ'].iloc[0]
        compartments = dataset_df['Compartment'].unique().tolist()

        datasets.append({
            "name": dataset_name,
            "display_name": dataset_name.replace('_', ' '),
            "organ": organ,
            "compartments": compartments,
            "protein_count": int(dataset_df['Protein_ID'].nunique()),
            "ecm_count": int(dataset_df[dataset_df['Matrisome_Category'].notna()]['Protein_ID'].nunique())
        })

    return jsonify({"datasets": sorted(datasets, key=lambda x: x['name'])})

# ===== INDIVIDUAL DATASET ENDPOINTS =====

@app.route('/api/dataset/<dataset_name>/stats')
def get_dataset_stats(dataset_name):
    """Get statistics for a specific dataset"""
    dataset_df = df[df['Dataset_Name'] == dataset_name]

    if len(dataset_df) == 0:
        return jsonify({"error": "Dataset not found"}), 404

    compartments = dataset_df['Compartment'].unique()
    stats = {
        "dataset_name": dataset_name,
        "organ": dataset_df['Organ'].iloc[0],
        "total_proteins": int(dataset_df['Protein_ID'].nunique()),
        "ecm_proteins": int(dataset_df[dataset_df['Matrisome_Category'].notna()]['Protein_ID'].nunique()),
        "compartments": {}
    }

    for comp in compartments:
        comp_df = dataset_df[dataset_df['Compartment'] == comp]
        stats["compartments"][comp] = {
            "protein_count": int(comp_df['Protein_ID'].nunique()),
            "ecm_count": int(comp_df[comp_df['Matrisome_Category'].notna()]['Protein_ID'].nunique())
        }

    return jsonify(stats)

@app.route('/api/dataset/<dataset_name>/heatmap/<compartment>')
def get_dataset_heatmap(dataset_name, compartment):
    """Get heatmap data for a specific dataset and compartment"""
    n = int(request.args.get('n', 100))

    dataset_df = df[(df['Dataset_Name'] == dataset_name) & (df['Compartment'] == compartment)]

    if len(dataset_df) == 0:
        return jsonify({"error": "Dataset/compartment not found"}), 404

    # Get top N proteins with largest absolute Zscore_Delta
    dataset_df = dataset_df.copy()
    dataset_df['Abs_Zscore_Delta'] = dataset_df['Zscore_Delta'].abs()
    top_df = dataset_df.nlargest(n, 'Abs_Zscore_Delta').sort_values('Zscore_Delta', ascending=False)

    return jsonify({
        "genes": top_df['Gene_Symbol'].tolist(),
        "zscore_young": series_to_json_safe(top_df['Zscore_Young']),
        "zscore_old": series_to_json_safe(top_df['Zscore_Old']),
        "dataset": dataset_name,
        "compartment": compartment
    })

@app.route('/api/dataset/<dataset_name>/volcano/<compartment>')
def get_dataset_volcano(dataset_name, compartment):
    """Get volcano plot data"""
    dataset_df = df[(df['Dataset_Name'] == dataset_name) & (df['Compartment'] == compartment)]

    if len(dataset_df) == 0:
        return jsonify({"error": "Dataset/compartment not found"}), 404

    dataset_df = dataset_df.copy()
    dataset_df['Avg_Abundance'] = (dataset_df['Abundance_Young'] + dataset_df['Abundance_Old']) / 2
    dataset_df['NegLog_Abundance'] = -np.log10(dataset_df['Avg_Abundance'] + 1)

    return jsonify({
        "genes": dataset_df['Gene_Symbol'].tolist(),
        "zscore_delta": series_to_json_safe(dataset_df['Zscore_Delta']),
        "neglog_abundance": series_to_json_safe(dataset_df['NegLog_Abundance']),
        "zscore_young": series_to_json_safe(dataset_df['Zscore_Young']),
        "zscore_old": series_to_json_safe(dataset_df['Zscore_Old'])
    })

@app.route('/api/dataset/<dataset_name>/scatter/<compartment>')
def get_dataset_scatter(dataset_name, compartment):
    """Get scatter plot data"""
    dataset_df = df[(df['Dataset_Name'] == dataset_name) & (df['Compartment'] == compartment)]

    if len(dataset_df) == 0:
        return jsonify({"error": "Dataset/compartment not found"}), 404

    return jsonify({
        "genes": dataset_df['Gene_Symbol'].tolist(),
        "zscore_young": series_to_json_safe(dataset_df['Zscore_Young']),
        "zscore_old": series_to_json_safe(dataset_df['Zscore_Old']),
        "zscore_delta": series_to_json_safe(dataset_df['Zscore_Delta']),
        "is_ecm": dataset_df['Matrisome_Category'].notna().tolist(),
        "matrisome_category": dataset_df['Matrisome_Category'].fillna('Non-ECM').tolist()
    })

@app.route('/api/dataset/<dataset_name>/bars/<compartment>')
def get_dataset_bars(dataset_name, compartment):
    """Get top 20 aging markers for bar chart"""
    dataset_df = df[(df['Dataset_Name'] == dataset_name) & (df['Compartment'] == compartment)]

    if len(dataset_df) == 0:
        return jsonify({"error": "Dataset/compartment not found"}), 404

    top_increases = dataset_df.nlargest(10, 'Zscore_Delta')
    top_decreases = dataset_df.nsmallest(10, 'Zscore_Delta')
    top_combined = pd.concat([top_increases, top_decreases]).sort_values('Zscore_Delta')

    return jsonify({
        "genes": top_combined['Gene_Symbol'].tolist(),
        "zscore_delta": series_to_json_safe(top_combined['Zscore_Delta']),
        "matrisome_category": top_combined['Matrisome_Category'].fillna('Non-ECM').tolist()
    })

@app.route('/api/dataset/<dataset_name>/histogram/<compartment>')
def get_dataset_histogram(dataset_name, compartment):
    """Get histogram data"""
    dataset_df = df[(df['Dataset_Name'] == dataset_name) & (df['Compartment'] == compartment)]

    if len(dataset_df) == 0:
        return jsonify({"error": "Dataset/compartment not found"}), 404

    return jsonify({
        "zscore_delta": series_to_json_safe(dataset_df['Zscore_Delta']),
        "mean_delta": to_json_safe(dataset_df['Zscore_Delta'].mean())
    })

@app.route('/api/dataset/<dataset_name>/comparison')
def get_dataset_comparison(dataset_name):
    """Get compartment comparison data for datasets with multiple compartments"""
    dataset_df = df[df['Dataset_Name'] == dataset_name]

    if len(dataset_df) == 0:
        return jsonify({"error": "Dataset not found"}), 404

    compartments = dataset_df['Compartment'].unique()

    if len(compartments) < 2:
        return jsonify({"error": "Dataset has only one compartment"}), 400

    # For simplicity, compare first two compartments
    comp1, comp2 = compartments[0], compartments[1]

    df1 = dataset_df[dataset_df['Compartment'] == comp1][['Gene_Symbol', 'Zscore_Delta', 'Matrisome_Category']]
    df2 = dataset_df[dataset_df['Compartment'] == comp2][['Gene_Symbol', 'Zscore_Delta']]

    merged = pd.merge(df1, df2, on='Gene_Symbol', suffixes=(f'_{comp1}', f'_{comp2}'))

    return jsonify({
        "compartment1": comp1,
        "compartment2": comp2,
        "genes": merged['Gene_Symbol'].tolist(),
        "zscore_delta_comp1": series_to_json_safe(merged[f'Zscore_Delta_{comp1}']),
        "zscore_delta_comp2": series_to_json_safe(merged[f'Zscore_Delta_{comp2}']),
        "matrisome_category": merged['Matrisome_Category'].fillna('Non-ECM').tolist()
    })

# ===== CROSS-DATASET COMPARISON ENDPOINTS =====

@app.route('/api/compare/filters')
def get_compare_filters():
    """Get available filter options for cross-dataset comparison"""

    organs = []
    for organ in df['Organ'].unique():
        organs.append({
            "name": organ,
            "count": int(df[df['Organ'] == organ]['Protein_ID'].nunique())
        })

    compartments = []
    for compartment in df['Compartment'].unique():
        compartment_data = df[df['Compartment'] == compartment]
        compartments.append({
            "name": compartment,
            "count": int(compartment_data['Protein_ID'].nunique()),
            "organ": compartment_data['Organ'].iloc[0]
        })

    categories = []
    for category in df['Matrisome_Category'].dropna().unique():
        categories.append({
            "name": category,
            "count": int(df[df['Matrisome_Category'] == category]['Protein_ID'].nunique())
        })

    studies = []
    for study in df['Dataset_Name'].unique():
        studies.append({
            "name": study,
            "count": int(df[df['Dataset_Name'] == study]['Protein_ID'].nunique())
        })

    return jsonify({
        "organs": sorted(organs, key=lambda x: x['name']),
        "compartments": sorted(compartments, key=lambda x: x['name']),
        "categories": sorted(categories, key=lambda x: -x['count']),
        "studies": sorted(studies, key=lambda x: x['name'])
    })

@app.route('/api/compare/heatmap')
def get_compare_heatmap():
    """Get heatmap data for cross-dataset comparison with filters"""

    # Apply filters
    filtered_df = df.copy()

    if 'organs' in request.args:
        organs = request.args.get('organs').split(',')
        filtered_df = filtered_df[filtered_df['Organ'].isin(organs)]

    if 'compartments' in request.args:
        compartments = request.args.get('compartments').split(',')
        filtered_df = filtered_df[filtered_df['Compartment'].isin(compartments)]

    if 'categories' in request.args:
        categories = request.args.get('categories').split(',')
        filtered_df = filtered_df[filtered_df['Matrisome_Category'].isin(categories)]

    if 'studies' in request.args:
        studies = request.args.get('studies').split(',')
        filtered_df = filtered_df[filtered_df['Dataset_Name'].isin(studies)]

    if 'trend' in request.args:
        trend = request.args.get('trend')
        if trend == 'up':
            filtered_df = filtered_df[filtered_df['Zscore_Delta'] > 0.5]
        elif trend == 'down':
            filtered_df = filtered_df[filtered_df['Zscore_Delta'] < -0.5]
        elif trend == 'stable':
            filtered_df = filtered_df[filtered_df['Zscore_Delta'].abs() <= 0.5]

    if 'search' in request.args:
        search_query = request.args.get('search').lower()
        filtered_df = filtered_df[
            filtered_df['Gene_Symbol'].str.lower().str.contains(search_query, na=False) |
            filtered_df['Protein_ID'].str.lower().str.contains(search_query, na=False) |
            filtered_df['Protein_Name'].str.lower().str.contains(search_query, na=False)
        ]

    # Build heatmap data structure
    compartments = sorted(filtered_df['Compartment'].unique())
    proteins = filtered_df['Gene_Symbol'].unique()

    heatmap_data = {}
    protein_metadata = {}

    for protein in proteins:
        protein_data = filtered_df[filtered_df['Gene_Symbol'] == protein]
        protein_id = protein_data.iloc[0]['Protein_ID']

        protein_metadata[protein] = {
            "protein_id": protein_id,
            "protein_name": protein_data.iloc[0]['Protein_Name'],
            "matrisome_category": protein_data.iloc[0]['Matrisome_Category'] if pd.notna(protein_data.iloc[0]['Matrisome_Category']) else None
        }

        heatmap_data[protein] = {}
        for compartment in compartments:
            compartment_data = protein_data[protein_data['Compartment'] == compartment]

            if len(compartment_data) > 0:
                row = compartment_data.iloc[0]
                heatmap_data[protein][compartment] = {
                    "zscore_delta": float(row['Zscore_Delta']) if pd.notna(row['Zscore_Delta']) else None,
                    "zscore_young": float(row['Zscore_Young']) if pd.notna(row['Zscore_Young']) else None,
                    "zscore_old": float(row['Zscore_Old']) if pd.notna(row['Zscore_Old']) else None,
                    "dataset": row['Dataset_Name'],
                    "organ": row['Organ']
                }
            else:
                heatmap_data[protein][compartment] = None

    return jsonify({
        "proteins": list(proteins),
        "compartments": compartments,
        "data": heatmap_data,
        "metadata": protein_metadata,
        "summary": {
            "total_proteins": len(proteins),
            "total_compartments": len(compartments),
            "avg_zscore_delta": float(filtered_df['Zscore_Delta'].mean()) if len(filtered_df) > 0 else 0
        }
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ECM Atlas - Unified API Server")
    print("="*70)
    print("\nGlobal endpoints:")
    print("  GET /api/health - Health check")
    print("  GET /api/global_stats - Overall statistics")
    print("  GET /api/datasets - List all datasets")
    print("\nIndividual dataset endpoints:")
    print("  GET /api/dataset/<name>/stats - Dataset statistics")
    print("  GET /api/dataset/<name>/heatmap/<compartment>?n=100")
    print("  GET /api/dataset/<name>/volcano/<compartment>")
    print("  GET /api/dataset/<name>/scatter/<compartment>")
    print("  GET /api/dataset/<name>/bars/<compartment>")
    print("  GET /api/dataset/<name>/histogram/<compartment>")
    print("  GET /api/dataset/<name>/comparison")
    print("\nCross-dataset comparison endpoints:")
    print("  GET /api/compare/filters - Filter options")
    print("  GET /api/compare/heatmap?[filters] - Comparison heatmap")
    print("\nData loaded:")
    print(f"  {df['Protein_ID'].nunique()} unique proteins")
    print(f"  {len(df)} total entries")
    print(f"  {df['Dataset_Name'].nunique()} datasets")
    print(f"  {df['Organ'].nunique()} organs")
    print(f"  {df['Compartment'].nunique()} compartments")
    print("\nStarting server on http://localhost:5004")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=5004, debug=False)
