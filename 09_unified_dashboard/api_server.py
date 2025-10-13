#!/usr/bin/env python3
"""
Flask API Server for Unified Multi-Tissue ECM Dashboard
Provides REST API endpoints for merged ECM aging data
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load merged data on startup
print("Loading merged ECM aging data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} entries ({df['Protein_ID'].nunique()} unique proteins)")

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Unified ECM Dashboard API Server",
        "port": 5002
    })

@app.route('/api/stats')
def get_stats():
    """Get overall statistics"""
    return jsonify({
        "total_proteins": int(df['Protein_ID'].nunique()),
        "total_entries": len(df),
        "multi_compartment_proteins": int(df.groupby('Protein_ID')['Compartment'].nunique().gt(1).sum()),
        "organs": int(df['Organ'].nunique()),
        "compartments": int(df['Compartment'].nunique()),
        "studies": int(df['Dataset_Name'].nunique()),
        "avg_zscore_delta": float(df['Zscore_Delta'].mean()),
        "std_zscore_delta": float(df['Zscore_Delta'].std())
    })

@app.route('/api/filters')
def get_filters():
    """Get available filter options with counts"""

    # Organs
    organs = []
    for organ in df['Organ'].unique():
        organs.append({
            "name": organ,
            "count": int(df[df['Organ'] == organ]['Protein_ID'].nunique())
        })

    # Compartments
    compartments = []
    for compartment in df['Compartment'].unique():
        compartment_data = df[df['Compartment'] == compartment]
        compartments.append({
            "name": compartment,
            "count": int(compartment_data['Protein_ID'].nunique()),
            "organ": compartment_data['Organ'].iloc[0]
        })

    # Matrisome categories
    categories = []
    for category in df['Matrisome_Category'].dropna().unique():
        categories.append({
            "name": category,
            "count": int(df[df['Matrisome_Category'] == category]['Protein_ID'].nunique())
        })

    # Studies
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

@app.route('/api/proteins')
def get_proteins():
    """Get list of all proteins with metadata"""

    # Group by protein and aggregate compartments
    protein_list = []
    for protein_id in df['Protein_ID'].unique():
        protein_data = df[df['Protein_ID'] == protein_id].iloc[0]
        compartments = df[df['Protein_ID'] == protein_id]['Compartment'].tolist()

        protein_list.append({
            "protein_id": protein_id,
            "gene_symbol": protein_data['Gene_Symbol'],
            "protein_name": protein_data['Protein_Name'],
            "matrisome_category": protein_data['Matrisome_Category'] if pd.notna(protein_data['Matrisome_Category']) else None,
            "compartments": compartments
        })

    return jsonify({"proteins": protein_list})

@app.route('/api/heatmap')
def get_heatmap():
    """Get heatmap data with optional filters"""

    # Apply filters
    filtered_df = df.copy()

    # Filter by organs
    if 'organs' in request.args:
        organs = request.args.get('organs').split(',')
        filtered_df = filtered_df[filtered_df['Organ'].isin(organs)]

    # Filter by compartments
    if 'compartments' in request.args:
        compartments = request.args.get('compartments').split(',')
        filtered_df = filtered_df[filtered_df['Compartment'].isin(compartments)]

    # Filter by categories
    if 'categories' in request.args:
        categories = request.args.get('categories').split(',')
        filtered_df = filtered_df[filtered_df['Matrisome_Category'].isin(categories)]

    # Filter by studies
    if 'studies' in request.args:
        studies = request.args.get('studies').split(',')
        filtered_df = filtered_df[filtered_df['Dataset_Name'].isin(studies)]

    # Filter by trend
    if 'trend' in request.args:
        trend = request.args.get('trend')
        if trend == 'up':
            filtered_df = filtered_df[filtered_df['Zscore_Delta'] > 0.5]
        elif trend == 'down':
            filtered_df = filtered_df[filtered_df['Zscore_Delta'] < -0.5]
        elif trend == 'stable':
            filtered_df = filtered_df[filtered_df['Zscore_Delta'].abs() <= 0.5]

    # Filter by search query
    if 'search' in request.args:
        search_query = request.args.get('search').lower()
        filtered_df = filtered_df[
            filtered_df['Gene_Symbol'].str.lower().str.contains(search_query, na=False) |
            filtered_df['Protein_ID'].str.lower().str.contains(search_query, na=False) |
            filtered_df['Protein_Name'].str.lower().str.contains(search_query, na=False)
        ]

    # Get unique compartments and proteins
    compartments = sorted(filtered_df['Compartment'].unique())
    proteins = filtered_df['Gene_Symbol'].unique()

    # Build heatmap data structure
    heatmap_data = {}
    protein_metadata = {}

    for protein in proteins:
        protein_data = filtered_df[filtered_df['Gene_Symbol'] == protein]
        protein_id = protein_data.iloc[0]['Protein_ID']

        # Store metadata
        protein_metadata[protein] = {
            "protein_id": protein_id,
            "protein_name": protein_data.iloc[0]['Protein_Name'],
            "matrisome_category": protein_data.iloc[0]['Matrisome_Category'] if pd.notna(protein_data.iloc[0]['Matrisome_Category']) else None
        }

        # Store compartment data
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

@app.route('/api/protein/<protein_id>')
def get_protein_details(protein_id):
    """Get detailed information for a specific protein"""

    protein_data = df[df['Protein_ID'] == protein_id]

    if len(protein_data) == 0:
        return jsonify({"error": "Protein not found"}), 404

    # Get basic info from first row
    first_row = protein_data.iloc[0]

    # Get compartment data
    compartments = []
    for _, row in protein_data.iterrows():
        compartments.append({
            "organ": row['Organ'],
            "compartment": row['Compartment'],
            "dataset": row['Dataset_Name'],
            "zscore_young": float(row['Zscore_Young']) if pd.notna(row['Zscore_Young']) else None,
            "zscore_old": float(row['Zscore_Old']) if pd.notna(row['Zscore_Old']) else None,
            "zscore_delta": float(row['Zscore_Delta']) if pd.notna(row['Zscore_Delta']) else None,
            "abundance_young": float(row['Abundance_Young']) if pd.notna(row['Abundance_Young']) else None,
            "abundance_old": float(row['Abundance_Old']) if pd.notna(row['Abundance_Old']) else None
        })

    return jsonify({
        "protein_id": protein_id,
        "gene_symbol": first_row['Gene_Symbol'],
        "protein_name": first_row['Protein_Name'],
        "matrisome_category": first_row['Matrisome_Category'] if pd.notna(first_row['Matrisome_Category']) else None,
        "matrisome_division": first_row['Matrisome_Division'] if pd.notna(first_row['Matrisome_Division']) else None,
        "compartments": compartments
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Unified Multi-Tissue ECM Dashboard API Server")
    print("="*70)
    print("\nAvailable endpoints:")
    print("  GET /api/health - Health check")
    print("  GET /api/stats - Overall statistics")
    print("  GET /api/filters - Available filter options")
    print("  GET /api/proteins - List all proteins")
    print("  GET /api/heatmap?[filters] - Heatmap data with filters")
    print("      Filters: organs, compartments, categories, studies, trend, search")
    print("  GET /api/protein/<protein_id> - Detailed protein information")
    print("\nData loaded:")
    print(f"  {df['Protein_ID'].nunique()} unique proteins")
    print(f"  {len(df)} total entries")
    print(f"  {df['Organ'].nunique()} organs: {', '.join(df['Organ'].unique())}")
    print(f"  {df['Compartment'].nunique()} compartments: {', '.join(df['Compartment'].unique())}")
    print("\nStarting server on http://localhost:5002")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=5002, debug=False)
