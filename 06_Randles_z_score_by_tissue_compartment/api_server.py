#!/usr/bin/env python3
"""
Flask API Server for Z-Score Dashboard
Provides REST API endpoints to query z-score data from CSV files
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load data on startup
print("Loading z-score data...")
df_glom = pd.read_csv("claude_code/Randles_2021_Glomerular_zscore.csv")
df_tubu = pd.read_csv("claude_code/Randles_2021_Tubulointerstitial_zscore.csv")
print(f"Loaded: {len(df_glom)} Glomerular proteins, {len(df_tubu)} Tubulointerstitial proteins")

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Z-Score API Server"})

@app.route('/api/stats')
def get_stats():
    """Get summary statistics"""
    return jsonify({
        "glomerular_proteins": len(df_glom),
        "tubulointerstitial_proteins": len(df_tubu),
        "glomerular_ecm": int(df_glom['Matrisome_Category'].notna().sum()),
        "tubulointerstitial_ecm": int(df_tubu['Matrisome_Category'].notna().sum()),
        "total_proteins": len(df_glom) + len(df_tubu)
    })

@app.route('/api/heatmap/<compartment>')
def get_heatmap_data(compartment):
    """Get heatmap data for specified compartment"""
    df = df_glom if compartment == 'glomerular' else df_tubu

    # Get top N proteins with largest absolute Zscore_Delta
    n = int(request.args.get('n', 100))

    df_copy = df.copy()
    df_copy['Abs_Zscore_Delta'] = df_copy['Zscore_Delta'].abs()
    df_top = df_copy.nlargest(n, 'Abs_Zscore_Delta').sort_values('Zscore_Delta', ascending=False)

    return jsonify({
        "genes": df_top['Gene_Symbol'].tolist(),
        "zscore_young": df_top['Zscore_Young'].tolist(),
        "zscore_old": df_top['Zscore_Old'].tolist(),
        "compartment": compartment.capitalize()
    })

@app.route('/api/volcano/<compartment>')
def get_volcano_data(compartment):
    """Get volcano plot data"""
    df = df_glom if compartment == 'glomerular' else df_tubu

    df_copy = df.copy()
    df_copy['Avg_Abundance'] = (df_copy['Abundance_Young'] + df_copy['Abundance_Old']) / 2
    df_copy['NegLog_Abundance'] = -np.log10(df_copy['Avg_Abundance'] + 1)

    return jsonify({
        "genes": df_copy['Gene_Symbol'].tolist(),
        "zscore_delta": df_copy['Zscore_Delta'].tolist(),
        "neglog_abundance": df_copy['NegLog_Abundance'].tolist(),
        "zscore_young": df_copy['Zscore_Young'].tolist(),
        "zscore_old": df_copy['Zscore_Old'].tolist()
    })

@app.route('/api/scatter/<compartment>')
def get_scatter_data(compartment):
    """Get scatter plot data"""
    df = df_glom if compartment == 'glomerular' else df_tubu

    return jsonify({
        "genes": df['Gene_Symbol'].tolist(),
        "zscore_young": df['Zscore_Young'].tolist(),
        "zscore_old": df['Zscore_Old'].tolist(),
        "zscore_delta": df['Zscore_Delta'].tolist(),
        "is_ecm": df['Matrisome_Category'].notna().tolist(),
        "matrisome_category": df['Matrisome_Category'].fillna('Non-ECM').tolist()
    })

@app.route('/api/bars/<compartment>')
def get_bar_data(compartment):
    """Get top 20 aging markers for bar chart"""
    df = df_glom if compartment == 'glomerular' else df_tubu

    top_increases = df.nlargest(10, 'Zscore_Delta')
    top_decreases = df.nsmallest(10, 'Zscore_Delta')

    top_combined = pd.concat([top_increases, top_decreases]).sort_values('Zscore_Delta')

    return jsonify({
        "genes": top_combined['Gene_Symbol'].tolist(),
        "zscore_delta": top_combined['Zscore_Delta'].tolist(),
        "matrisome_category": top_combined['Matrisome_Category'].fillna('Non-ECM').tolist()
    })

@app.route('/api/histogram/<compartment>')
def get_histogram_data(compartment):
    """Get histogram data"""
    df = df_glom if compartment == 'glomerular' else df_tubu

    return jsonify({
        "zscore_delta": df['Zscore_Delta'].tolist(),
        "mean_delta": float(df['Zscore_Delta'].mean())
    })

@app.route('/api/comparison')
def get_comparison_data():
    """Get compartment comparison data"""

    # Merge on Gene_Symbol
    df_merged = pd.merge(
        df_glom[['Gene_Symbol', 'Zscore_Delta', 'Matrisome_Category']],
        df_tubu[['Gene_Symbol', 'Zscore_Delta']],
        on='Gene_Symbol',
        suffixes=('_Glom', '_Tubu')
    )

    return jsonify({
        "genes": df_merged['Gene_Symbol'].tolist(),
        "zscore_delta_glom": df_merged['Zscore_Delta_Glom'].tolist(),
        "zscore_delta_tubu": df_merged['Zscore_Delta_Tubu'].tolist(),
        "matrisome_category": df_merged['Matrisome_Category'].fillna('Non-ECM').tolist()
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Z-Score Dashboard API Server")
    print("="*60)
    print("\nAvailable endpoints:")
    print("  GET /api/health - Health check")
    print("  GET /api/stats - Summary statistics")
    print("  GET /api/heatmap/<compartment>?n=100 - Heatmap data")
    print("  GET /api/volcano/<compartment> - Volcano plot data")
    print("  GET /api/scatter/<compartment> - Scatter plot data")
    print("  GET /api/bars/<compartment> - Top 20 aging markers")
    print("  GET /api/histogram/<compartment> - Histogram data")
    print("  GET /api/comparison - Compartment comparison data")
    print("\nStarting server on http://localhost:5001")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5002, debug=False)
