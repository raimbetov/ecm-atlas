#!/usr/bin/env python3
"""
Z-Score Visualization Dashboard Generator
Creates interactive HTML dashboard with multiple visualizations for Randles 2021 z-score data.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from datetime import datetime

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

print("Loading z-score normalized data...")

# Load both compartments
df_glom = pd.read_csv("claude_code/Randles_2021_Glomerular_zscore.csv")
df_tubu = pd.read_csv("claude_code/Randles_2021_Tubulointerstitial_zscore.csv")

print(f"Glomerular: {len(df_glom)} proteins")
print(f"Tubulointerstitial: {len(df_tubu)} proteins")

# Add compartment label for combined analysis
df_glom['Compartment'] = 'Glomerular'
df_tubu['Compartment'] = 'Tubulointerstitial'

# Combine for some visualizations
df_combined = pd.concat([df_glom, df_tubu], ignore_index=True)

# ==============================================================================
# 2. VISUALIZATION 1: HEATMAP - Z-Score Matrix with Color Gradient
# ==============================================================================

print("\n1. Creating heatmap with color gradient (-5 to +5)...")

def create_heatmap(df_comp, title, compartment_name):
    """Create heatmap for z-scores with color gradient."""

    # Select top 100 proteins with largest absolute Zscore_Delta for better visibility
    df_comp_copy = df_comp.copy()
    df_comp_copy['Abs_Zscore_Delta'] = df_comp_copy['Zscore_Delta'].abs()
    df_top = df_comp_copy.nlargest(100, 'Abs_Zscore_Delta', keep='all').sort_values('Zscore_Delta', ascending=False)

    # Prepare data matrix (proteins as rows, age groups as columns)
    z_matrix = df_top[['Zscore_Young', 'Zscore_Old']].values
    gene_labels = df_top['Gene_Symbol'].tolist()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        y=gene_labels,
        x=['Young', 'Old'],
        colorscale='RdBu_r',  # Red (high) to Blue (low)
        zmid=0,  # Center at 0
        zmin=-5,
        zmax=5,
        colorbar=dict(
            title=dict(text="Z-Score", side="right"),
            tickmode="linear",
            tick0=-5,
            dtick=1
        ),
        hovertemplate='Gene: %{y}<br>Age: %{x}<br>Z-Score: %{z:.2f}<extra></extra>',
        showscale=True
    ))

    fig.update_layout(
        title=f"{title}<br><sub>Top 100 proteins with largest aging changes</sub>",
        yaxis_title="Gene Symbol",
        xaxis_title="Age Group",
        height=1200,  # Tall enough to show 100 proteins
        yaxis={'tickfont': {'size': 8}},
        template='plotly_white',
        margin=dict(l=150, r=50, t=80, b=50)
    )

    return fig

# Create heatmaps for both compartments
fig_heatmap_glom = create_heatmap(df_glom, "Glomerular Z-Score Heatmap", "Glomerular")
fig_heatmap_tubu = create_heatmap(df_tubu, "Tubulointerstitial Z-Score Heatmap", "Tubulointerstitial")

# ==============================================================================
# 3. VISUALIZATION 2: VOLCANO PLOT - Differential Expression
# ==============================================================================

print("2. Creating volcano plots (Zscore_Delta vs Abundance)...")

def create_volcano_plot(df_comp, title):
    """Create volcano plot showing differential expression."""

    # Calculate -log10 of average abundance for y-axis (proxy for significance)
    df_comp['Avg_Abundance'] = (df_comp['Abundance_Young'] + df_comp['Abundance_Old']) / 2
    df_comp['NegLog_Abundance'] = -np.log10(df_comp['Avg_Abundance'] + 1)

    # Color by significance threshold
    df_comp['Significance'] = 'Not significant'
    df_comp.loc[df_comp['Zscore_Delta'].abs() > 1.0, 'Significance'] = 'Moderate change (|Δ| > 1)'
    df_comp.loc[df_comp['Zscore_Delta'].abs() > 2.0, 'Significance'] = 'Large change (|Δ| > 2)'

    # Create volcano plot
    fig = px.scatter(
        df_comp,
        x='Zscore_Delta',
        y='NegLog_Abundance',
        color='Significance',
        hover_data=['Gene_Symbol', 'Zscore_Young', 'Zscore_Old', 'Zscore_Delta'],
        color_discrete_map={
            'Not significant': '#CCCCCC',
            'Moderate change (|Δ| > 1)': '#FFA500',
            'Large change (|Δ| > 2)': '#FF0000'
        },
        title=title,
        labels={
            'Zscore_Delta': 'Z-Score Change (Old - Young)',
            'NegLog_Abundance': '-log10(Avg Abundance)',
        }
    )

    # Add vertical lines for thresholds
    fig.add_vline(x=-1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=-2, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_vline(x=2, line_dash="dash", line_color="red", opacity=0.5)

    fig.update_layout(height=500, template='plotly_white')

    return fig

fig_volcano_glom = create_volcano_plot(df_glom, "Glomerular Volcano Plot - Aging Signature")
fig_volcano_tubu = create_volcano_plot(df_tubu, "Tubulointerstitial Volcano Plot - Aging Signature")

# ==============================================================================
# 4. VISUALIZATION 3: SCATTER PLOT - Young vs Old Z-Scores
# ==============================================================================

print("3. Creating scatter plots (Zscore_Young vs Zscore_Old)...")

def create_scatter_young_vs_old(df_comp, title):
    """Scatter plot comparing Young vs Old z-scores."""

    # Color by ECM annotation
    df_comp['Is_ECM'] = df_comp['Matrisome_Category'].notna()

    fig = px.scatter(
        df_comp,
        x='Zscore_Young',
        y='Zscore_Old',
        color='Is_ECM',
        hover_data=['Gene_Symbol', 'Zscore_Delta', 'Matrisome_Category'],
        color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'},
        labels={
            'Zscore_Young': 'Z-Score (Young)',
            'Zscore_Old': 'Z-Score (Old)',
            'Is_ECM': 'ECM Protein'
        },
        title=title
    )

    # Add diagonal line (no change)
    fig.add_trace(go.Scatter(
        x=[-5, 5],
        y=[-5, 5],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        height=500,
        template='plotly_white',
        xaxis=dict(range=[-5, 5]),
        yaxis=dict(range=[-5, 5])
    )

    return fig

fig_scatter_glom = create_scatter_young_vs_old(df_glom, "Glomerular: Young vs Old Z-Scores")
fig_scatter_tubu = create_scatter_young_vs_old(df_tubu, "Tubulointerstitial: Young vs Old Z-Scores")

# ==============================================================================
# 5. VISUALIZATION 4: BAR CHART - Top Aging Markers
# ==============================================================================

print("4. Creating bar charts (Top 20 aging markers)...")

def create_top_markers_bar(df_comp, title):
    """Bar chart showing top proteins with largest aging changes."""

    # Get top 10 increases and decreases
    top_increases = df_comp.nlargest(10, 'Zscore_Delta')[['Gene_Symbol', 'Zscore_Delta', 'Matrisome_Category']]
    top_decreases = df_comp.nsmallest(10, 'Zscore_Delta')[['Gene_Symbol', 'Zscore_Delta', 'Matrisome_Category']]

    top_combined = pd.concat([top_increases, top_decreases]).sort_values('Zscore_Delta')

    # Color by direction
    top_combined['Color'] = top_combined['Zscore_Delta'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')

    fig = px.bar(
        top_combined,
        x='Zscore_Delta',
        y='Gene_Symbol',
        color='Color',
        hover_data=['Matrisome_Category'],
        orientation='h',
        color_discrete_map={'Increase': '#FF6B6B', 'Decrease': '#4ECDC4'},
        title=title,
        labels={
            'Zscore_Delta': 'Z-Score Change (Old - Young)',
            'Gene_Symbol': 'Gene'
        }
    )

    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=True
    )

    return fig

fig_bar_glom = create_top_markers_bar(df_glom, "Glomerular: Top 20 Aging Markers")
fig_bar_tubu = create_top_markers_bar(df_tubu, "Tubulointerstitial: Top 20 Aging Markers")

# ==============================================================================
# 6. VISUALIZATION 5: DISTRIBUTION HISTOGRAM - Zscore_Delta
# ==============================================================================

print("5. Creating distribution histograms (Zscore_Delta)...")

def create_distribution_histogram(df_comp, title):
    """Histogram showing distribution of z-score changes."""

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_comp['Zscore_Delta'],
        nbinsx=50,
        marker=dict(
            color=df_comp['Zscore_Delta'],
            colorscale='RdBu_r',
            cmid=0,
            colorbar=dict(title="ΔZ-Score")
        ),
        hovertemplate='ΔZ-Score: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))

    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.7)

    # Add mean line
    mean_delta = df_comp['Zscore_Delta'].mean()
    fig.add_vline(x=mean_delta, line_dash="dot", line_color="red", opacity=0.5,
                  annotation_text=f"Mean: {mean_delta:.3f}", annotation_position="top")

    fig.update_layout(
        title=title,
        xaxis_title="Z-Score Change (Old - Young)",
        yaxis_title="Frequency",
        height=400,
        template='plotly_white'
    )

    return fig

fig_hist_glom = create_distribution_histogram(df_glom, "Glomerular: Distribution of Z-Score Changes")
fig_hist_tubu = create_distribution_histogram(df_tubu, "Tubulointerstitial: Distribution of Z-Score Changes")

# ==============================================================================
# 7. VISUALIZATION 6: COMPARTMENT COMPARISON - Side-by-Side
# ==============================================================================

print("6. Creating compartment comparison (Glomerular vs Tubulointerstitial)...")

# Merge compartments on Gene_Symbol for direct comparison
df_comparison = pd.merge(
    df_glom[['Gene_Symbol', 'Zscore_Delta', 'Matrisome_Category']],
    df_tubu[['Gene_Symbol', 'Zscore_Delta']],
    on='Gene_Symbol',
    suffixes=('_Glom', '_Tubu')
)

# Create scatter plot
fig_comparison = px.scatter(
    df_comparison,
    x='Zscore_Delta_Glom',
    y='Zscore_Delta_Tubu',
    color='Matrisome_Category',
    hover_data=['Gene_Symbol'],
    title="Compartment Comparison: Glomerular vs Tubulointerstitial Aging Changes",
    labels={
        'Zscore_Delta_Glom': 'Glomerular ΔZ-Score',
        'Zscore_Delta_Tubu': 'Tubulointerstitial ΔZ-Score'
    }
)

# Add diagonal line (equal change in both compartments)
fig_comparison.add_trace(go.Scatter(
    x=[-5, 5],
    y=[-5, 5],
    mode='lines',
    line=dict(color='gray', dash='dash'),
    showlegend=False,
    name='Equal change',
    hoverinfo='skip'
))

fig_comparison.update_layout(
    height=600,
    template='plotly_white',
    xaxis=dict(range=[-5, 5]),
    yaxis=dict(range=[-5, 5])
)

# ==============================================================================
# 8. GENERATE HTML DASHBOARD
# ==============================================================================

print("\n7. Generating HTML dashboard...")

# Create HTML with all visualizations
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Randles 2021 Z-Score Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .viz-section {{
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .viz-section h2 {{
            color: #667eea;
            margin-top: 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(800px, 1fr));
            gap: 30px;
        }}
        .stats-box {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            margin-top: 50px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Randles 2021 Z-Score Visualization Dashboard</h1>
            <p><strong>Study:</strong> Kidney ECM Aging Proteomics (PMID: 34049963)</p>
            <p><strong>Analysis:</strong> Compartment-Specific Z-Score Normalization</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <!-- Summary Statistics -->
        <div class="viz-section">
            <h2>📊 Summary Statistics</h2>
            <div class="stats-box">
                <div class="stat-card">
                    <div class="label">Total Proteins</div>
                    <div class="number">{len(df_combined)//2}</div>
                    <div class="label">Unique</div>
                </div>
                <div class="stat-card">
                    <div class="label">Glomerular Proteins</div>
                    <div class="number">{len(df_glom)}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Tubulointerstitial Proteins</div>
                    <div class="number">{len(df_tubu)}</div>
                </div>
                <div class="stat-card">
                    <div class="label">ECM Proteins (Glom)</div>
                    <div class="number">{df_glom['Matrisome_Category'].notna().sum()}</div>
                </div>
                <div class="stat-card">
                    <div class="label">ECM Proteins (Tubu)</div>
                    <div class="number">{df_tubu['Matrisome_Category'].notna().sum()}</div>
                </div>
            </div>
        </div>

        <!-- Visualization 1: Heatmaps -->
        <div class="viz-section">
            <h2>🔥 1. Z-Score Heatmaps (Color Gradient: -5 to +5)</h2>
            <p><em>Градиент цвета показывает уровень экспрессии белков. Красный = высокий z-score, Синий = низкий z-score.</em></p>
            <div class="viz-grid">
                <div id="heatmap-glom"></div>
                <div id="heatmap-tubu"></div>
            </div>
        </div>

        <!-- Visualization 2: Volcano Plots -->
        <div class="viz-section">
            <h2>🌋 2. Volcano Plots - Differential Expression</h2>
            <p><em>Показывает белки с наибольшими изменениями при старении. Красные точки = большие изменения (|ΔZ| > 2).</em></p>
            <div class="viz-grid">
                <div id="volcano-glom"></div>
                <div id="volcano-tubu"></div>
            </div>
        </div>

        <!-- Visualization 3: Scatter Plots -->
        <div class="viz-section">
            <h2>📈 3. Young vs Old Z-Score Comparison</h2>
            <p><em>Точки выше диагонали = увеличение с возрастом, ниже диагонали = уменьшение.</em></p>
            <div class="viz-grid">
                <div id="scatter-glom"></div>
                <div id="scatter-tubu"></div>
            </div>
        </div>

        <!-- Visualization 4: Bar Charts -->
        <div class="viz-section">
            <h2>📊 4. Top 20 Aging Markers</h2>
            <p><em>Белки с самыми большими изменениями экспрессии при старении.</em></p>
            <div class="viz-grid">
                <div id="bar-glom"></div>
                <div id="bar-tubu"></div>
            </div>
        </div>

        <!-- Visualization 5: Histograms -->
        <div class="viz-section">
            <h2>📊 5. Distribution of Z-Score Changes</h2>
            <p><em>Распределение изменений z-score (Old - Young). Центр в 0 = нет систематических изменений.</em></p>
            <div class="viz-grid">
                <div id="hist-glom"></div>
                <div id="hist-tubu"></div>
            </div>
        </div>

        <!-- Visualization 6: Compartment Comparison -->
        <div class="viz-section">
            <h2>🔀 6. Compartment Comparison</h2>
            <p><em>Сравнение изменений между Glomerular и Tubulointerstitial. Точки на диагонали = одинаковые изменения в обоих компартментах.</em></p>
            <div id="comparison"></div>
        </div>

        <div class="footer">
            <p>Generated by ECM Atlas Analysis Pipeline | Claude Code</p>
            <p>Data source: 06_Randles_z_score_by_tissue_compartment/claude_code/</p>
        </div>
    </div>

    <script>
        // Render all plots
        var heatmapGlomData = JSON.parse('{fig_heatmap_glom.to_json()}');
        Plotly.newPlot('heatmap-glom', heatmapGlomData.data, heatmapGlomData.layout);

        var heatmapTubuData = JSON.parse('{fig_heatmap_tubu.to_json()}');
        Plotly.newPlot('heatmap-tubu', heatmapTubuData.data, heatmapTubuData.layout);

        var volcanoGlomData = JSON.parse('{fig_volcano_glom.to_json()}');
        Plotly.newPlot('volcano-glom', volcanoGlomData.data, volcanoGlomData.layout);

        var volcanoTubuData = JSON.parse('{fig_volcano_tubu.to_json()}');
        Plotly.newPlot('volcano-tubu', volcanoTubuData.data, volcanoTubuData.layout);

        var scatterGlomData = JSON.parse('{fig_scatter_glom.to_json()}');
        Plotly.newPlot('scatter-glom', scatterGlomData.data, scatterGlomData.layout);

        var scatterTubuData = JSON.parse('{fig_scatter_tubu.to_json()}');
        Plotly.newPlot('scatter-tubu', scatterTubuData.data, scatterTubuData.layout);

        var barGlomData = JSON.parse('{fig_bar_glom.to_json()}');
        Plotly.newPlot('bar-glom', barGlomData.data, barGlomData.layout);

        var barTubuData = JSON.parse('{fig_bar_tubu.to_json()}');
        Plotly.newPlot('bar-tubu', barTubuData.data, barTubuData.layout);

        var histGlomData = JSON.parse('{fig_hist_glom.to_json()}');
        Plotly.newPlot('hist-glom', histGlomData.data, histGlomData.layout);

        var histTubuData = JSON.parse('{fig_hist_tubu.to_json()}');
        Plotly.newPlot('hist-tubu', histTubuData.data, histTubuData.layout);

        var comparisonData = JSON.parse('{fig_comparison.to_json()}');
        Plotly.newPlot('comparison', comparisonData.data, comparisonData.layout);
    </script>
</body>
</html>
"""

# Save HTML dashboard
output_file = "zscore_dashboard.html"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✅ Dashboard generated: {output_file}")
print(f"\nTo view the dashboard, open: {output_file}")
print("\nDashboard includes:")
print("  1. Heatmaps with color gradient (-5 to +5)")
print("  2. Volcano plots for differential expression")
print("  3. Young vs Old scatter plots")
print("  4. Top 20 aging markers bar charts")
print("  5. Z-score change distribution histograms")
print("  6. Compartment comparison scatter plot")
