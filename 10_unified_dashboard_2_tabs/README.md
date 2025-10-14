# ECM Atlas - Unified Analysis Dashboard v2

Comprehensive interactive dashboard for analyzing extracellular matrix (ECM) protein changes across aging tissues.

## Features

### 🔬 Individual Dataset Analysis
Detailed analysis of each dataset with 6 types of visualizations:

1. **Heatmap** - Top 100 aging-associated proteins with color gradients
2. **Volcano Plot** - Differential expression analysis
3. **Scatter Plot** - Young vs Old comparison with ECM protein highlighting
4. **Bar Chart** - Top 20 aging markers (10 increases + 10 decreases)
5. **Histograms** - Distribution of z-score changes per compartment
6. **Compartment Comparison** - Correlation between compartments (for multi-compartment datasets)

### 📊 Compare Datasets
Cross-dataset protein comparison with:
- Multi-tissue heatmap showing protein expression across all compartments
- Interactive filters (organs, compartments, categories, studies, trends)
- Protein search functionality
- Sortable by magnitude, category, or name

## Available Datasets

1. **Randles 2021** - Kidney (Glomerular, Tubulointerstitial)
2. **Angelidis 2019** - Lung
3. **Dipali 2023** - Ovary
4. **Tam 2020** - Intervertebral disc (NP, IAF, OAF)
5. **LiDermis 2021** - Skin dermis

## Quick Start

### 1. Start the servers

```bash
./start_servers.sh
```

This will start:
- API server on **http://localhost:5004**
- Dashboard on **http://localhost:8083/dashboard.html**

### 2. Open the dashboard

Navigate to: **http://localhost:8083/dashboard.html**

Or use the command:
```bash
open http://localhost:8083/dashboard.html
```

## Usage

### Individual Dataset Analysis

1. Click on "🔬 Individual Dataset Analysis" tab
2. Select a dataset from the dropdown menu
3. Use compartment tabs to switch between tissue compartments
4. Explore different visualizations:
   - Heatmaps show protein expression patterns
   - Volcano plots identify differentially expressed proteins
   - Scatter plots compare young vs old directly
   - Bar charts highlight top aging markers
   - Histograms show overall distribution
   - Comparison plots correlate compartment changes

### Compare Datasets

1. Click on "📊 Compare Datasets" tab
2. Use filters on the left to narrow down proteins:
   - **Organs**: Filter by tissue type
   - **Compartments**: Select specific tissue compartments
   - **Matrisome Categories**: Filter by ECM protein type
   - **Studies**: Include/exclude specific datasets
   - **Aging Trend**: Filter by increase/decrease/stable
   - **Search**: Find specific proteins by name
3. Click "Apply Filters" to update the heatmap
4. Use "Sort by" dropdown to organize proteins:
   - **Magnitude**: Proteins with largest changes
   - **Category**: Group by matrisome category
   - **Name**: Alphabetical order

## Architecture

### File Structure

```
10_unified_dashboard_ver2/
├── dashboard.html                  # Main HTML file with tab structure
├── api_server.py                   # Unified Flask API server
├── start_servers.sh                # Server startup script
├── static/
│   ├── styles.css                 # Shared styles
│   ├── main.js                    # Core functionality and navigation
│   ├── individual_dataset.js      # Individual analysis module
│   └── compare_datasets.js        # Cross-dataset comparison module
└── README.md                       # This file
```

### API Endpoints

#### Global Endpoints
- `GET /api/health` - Health check
- `GET /api/global_stats` - Overall statistics
- `GET /api/datasets` - List all datasets with metadata

#### Individual Dataset Endpoints
- `GET /api/dataset/<name>/stats` - Dataset statistics
- `GET /api/dataset/<name>/heatmap/<compartment>?n=100` - Heatmap data
- `GET /api/dataset/<name>/volcano/<compartment>` - Volcano plot data
- `GET /api/dataset/<name>/scatter/<compartment>` - Scatter plot data
- `GET /api/dataset/<name>/bars/<compartment>` - Bar chart data
- `GET /api/dataset/<name>/histogram/<compartment>` - Histogram data
- `GET /api/dataset/<name>/comparison` - Compartment comparison data

#### Cross-Dataset Comparison Endpoints
- `GET /api/compare/filters` - Available filter options
- `GET /api/compare/heatmap?[filters]` - Comparison heatmap with filters

## Data Source

All data is loaded from: `../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

This file contains:
- **908 unique proteins** across all datasets
- **2177 total entries** (protein-compartment combinations)
- **5 organs**: Kidney, Lung, Ovary, Intervertebral disc, Skin dermis
- **8 compartments**: Glomerular, Tubulointerstitial, Lung, Ovary, NP, IAF, OAF, Skin dermis

## Technology Stack

- **Backend**: Flask (Python) with CORS support
- **Frontend**: Vanilla JavaScript (modular architecture)
- **Visualization**: Plotly.js for interactive charts
- **Data**: Pandas for data processing

## Development

### Modifying Visualizations

Each tab has its own JavaScript module:
- **individual_dataset.js**: Edit to modify detailed analysis visualizations
- **compare_datasets.js**: Edit to modify cross-dataset comparison features

### Adding New Datasets

1. Add data to `merged_ecm_aging_zscore.csv`
2. Restart the API server
3. New dataset will appear automatically in the dropdown

## Stopping Servers

Press `Ctrl+C` in the terminal where you started `start_servers.sh`

Or manually kill processes:
```bash
lsof -i :5004  # Find API server PID
lsof -i :8083  # Find HTTP server PID
kill <PID>
```

## Troubleshooting

### Port Already in Use
If ports 5004 or 8083 are already in use, modify `start_servers.sh` and update:
- API server port in `api_server.py` (line 148)
- HTTP server port in `start_servers.sh`
- API_BASE in `static/main.js`

### Data Not Loading
1. Check that `../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv` exists
2. Verify API server logs for errors
3. Check browser console for JavaScript errors

### Visualizations Not Rendering
1. Ensure Plotly.js CDN is accessible
2. Check browser console for errors
3. Try refreshing the page

## Future Enhancements

- [ ] Export functionality for filtered data
- [ ] Protein detail panel with UniProt links
- [ ] Statistical significance indicators
- [ ] Pathway enrichment analysis
- [ ] Save/load filter presets
- [ ] PDF report generation

## Credits

Dashboard created for ECM Atlas project analyzing aging-related changes in extracellular matrix proteins across multiple tissues and organisms.
