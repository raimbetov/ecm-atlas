# Unified Multi-Tissue ECM Aging Dashboard

Interactive web dashboard for visualizing and comparing ECM protein expression across multiple tissues, compartments, and studies.

## Overview

- **498 unique proteins** from 2 studies
- **5 compartments:** Glomerular, Tubulointerstitial, NP, IAF, OAF
- **2 organs:** Kidney, Intervertebral Disc
- **1,451 total entries** (protein × compartment pairs)

## Features

### 🔥 Main Heatmap Visualization
- Interactive protein × compartment matrix
- Color gradient: Blue (decreased) → White (stable) → Red (increased)
- Hover tooltips with detailed z-scores
- Click on protein to view details
- Handles missing data (gray N/A cells)

### 🎛️ Interactive Filters
- **Organs:** Kidney, Intervertebral Disc
- **Compartments:** All 5 compartments (multi-select)
- **Matrisome Categories:** Collagens, ECM Glycoproteins, etc.
- **Studies:** Randles_2021, Tam_2020
- **Aging Trend:** Increased/Decreased/Stable
- **Search:** By gene symbol or protein ID

### 📊 Protein Detail Panel
- Opens when clicking on protein in heatmap
- Shows expression across all compartments
- Displays z-scores (Young, Old, Delta)
- Indicates trend direction (⬆️ Up / ⬇️ Down / ⬌ Stable)

### 📈 Sorting Options
- By magnitude (sum of absolute z-scores)
- By matrisome category
- Alphabetically by protein name

## Quick Start

### 1. Start Servers

```bash
cd 09_unified_dashboard
./00_start_servers.sh
```

This will start:
- **API Server** on http://localhost:5002
- **HTTP Server** on http://localhost:8081

### 2. Open Dashboard

Open in browser: http://localhost:8081/dashboard.html

### 3. Stop Servers

Press `Ctrl+C` in the terminal where servers are running.

## Manual Server Start

If you prefer to start servers manually:

```bash
# Terminal 1: Start API server
cd 09_unified_dashboard
python3 api_server.py

# Terminal 2: Start HTTP server
cd 09_unified_dashboard
python3 -m http.server 8081

# Open in browser
open http://localhost:8081/dashboard.html
```

## API Endpoints

The Flask API server provides the following endpoints:

### GET /api/health
Health check

**Response:**
```json
{
  "status": "healthy",
  "message": "Unified ECM Dashboard API Server",
  "port": 5002
}
```

### GET /api/stats
Overall statistics

**Response:**
```json
{
  "total_proteins": 498,
  "total_entries": 1451,
  "multi_compartment_proteins": 419,
  "organs": 2,
  "compartments": 5,
  "studies": 2,
  "avg_zscore_delta": 0.127,
  "std_zscore_delta": 0.712
}
```

### GET /api/filters
Available filter options with counts

**Response:**
```json
{
  "organs": [{"name": "Kidney", "count": 229}, ...],
  "compartments": [{"name": "Glomerular", "count": 229, "organ": "Kidney"}, ...],
  "categories": [{"name": "ECM Glycoproteins", "count": 149}, ...],
  "studies": [{"name": "Randles_2021", "count": 458}, ...]
}
```

### GET /api/heatmap?[filters]
Heatmap data with optional filters

**Query parameters:**
- `organs` - comma-separated list (e.g., `organs=Kidney,Intervertebral_disc`)
- `compartments` - comma-separated list (e.g., `compartments=Glomerular,NP`)
- `categories` - comma-separated list (e.g., `categories=Collagens,ECM Glycoproteins`)
- `studies` - comma-separated list (e.g., `studies=Randles_2021`)
- `trend` - "up", "down", or "stable"
- `search` - search query for gene/protein name

**Response:**
```json
{
  "proteins": ["COL1A1", "COL1A2", ...],
  "compartments": ["Glomerular", "Tubulointerstitial", ...],
  "data": {
    "COL1A1": {
      "Glomerular": {
        "zscore_delta": 1.5,
        "zscore_young": -0.5,
        "zscore_old": 1.0,
        "dataset": "Randles_2021",
        "organ": "Kidney"
      },
      "NP": null
    }
  },
  "metadata": {
    "COL1A1": {
      "protein_id": "P02452",
      "protein_name": "Collagen alpha-1(I) chain",
      "matrisome_category": "Collagens"
    }
  },
  "summary": {
    "total_proteins": 150,
    "total_compartments": 5,
    "avg_zscore_delta": 0.8
  }
}
```

### GET /api/protein/<protein_id>
Detailed information for a specific protein

**Example:** `GET /api/protein/P02452`

**Response:**
```json
{
  "protein_id": "P02452",
  "gene_symbol": "COL1A1",
  "protein_name": "Collagen alpha-1(I) chain",
  "matrisome_category": "Collagens",
  "matrisome_division": "Core matrisome",
  "compartments": [
    {
      "organ": "Kidney",
      "compartment": "Glomerular",
      "dataset": "Randles_2021",
      "zscore_young": -0.5,
      "zscore_old": 1.0,
      "zscore_delta": 1.5,
      "abundance_young": 1000.5,
      "abundance_old": 1500.2
    }
  ]
}
```

## Data Source

The dashboard reads from `../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`, which contains:
- Randles 2021: Kidney (Glomerular, Tubulointerstitial)
- Tam 2020: Intervertebral Disc (NP, IAF, OAF)

## Architecture

### Backend
- **Flask REST API** on port 5002
- Loads merged CSV on startup
- Provides 6 endpoints for data queries
- Handles filtering, searching, and data aggregation

### Frontend
- **HTML + Plotly.js** for interactive visualizations
- **Vanilla JavaScript** (no frameworks)
- **CSS Grid + Flexbox** for responsive layout
- Served via Python HTTP server on port 8081

## File Structure

```
09_unified_dashboard/
├── 00_start_servers.sh          # Startup script
├── api_server.py                # Flask API (port 5002)
├── dashboard.html               # Main HTML page
├── static/
│   ├── app.js                   # JavaScript application logic
│   └── styles.css               # CSS styling
├── 00_TASK_UNIFIED_DASHBOARD.md # Task specification
├── MOCKUP.md                    # Visual mockups
└── README.md                    # This file
```

## Requirements

```bash
pip install flask flask-cors pandas numpy
```

## Browser Compatibility

Tested on:
- Chrome/Edge (recommended)
- Firefox
- Safari

## Troubleshooting

### Dashboard shows "Loading..." indefinitely
- Check that API server is running on port 5002
- Check browser console for CORS errors
- Verify merged CSV exists at `../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

### Port already in use
- API port 5002: Change in `api_server.py` (line 148) and `static/app.js` (line 2)
- HTTP port 8081: Change in `00_start_servers.sh` (line 16)

### No data in heatmap
- Check that at least one filter is selected in each category
- Try clicking "Clear All" then "Apply Filters"

## Future Enhancements

Possible improvements:
- Export heatmap as CSV/PNG
- Additional visualizations (volcano plots, scatter plots)
- Statistical analysis (correlation, PCA)
- User-adjustable color scale
- Mobile-optimized view
- Dark mode

## Credits

Built with [Claude Code](https://claude.com/claude-code)

Data sources:
- Randles et al. 2021 - Kidney aging proteomics
- Tam et al. 2020 - Intervertebral disc aging proteomics
