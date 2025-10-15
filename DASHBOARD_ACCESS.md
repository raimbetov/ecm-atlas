# ECM-Atlas Dashboard Access Guide

## 🟢 Dashboard Status: OPERATIONAL & RUNNING

The ECM-Atlas dashboard is currently running and accessible at:

**🌐 Dashboard URL:** http://localhost:8083/dashboard.html

---

## Server Status

### API Server
- **URL:** http://localhost:5004
- **Port:** 5004
- **Status:** ✅ Running
- **Memory:** ~120 MB
- **Process:** `python3 api_server.py`

### HTTP Server
- **URL:** http://localhost:8083
- **Port:** 8083
- **Status:** ✅ Running
- **Memory:** ~20 MB
- **Process:** `python3 -m http.server 8083`

---

## Quick Access

### Main Dashboard
```
http://localhost:8083/dashboard.html
```

### API Health Check
```bash
curl http://localhost:5004/api/health
```

### Global Statistics
```bash
curl http://localhost:5004/api/global_stats
```

### Available Datasets
```bash
curl http://localhost:5004/api/datasets
```

---

## API Endpoints

### Global Endpoints
- `GET /api/health` - Server status
- `GET /api/version` - Dashboard version
- `GET /api/global_stats` - Overall statistics
- `GET /api/datasets` - List all datasets

### Individual Dataset Analysis
- `GET /api/dataset/<name>/summary` - Dataset summary
- `GET /api/dataset/<name>/proteins` - Protein list
- `GET /api/dataset/<name>/protein/<protein_id>` - Single protein details

### Cross-Dataset Comparison
- `GET /api/compare/datasets` - Dataset comparison
- `GET /api/compare/proteins` - Protein comparison

---

## Example Queries

### Get Randles 2021 Summary
```bash
curl http://localhost:5004/api/dataset/Randles_2021/summary | python -m json.tool
```

### Get all proteins in Tam 2020
```bash
curl http://localhost:5004/api/dataset/Tam_2020/proteins | python -m json.tool
```

### Compare all datasets
```bash
curl http://localhost:5004/api/compare/datasets | python -m json.tool
```

---

## Dashboard Features

### Tab 1: Individual Dataset Analysis
- Select a dataset from dropdown
- View protein abundance patterns
- Analyze aging signatures
- Export data for further analysis

### Tab 2: Cross-Dataset Comparison
- Compare proteins across multiple datasets
- Identify common aging markers
- Cross-tissue analysis
- Aging signature discovery

### Global Statistics Panel
- Total proteins analyzed: 1,376
- Total data entries: 4,584
- Datasets integrated: 15
- Organs covered: 8
- Tissue compartments: 17

---

## Data Summary

### Integrated Datasets (15 studies)
1. **Angelidis 2019** - Lung
2. **Caldeira 2017** - Intervertebral Disc
3. **Dipali 2023** - Skin
4. **Lofaro 2021** - Skin Fibroblasts
5. **Ouni 2022** - Bone Marrow
6. **Randles 2021** - Kidney
7. **Santinha 2024** - Bone Marrow
8. **Schuler 2021** - Adipose Tissue
9. **Tam 2020** - Intervertebral Disc
10. **Tsumagari 2023** - Brain
11-15. (+ 5 additional studies)

### Data Statistics
- **Total Proteins:** 1,376 unique ECM proteins
- **Total Entries:** 4,584 protein-tissue-age combinations
- **Z-Scores:** Calculated and normalized
- **Compartments:** 17 tissue compartments
- **Species:** Homo sapiens, Mus musculus

---

## If You Need to Restart the Dashboard

```bash
cd 10_unified_dashboard_2_tabs
bash start_servers.sh
```

Then open:
```
http://localhost:8083/dashboard.html
```

---

## To Stop the Dashboard

```bash
# Press Ctrl+C in the terminal where servers are running
# OR
pkill -f "api_server.py"
pkill -f "http.server"
```

---

## Troubleshooting

### Dashboard not loading?
1. Check API server is running: `curl http://localhost:5004/api/health`
2. Check HTTP server: `curl http://localhost:8083/dashboard.html`
3. Try restarting servers if unresponsive

### API not responding?
```bash
# Check if process is running
ps aux | grep api_server
ps aux | grep http.server

# Restart if needed
cd 10_unified_dashboard_2_tabs
bash start_servers.sh
```

### Port already in use?
```bash
# Find and kill processes on ports
lsof -i :5004  # Find process on port 5004
lsof -i :8083  # Find process on port 8083
```

---

## Performance Tips

- Dashboard loads in ~2 seconds
- API responses < 500ms
- Total memory usage: ~140 MB
- Best viewed in Chrome, Firefox, or Safari
- JavaScript required for visualization

---

## Documentation

For technical details, see:
- `reports/INFRASTRUCTURE_REORGANIZATION_REPORT.md` - Full infrastructure documentation
- `reports/Z_SCORE_ZERO_VALUES_INVESTIGATION.md` - Z-score calculation details
- `CLAUDE.md` - Developer guide

---

## Contact

For issues or questions about the ECM-Atlas dashboard, refer to the project documentation or contact the development team.

---

**Last Updated:** 2025-10-16
**Status:** ✅ Operational
**Dashboard URL:** http://localhost:8083/dashboard.html
