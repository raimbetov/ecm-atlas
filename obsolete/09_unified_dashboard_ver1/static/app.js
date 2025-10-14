// API Configuration
const API_BASE = 'http://localhost:5002';

// Global State
let allFilters = {};
let currentHeatmapData = null;

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', async () => {
    await loadInitialData();
    setupEventListeners();
});

// Load Initial Data
async function loadInitialData() {
    try {
        showLoading(true);

        // Load filters and stats
        const [filters, stats] = await Promise.all([
            fetchAPI('/api/filters'),
            fetchAPI('/api/stats')
        ]);

        allFilters = filters;
        renderFilters(filters);
        renderStats(stats);

        // Load initial heatmap data
        await loadHeatmap();

        showLoading(false);
    } catch (error) {
        console.error('Error loading initial data:', error);
        showLoading(false);
        alert('Error loading data. Please check if API server is running on port 5002.');
    }
}

// Fetch from API
async function fetchAPI(endpoint) {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
    }
    return await response.json();
}

// Render Filters
function renderFilters(filters) {
    // Organs
    const organFilters = document.getElementById('organ-filters');
    organFilters.innerHTML = filters.organs.map(organ => `
        <label>
            <input type="checkbox" class="organ-checkbox" value="${organ.name}" checked />
            ${organ.name} (${organ.count})
        </label>
    `).join('');

    // Compartments
    const compartmentFilters = document.getElementById('compartment-filters');
    compartmentFilters.innerHTML = filters.compartments.map(comp => `
        <label>
            <input type="checkbox" class="compartment-checkbox" value="${comp.name}" checked />
            ${comp.name} (${comp.count})
        </label>
    `).join('');

    // Categories
    const categoryFilters = document.getElementById('category-filters');
    categoryFilters.innerHTML = filters.categories.map(cat => `
        <label>
            <input type="checkbox" class="category-checkbox" value="${cat.name}" checked />
            ${cat.name} (${cat.count})
        </label>
    `).join('');

    // Studies
    const studyFilters = document.getElementById('study-filters');
    studyFilters.innerHTML = filters.studies.map(study => `
        <label>
            <input type="checkbox" class="study-checkbox" value="${study.name}" checked />
            ${study.name} (${study.count})
        </label>
    `).join('');
}

// Render Stats
function renderStats(stats) {
    const statsSummary = document.getElementById('stats-summary');
    statsSummary.innerHTML = `
        <div class="stat-item">📊 <strong>${stats.total_proteins}</strong> Proteins</div>
        <div class="stat-item">🫀 <strong>${stats.organs}</strong> Organs</div>
        <div class="stat-item">🔬 <strong>${stats.compartments}</strong> Compartments</div>
        <div class="stat-item">Δ Avg: <strong>${stats.avg_zscore_delta.toFixed(2)}</strong></div>
    `;
}

// Get Selected Filters
function getSelectedFilters() {
    const filters = {};

    // Organs
    const organs = Array.from(document.querySelectorAll('.organ-checkbox:checked'))
        .map(cb => cb.value);
    if (organs.length > 0) filters.organs = organs.join(',');

    // Compartments
    const compartments = Array.from(document.querySelectorAll('.compartment-checkbox:checked'))
        .map(cb => cb.value);
    if (compartments.length > 0) filters.compartments = compartments.join(',');

    // Categories
    const categories = Array.from(document.querySelectorAll('.category-checkbox:checked'))
        .map(cb => cb.value);
    if (categories.length > 0) filters.categories = categories.join(',');

    // Studies
    const studies = Array.from(document.querySelectorAll('.study-checkbox:checked'))
        .map(cb => cb.value);
    if (studies.length > 0) filters.studies = studies.join(',');

    // Trend
    const trends = Array.from(document.querySelectorAll('.trend-checkbox:checked'))
        .map(cb => cb.value);
    if (trends.length === 1) filters.trend = trends[0];

    // Search
    const search = document.getElementById('search-input').value.trim();
    if (search) filters.search = search;

    return filters;
}

// Load Heatmap Data
async function loadHeatmap() {
    try {
        showLoading(true);

        const filters = getSelectedFilters();
        const queryString = new URLSearchParams(filters).toString();
        const endpoint = `/api/heatmap${queryString ? '?' + queryString : ''}`;

        const data = await fetchAPI(endpoint);
        currentHeatmapData = data;

        renderHeatmap(data);
        updateProteinCount(data.summary.total_proteins);

        showLoading(false);
    } catch (error) {
        console.error('Error loading heatmap:', error);
        showLoading(false);
    }
}

// Render Heatmap
function renderHeatmap(data) {
    if (!data.proteins || data.proteins.length === 0) {
        document.getElementById('heatmap').innerHTML = '<p style="text-align: center; padding: 2rem;">No proteins match the selected filters.</p>';
        return;
    }

    // Prepare data for Plotly heatmap
    const proteins = data.proteins;
    const compartments = data.compartments;

    // Create z-values matrix
    const zValues = [];
    const hoverText = [];
    const customData = [];

    for (const protein of proteins) {
        const row = [];
        const hoverRow = [];
        const dataRow = [];

        for (const compartment of compartments) {
            const compartmentData = data.data[protein][compartment];

            if (compartmentData && compartmentData.zscore_delta !== null) {
                row.push(compartmentData.zscore_delta);
                hoverRow.push(
                    `<b>${protein}</b><br>` +
                    `Compartment: ${compartment}<br>` +
                    `Dataset: ${compartmentData.dataset}<br>` +
                    `Organ: ${compartmentData.organ}<br><br>` +
                    `Zscore Young: ${compartmentData.zscore_young?.toFixed(2) || 'N/A'}<br>` +
                    `Zscore Old: ${compartmentData.zscore_old?.toFixed(2) || 'N/A'}<br>` +
                    `<b>Zscore Delta: ${compartmentData.zscore_delta.toFixed(2)}</b>`
                );
                dataRow.push({
                    protein_id: data.metadata[protein].protein_id,
                    protein_name: data.metadata[protein].protein_name
                });
            } else {
                row.push(null);
                hoverRow.push(`<b>${protein}</b><br>Compartment: ${compartment}<br>No data available`);
                dataRow.push(null);
            }
        }

        zValues.push(row);
        hoverText.push(hoverRow);
        customData.push(dataRow);
    }

    // Create Plotly heatmap
    const trace = {
        z: zValues,
        x: compartments,
        y: proteins,
        type: 'heatmap',
        colorscale: [
            [0, '#3b82f6'],    // Blue (-3)
            [0.25, '#60a5fa'], // Light blue (-1)
            [0.5, '#f3f4f6'],  // Gray (0)
            [0.75, '#fbbf24'], // Yellow (+1)
            [1, '#ef4444']     // Red (+3)
        ],
        zmin: -3,
        zmax: 3,
        hovertemplate: '%{text}<extra></extra>',
        text: hoverText,
        customdata: customData,
        colorbar: {
            title: 'Δ Z-Score',
            titleside: 'right',
            tickmode: 'linear',
            tick0: -3,
            dtick: 1
        }
    };

    const layout = {
        xaxis: {
            title: 'Compartment',
            side: 'bottom',
            tickangle: -45
        },
        yaxis: {
            title: 'Protein',
            autorange: 'reversed',
            tickfont: { size: 10 }
        },
        margin: { l: 100, r: 100, t: 50, b: 100 },
        height: Math.max(600, proteins.length * 20),
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: 'ecm_heatmap',
            height: layout.height,
            width: 1200,
            scale: 2
        }
    };

    Plotly.newPlot('heatmap', [trace], layout, config);

    // Add click event for protein details
    document.getElementById('heatmap').on('plotly_click', function(eventData) {
        const point = eventData.points[0];
        const protein = point.y;
        const proteinId = data.metadata[protein]?.protein_id;

        if (proteinId) {
            showProteinDetails(proteinId);
        }
    });
}

// Show Protein Details
async function showProteinDetails(proteinId) {
    try {
        const detailPanel = document.getElementById('detail-panel');
        const detailContent = document.getElementById('detail-content');

        detailPanel.style.display = 'block';
        detailContent.innerHTML = '<p>Loading...</p>';

        const protein = await fetchAPI(`/api/protein/${proteinId}`);

        detailContent.innerHTML = `
            <div class="protein-info">
                <h3>${protein.gene_symbol}</h3>
                <p><strong>Name:</strong> ${protein.protein_name}</p>
                <p><strong>ID:</strong> <span class="protein-id">${protein.protein_id}</span></p>
                <p><strong>Category:</strong> ${protein.matrisome_category || 'N/A'}</p>
                <p><strong>Division:</strong> ${protein.matrisome_division || 'N/A'}</p>
            </div>

            <h4>Expression Across Compartments</h4>
            <table class="compartment-table">
                <thead>
                    <tr>
                        <th>Compartment</th>
                        <th>Organ</th>
                        <th>Young</th>
                        <th>Old</th>
                        <th>Delta</th>
                        <th>Trend</th>
                    </tr>
                </thead>
                <tbody>
                    ${protein.compartments.map(comp => {
                        const trend = comp.zscore_delta > 0.5 ? '⬆️ Up' :
                                     comp.zscore_delta < -0.5 ? '⬇️ Down' :
                                     '⬌ Stable';
                        const trendClass = comp.zscore_delta > 0.5 ? 'trend-up' :
                                          comp.zscore_delta < -0.5 ? 'trend-down' :
                                          'trend-stable';

                        return `
                            <tr>
                                <td><strong>${comp.compartment}</strong></td>
                                <td>${comp.organ}</td>
                                <td>${comp.zscore_young?.toFixed(2) || 'N/A'}</td>
                                <td>${comp.zscore_old?.toFixed(2) || 'N/A'}</td>
                                <td><strong>${comp.zscore_delta?.toFixed(2) || 'N/A'}</strong></td>
                                <td class="${trendClass}">${trend}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        `;
    } catch (error) {
        console.error('Error loading protein details:', error);
        alert('Error loading protein details');
    }
}

// Update Protein Count
function updateProteinCount(count) {
    document.getElementById('protein-count').textContent = count;
}

// Setup Event Listeners
function setupEventListeners() {
    // Apply filters button
    document.getElementById('apply-filters').addEventListener('click', loadHeatmap);

    // Clear filters button
    document.getElementById('clear-filters').addEventListener('click', () => {
        document.querySelectorAll('.filters-panel input[type="checkbox"]').forEach(cb => {
            cb.checked = false;
        });
        document.getElementById('search-input').value = '';
    });

    // Close detail panel
    document.getElementById('close-details').addEventListener('click', () => {
        document.getElementById('detail-panel').style.display = 'none';
    });

    // Search on Enter
    document.getElementById('search-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            loadHeatmap();
        }
    });

    // Sort by selector
    document.getElementById('sort-by').addEventListener('change', (e) => {
        if (currentHeatmapData) {
            sortAndRenderHeatmap(e.target.value);
        }
    });
}

// Sort and Re-render Heatmap
function sortAndRenderHeatmap(sortBy) {
    if (!currentHeatmapData) return;

    const data = { ...currentHeatmapData };
    let proteins = [...data.proteins];

    if (sortBy === 'magnitude') {
        // Sort by sum of absolute z-scores
        proteins.sort((a, b) => {
            const sumA = Object.values(data.data[a])
                .filter(v => v && v.zscore_delta !== null)
                .reduce((sum, v) => sum + Math.abs(v.zscore_delta), 0);
            const sumB = Object.values(data.data[b])
                .filter(v => v && v.zscore_delta !== null)
                .reduce((sum, v) => sum + Math.abs(v.zscore_delta), 0);
            return sumB - sumA;
        });
    } else if (sortBy === 'category') {
        // Sort by matrisome category
        proteins.sort((a, b) => {
            const catA = data.metadata[a].matrisome_category || 'ZZZ';
            const catB = data.metadata[b].matrisome_category || 'ZZZ';
            return catA.localeCompare(catB);
        });
    } else if (sortBy === 'name') {
        // Sort alphabetically
        proteins.sort((a, b) => a.localeCompare(b));
    }

    data.proteins = proteins;
    renderHeatmap(data);
}

// Show/Hide Loading Overlay
function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (show) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}
