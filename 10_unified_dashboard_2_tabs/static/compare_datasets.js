// Compare Datasets Module

const CompareDatasets = (function() {
    let allFilters = {};
    let currentHeatmapData = null;

    function init() {
        console.log('Compare Datasets module initialized');
        loadCompareTab();
    }

    async function loadCompareTab() {
        try {
            window.ECMAtlas.showLoading(true);

            // Load filters
            const filters = await window.ECMAtlas.fetchAPI('/api/compare/filters');
            allFilters = filters;

            renderCompareContent();
            renderFilters(filters);

            // Load initial heatmap
            await loadHeatmap();

            window.ECMAtlas.showLoading(false);
        } catch (error) {
            console.error('Error loading compare tab:', error);
            window.ECMAtlas.showLoading(false);
        }
    }

    function renderCompareContent() {
        const content = document.getElementById('compare-content');
        content.innerHTML = `
            <div class="grid-2" style="grid-template-columns: 300px 1fr; gap: 20px;">
                <!-- Filters Panel -->
                <aside class="filters-panel">
                    <h2 style="margin-top: 0;">Filters</h2>

                    <!-- Search -->
                    <div class="filter-group">
                        <h3>🔍 Search Protein</h3>
                        <input type="text" id="search-input" placeholder="COL1A1, Collagen..."
                               style="width: 100%; padding: 8px; border: 2px solid #e0e0e0; border-radius: 5px;" />
                    </div>

                    <!-- Compartments by Organ -->
                    <div class="filter-group">
                        <h3>🔬 Compartments</h3>
                        <div id="compartment-filters"></div>
                    </div>

                    <!-- Categories -->
                    <div class="filter-group">
                        <h3>🧬 Matrisome Categories</h3>
                        <div id="category-filters"></div>
                    </div>

                    <!-- Studies -->
                    <div class="filter-group">
                        <h3>📚 Studies</h3>
                        <div id="study-filters"></div>
                    </div>

                    <!-- Trend -->
                    <div class="filter-group">
                        <h3>📈 Aging Trend</h3>
                        <div id="trend-filters">
                            <label><input type="checkbox" value="up" class="trend-checkbox" /> Increased (&gt;+0.5)</label>
                            <label><input type="checkbox" value="down" class="trend-checkbox" /> Decreased (&lt;-0.5)</label>
                            <label><input type="checkbox" value="stable" class="trend-checkbox" /> Stable (±0.5)</label>
                        </div>
                    </div>

                    <div class="filter-actions">
                        <button id="apply-filters" class="btn-primary" style="width: 100%;">Apply Filters</button>
                        <button id="clear-filters" class="btn-secondary" style="width: 100%;">Clear All</button>
                    </div>

                    <div style="margin-top: 15px; padding: 10px; background: #f5f7fa; border-radius: 5px; text-align: center;">
                        <strong>Showing:</strong> <span id="protein-count">-</span> proteins
                    </div>
                </aside>

                <!-- Main Heatmap -->
                <main>
                    <div class="section" style="margin: 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                            <h2 style="margin: 0; border: none; padding: 0;">Multi-Compartment ECM Protein Expression</h2>
                            <div>
                                <label>
                                    Sort by:
                                    <select id="sort-by" style="padding: 8px; border: 2px solid #e0e0e0; border-radius: 5px;">
                                        <option value="magnitude">Magnitude</option>
                                        <option value="category">Category</option>
                                        <option value="name">Protein Name</option>
                                    </select>
                                </label>
                            </div>
                        </div>

                        

                        <div id="heatmap" class="chart-container tall">
                            <div class="loading">Loading heatmap...</div>
                        </div>

                        
                    </div>
                </main>
            </div>
        `;

        setupEventListeners();
    }

    function renderFilters(filters) {
        // Compartments grouped by organ with organ checkbox
        const compartmentFilters = document.getElementById('compartment-filters');
        const compartmentsByOrgan = {};
        filters.compartments.forEach(comp => {
            if (!compartmentsByOrgan[comp.organ]) {
                compartmentsByOrgan[comp.organ] = [];
            }
            compartmentsByOrgan[comp.organ].push(comp);
        });
        
        compartmentFilters.innerHTML = Object.keys(compartmentsByOrgan).sort().map(organ => `
            <div style="margin-bottom: 12px;">
                <label style="display: block; cursor: pointer; margin-bottom: 5px;">
                    <input type="checkbox" class="organ-checkbox" data-organ="${organ}" checked />
                    <strong style="color: #3b82f6;">${organ}</strong>
                </label>
                ${compartmentsByOrgan[organ].map(comp => `
                    <label style="display: block; padding: 3px 0 3px 25px; cursor: pointer;">
                        <input type="checkbox" class="compartment-checkbox" data-organ="${organ}" value="${comp.name}" checked />
                        ${comp.name} (${comp.count})
                    </label>
                `).join('')}
            </div>
        `).join('');

        // Categories
        const categoryFilters = document.getElementById('category-filters');
        categoryFilters.innerHTML = filters.categories.map(cat => `
            <label style="display: block; padding: 5px 0; cursor: pointer;">
                <input type="checkbox" class="category-checkbox" value="${cat.name}" checked />
                ${cat.name} (${cat.count})
            </label>
        `).join('');

        // Studies
        const studyFilters = document.getElementById('study-filters');
        studyFilters.innerHTML = filters.studies.map(study => `
            <label style="display: block; padding: 5px 0; cursor: pointer;">
                <input type="checkbox" class="study-checkbox" value="${study.name}" checked />
                ${study.name.replace('_', ' ')} (${study.count})
            </label>
        `).join('');
    }

    function setupEventListeners() {
        // Apply filters
        document.getElementById('apply-filters').addEventListener('click', loadHeatmap);

        // Clear filters
        document.getElementById('clear-filters').addEventListener('click', () => {
            document.querySelectorAll('.filters-panel input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });
            document.getElementById('search-input').value = '';
        });

        // Search on Enter
        document.getElementById('search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                loadHeatmap();
            }
        });

        // Sort selector
        document.getElementById('sort-by').addEventListener('change', (e) => {
            if (currentHeatmapData) {
                sortAndRenderHeatmap(e.target.value);
            }
        });

        // Organ checkbox toggle all compartments
        document.getElementById('compartment-filters').addEventListener('change', (e) => {
            if (e.target.classList.contains('organ-checkbox')) {
                const organ = e.target.dataset.organ;
                const checked = e.target.checked;
                document.querySelectorAll(`.compartment-checkbox[data-organ="${organ}"]`).forEach(cb => {
                    cb.checked = checked;
                });
            }
        });
    }

    function getSelectedFilters() {
        const filters = {};

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

        // Trend (allow multiple selections)
        const trends = Array.from(document.querySelectorAll('.trend-checkbox:checked'))
            .map(cb => cb.value);
        if (trends.length > 0) filters.trends = trends.join(',');

        // Search
        const search = document.getElementById('search-input').value.trim();
        if (search) filters.search = search;

        return filters;
    }

    async function loadHeatmap() {
        try {
            window.ECMAtlas.showLoading(true);

            const filters = getSelectedFilters();
            const queryString = new URLSearchParams(filters).toString();
            const endpoint = `/api/compare/heatmap${queryString ? '?' + queryString : ''}`;

            const data = await window.ECMAtlas.fetchAPI(endpoint);
            currentHeatmapData = data;

            renderHeatmap(data);
            updateProteinCount(data.summary.total_proteins);

            window.ECMAtlas.showLoading(false);
        } catch (error) {
            console.error('Error loading heatmap:', error);
            window.ECMAtlas.showLoading(false);
        }
    }

    function renderHeatmap(data) {
        const container = document.getElementById('heatmap');

        if (!data.proteins || data.proteins.length === 0) {
            container.innerHTML = '<p style="text-align: center; padding: 2rem;">No proteins match the selected filters.</p>';
            return;
        }

        const proteins = data.proteins;
        const compartments = data.compartments;

        // Create z-values matrix
        const zValues = [];
        const hoverText = [];

        for (const protein of proteins) {
            const row = [];
            const hoverRow = [];

            for (const compartment of compartments) {
                const compartmentData = data.data[protein][compartment];

                if (compartmentData && compartmentData.zscore_delta !== null) {
                    row.push(compartmentData.zscore_delta);
                    const isoformText = compartmentData.isoforms > 1
                        ? `<br><i>(${compartmentData.isoforms} isoforms - showing max)</i>`
                        : '';
                    hoverRow.push(
                        `<b>${protein}</b><br>` +
                        `Compartment: ${compartment}<br>` +
                        `Dataset: ${compartmentData.dataset}<br>` +
                        `Organ: ${compartmentData.organ}${isoformText}<br><br>` +
                        `Zscore Young: ${compartmentData.zscore_young?.toFixed(2) || 'N/A'}<br>` +
                        `Zscore Old: ${compartmentData.zscore_old?.toFixed(2) || 'N/A'}<br>` +
                        `<b>Zscore Delta: ${compartmentData.zscore_delta.toFixed(2)}</b>`
                    );
                } else {
                    row.push(null);
                    hoverRow.push(`<b>${protein}</b><br>Compartment: ${compartment}<br>No data available`);
                }
            }

            zValues.push(row);
            hoverText.push(hoverRow);
        }

        // Create Plotly heatmap with scientific diverging color scheme (RdBu-like)
        // Best practice: Use perceptually uniform diverging colormap for easy interpretation
        const trace = {
            z: zValues,
            x: compartments,
            y: proteins,
            type: 'heatmap',
            colorscale: [
                [0.0, '#0033ff'],
                [0.2, '#0066ff'],
                [0.35, '#66b3ff'],
                [0.5, '#f7f7f7'],
                [0.65, '#ffcc00'],
                [0.8, '#ff6600'],
                [1.0, '#ff0000']
            ],
            zmin: -3,
            zmax: 3,
            hovertemplate: '%{text}<extra></extra>',
            text: hoverText,
            showscale: false
        };

        const topAnnotations = compartments.map((comp, idx) => ({
            x: idx,
            y: 1,
            xref: 'x',
            yref: 'paper',
            text: comp,
            showarrow: false,
            textangle: -45,
            xanchor: 'left',
            yanchor: 'bottom',
            font: { size: 10 }
        }));

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
            annotations: topAnnotations,
            margin: { l: 100, r: 20, t: 120, b: 80 },
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
                filename: 'ecm_comparison_heatmap',
                height: layout.height,
                width: 1200,
                scale: 2
            }
        };

        Plotly.newPlot('heatmap', [trace], layout, config);
    }

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

    function updateProteinCount(count) {
        const countElem = document.getElementById('protein-count');
        if (countElem) {
            countElem.textContent = count;
        }
    }

    return {
        init
    };
})();

// Make available globally
window.CompareDatasets = CompareDatasets;
