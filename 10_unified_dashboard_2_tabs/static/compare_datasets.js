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
            // Use per-item loaders instead of global overlay

            // Load filters
            const filters = await window.ECMAtlas.fetchAPI('/api/compare/filters');
            allFilters = filters;

            renderCompareContent();
            renderFilters(filters);

            // Setup event listeners after filters are rendered
            setTimeout(() => {
                setupEventListeners();
            }, 100);

            // Load initial heatmap
            await loadHeatmap();

            
        } catch (error) {
            console.error('Error loading compare tab:', error);
        }
    }

    function renderCompareContent() {
        // Render top search bar
        const searchContainer = document.getElementById('search-container');
        searchContainer.innerHTML = `
            <div class="top-search-input-container">
                <input type="text" id="search-input" placeholder="Search proteins (e.g., COL1A1, Collagen)..." autocomplete="off" />
                <div id="search-suggestions" class="search-suggestions-top" style="display: none;"></div>
            </div>
        `;

        // Render filters in the left panel as a single tree component
        const filtersContent = document.getElementById('filters-content');
        filtersContent.innerHTML = `
            <!-- Unified Filter Tree -->
            <div class="filter-tree">
                <div class="tree-node root-node expanded">
                    <div class="tree-node-header">
                        <span class="tree-toggle">▼</span>
                        <span class="tree-label">All Filters</span>
                        <span class="filter-count" id="total-filter-count"></span>
                    </div>
                    <div class="tree-node-content">

                        <!-- Compartments Branch -->
                        <div class="tree-node compartment-node expanded">
                            <div class="tree-node-header">
                                <span class="tree-toggle">▼</span>
                                <span class="tree-label">Compartments</span>
                                <span class="filter-count" id="compartment-count"></span>
                            </div>
                            <div class="tree-node-content">
                                <div id="compartment-filters"></div>
                            </div>
                        </div>

                        <!-- Categories Branch -->
                        <div class="tree-node category-node expanded">
                            <div class="tree-node-header">
                                <span class="tree-toggle">▼</span>
                                <span class="tree-label">Matrisome Categories</span>
                                <span class="filter-count" id="category-count"></span>
                            </div>
                            <div class="tree-node-content">
                                <div id="category-filters"></div>
                            </div>
                        </div>

                        <!-- Studies Branch -->
                        <div class="tree-node study-node">
                            <div class="tree-node-header">
                                <span class="tree-toggle">▶</span>
                                <span class="tree-label">Studies</span>
                                <span class="filter-count" id="study-count"></span>
                            </div>
                            <div class="tree-node-content" style="display: none;">
                                <div id="study-filters"></div>
                            </div>
                        </div>

                        <!-- Aging Trend Branch -->
                        <div class="tree-node trend-node expanded">
                            <div class="tree-node-header">
                                <span class="tree-toggle">▼</span>
                                <span class="tree-label">Aging Trend</span>
                                <span class="filter-count" id="trend-count"></span>
                            </div>
                            <div class="tree-node-content">
                                <div id="trend-filters">
                                    <label class="tree-leaf filter-label">
                                        <input type="checkbox" value="up" class="trend-checkbox filter-checkbox" />
                                        <span class="trend-label up">Increased (>+0.5)</span>
                                    </label>
                                    <label class="tree-leaf filter-label">
                                        <input type="checkbox" value="down" class="trend-checkbox filter-checkbox" />
                                        <span class="trend-label down">Decreased (<-0.5)</span>
                                    </label>
                                    <label class="tree-leaf filter-label">
                                        <input type="checkbox" value="stable" class="trend-checkbox filter-checkbox" />
                                        <span class="trend-label stable">Stable (±0.5)</span>
                                    </label>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>

            <!-- Filter Controls -->
            <div class="filter-controls">
                <div class="control-row">
                    <label class="control-option">
                        <input type="checkbox" id="auto-apply-toggle" class="filter-checkbox" checked />
                        <span>Auto-apply filters</span>
                    </label>
                </div>
                <div class="control-buttons">
                    <button id="apply-filters" class="btn-primary">Apply Filters</button>
                    <button id="clear-filters" class="btn-secondary">Clear All</button>
                </div>
            </div>

            <!-- Results Summary -->
            <div class="results-summary">
                <div class="summary-content">
                    <strong>Showing:</strong> <span id="protein-count">-</span> proteins
                    <div id="filter-summary" class="filter-summary-text"></div>
                </div>
            </div>
        `;

        // Render heatmap in the right panel
        const content = document.getElementById('compare-content');
        content.innerHTML = `
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
        `;
    }

    function renderFilters(filters) {
        // Compartments grouped by organ as tree nodes
        const compartmentFilters = document.getElementById('compartment-filters');
        const compartmentsByOrgan = {};
        filters.compartments.forEach(comp => {
            if (!compartmentsByOrgan[comp.organ]) {
                compartmentsByOrgan[comp.organ] = [];
            }
            compartmentsByOrgan[comp.organ].push(comp);
        });

        compartmentFilters.innerHTML = Object.keys(compartmentsByOrgan).sort().map(organ => {
            const organCount = compartmentsByOrgan[organ].reduce((sum, comp) => sum + comp.count, 0);
            return `
            <div class="tree-node organ-node expanded">
                <div class="tree-node-header">
                    <span class="tree-toggle">▼</span>
                    <input type="checkbox" class="organ-checkbox filter-checkbox" data-organ="${organ}" checked />
                    <span class="tree-label" style="color: #3b82f6; font-weight: 600;">${organ}<span class="dataset-item-count">(${organCount})</span></span>
                </div>
                <div class="tree-node-content">
                    ${compartmentsByOrgan[organ].map(comp => `
                        <label class="tree-leaf filter-label">
                            <input type="checkbox" class="compartment-checkbox filter-checkbox" data-organ="${organ}" value="${comp.name}" checked />
                            <span class="tree-label">${comp.name}<span class="dataset-item-count">(${comp.count})</span></span>
                        </label>
                    `).join('')}
                </div>
            </div>
        `;
        }).join('');

        // Categories as tree leaves
        const categoryFilters = document.getElementById('category-filters');
        categoryFilters.innerHTML = filters.categories.map(cat => `
            <label class="tree-leaf filter-label">
                <input type="checkbox" class="category-checkbox filter-checkbox" value="${cat.name}" checked />
                <span class="tree-label">${cat.name}<span class="dataset-item-count">(${cat.count})</span></span>
            </label>
        `).join('');

        // Studies as tree leaves
        const studyFilters = document.getElementById('study-filters');
        studyFilters.innerHTML = filters.studies.map(study => `
            <label class="tree-leaf filter-label">
                <input type="checkbox" class="study-checkbox filter-checkbox" value="${study.name}" checked />
                <span class="tree-label">${study.name.replace('_', ' ')}<span class="dataset-item-count">(${study.count})</span></span>
            </label>
        `).join('');
    }

    function setupEventListeners() {
        // Apply filters
        document.getElementById('apply-filters').addEventListener('click', loadHeatmap);

        // Clear all filters
        document.getElementById('clear-filters').addEventListener('click', () => {
            clearAllFilters();
        });



        // Auto-apply toggle
        document.getElementById('auto-apply-toggle').addEventListener('change', (e) => {
            const autoApply = e.target.checked;

            if (autoApply) {
                // Auto-apply immediately when enabled
                loadHeatmap();
            }
        });

        // Search input with autocomplete (now at top)
        const searchInput = document.getElementById('search-input');
        searchInput.addEventListener('input', handleSearchInput);
        searchInput.addEventListener('keydown', handleSearchKeydown);
        searchInput.addEventListener('blur', () => {
            setTimeout(() => hideSearchSuggestions(), 150); // Delay to allow click on suggestions
        });

        // Search on Enter (fallback)
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                loadHeatmap();
                hideSearchSuggestions();
            }
        });

        // Sort selector
        document.getElementById('sort-by').addEventListener('change', (e) => {
            if (currentHeatmapData) {
                sortAndRenderHeatmap(e.target.value);
            }
        });

        // Attach event listeners to all filter checkboxes directly
        // Organ checkboxes
        document.querySelectorAll('.organ-checkbox').forEach(cb => {
            cb.addEventListener('change', (e) => {
                const organ = e.target.dataset.organ;
                const checked = e.target.checked;
                document.querySelectorAll(`.compartment-checkbox[data-organ="${organ}"]`).forEach(compCb => {
                    compCb.checked = checked;
                });
                updateFilterDisplay();
                autoApplyIfEnabled();
            });
        });

        // Individual filter checkboxes
        document.querySelectorAll('.compartment-checkbox, .category-checkbox, .study-checkbox, .trend-checkbox').forEach(cb => {
            cb.addEventListener('change', (e) => {
                updateFilterDisplay();
                autoApplyIfEnabled();
            });
        });

        // Tree node toggling
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('tree-toggle') || e.target.closest('.tree-node-header')) {
                const header = e.target.closest('.tree-node-header');
                if (header) {
                    const node = header.parentElement;
                    toggleTreeNode(node);
                }
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
            const filters = getSelectedFilters();
            const queryString = new URLSearchParams(filters).toString();
            const endpoint = `/api/compare/heatmap${queryString ? '?' + queryString : ''}`;

            const data = await window.ECMAtlas.fetchAPI(endpoint);
            currentHeatmapData = data;

            renderHeatmap(data);
            updateProteinCount(data.summary.total_proteins);
            updateFilterDisplay();

        } catch (error) {
            console.error('Error loading heatmap:', error);
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
        updateFilterSummary();
    }

    // Enhanced UX Functions

    function clearAllFilters() {
        // Clear all checkboxes
        document.querySelectorAll('#filters-content input[type="checkbox"]').forEach(cb => {
            cb.checked = false;
        });
        // Clear search
        document.getElementById('search-input').value = '';
        // Update display
        updateFilterDisplay();
        // Apply filters
        loadHeatmap();
    }

    function autoApplyIfEnabled() {
        const autoApplyToggle = document.getElementById('auto-apply-toggle');
        if (autoApplyToggle && autoApplyToggle.checked) {
            // Debounce auto-apply to prevent excessive API calls
            clearTimeout(autoApplyIfEnabled.timeoutId);
            autoApplyIfEnabled.timeoutId = setTimeout(() => {
                loadHeatmap();
            }, 300);
        }
    }

    function updateFilterDisplay() {
        const activeFilters = getActiveFilters();
        updateFilterCounts(activeFilters);
    }

    function getActiveFilters() {
        const filters = {
            search: document.getElementById('search-input').value.trim(),
            compartments: Array.from(document.querySelectorAll('.compartment-checkbox:checked')).map(cb => ({
                value: cb.value,
                organ: cb.dataset.organ,
                label: cb.parentElement.textContent.trim().split(' (')[0]
            })),
            categories: Array.from(document.querySelectorAll('.category-checkbox:checked')).map(cb => ({
                value: cb.value,
                label: cb.parentElement.textContent.trim().split(' (')[0]
            })),
            studies: Array.from(document.querySelectorAll('.study-checkbox:checked')).map(cb => ({
                value: cb.value,
                label: cb.parentElement.textContent.trim().split(' (')[0]
            })),
            trends: Array.from(document.querySelectorAll('.trend-checkbox:checked')).map(cb => ({
                value: cb.value,
                label: getTrendLabel(cb.value)
            }))
        };
        return filters;
    }

    function getTrendLabel(value) {
        const labels = {
            'up': 'Increased (>+0.5)',
            'down': 'Decreased (<-0.5)',
            'stable': 'Stable (±0.5)'
        };
        return labels[value] || value;
    }


    function updateFilterCounts(filters) {
        // Update count badges for each tree node
        const totalCount = filters.compartments.length + filters.categories.length + filters.studies.length + filters.trends.length;

        document.getElementById('total-filter-count').textContent = totalCount || '';
        document.getElementById('total-filter-count').style.display = totalCount ? 'inline' : 'none';

        document.getElementById('compartment-count').textContent = filters.compartments.length || '';
        document.getElementById('compartment-count').style.display = filters.compartments.length ? 'inline' : 'none';

        document.getElementById('category-count').textContent = filters.categories.length || '';
        document.getElementById('category-count').style.display = filters.categories.length ? 'inline' : 'none';

        document.getElementById('study-count').textContent = filters.studies.length || '';
        document.getElementById('study-count').style.display = filters.studies.length ? 'inline' : 'none';

        document.getElementById('trend-count').textContent = filters.trends.length || '';
        document.getElementById('trend-count').style.display = filters.trends.length ? 'inline' : 'none';
    }



    function updateFilterSummary() {
        const activeFilters = getActiveFilters();
        const summaryElem = document.getElementById('filter-summary');
        const summaryTexts = [];

        if (activeFilters.search) summaryTexts.push(`search: "${activeFilters.search}"`);
        if (activeFilters.compartments.length) summaryTexts.push(`${activeFilters.compartments.length} compartments`);
        if (activeFilters.categories.length) summaryTexts.push(`${activeFilters.categories.length} categories`);
        if (activeFilters.studies.length) summaryTexts.push(`${activeFilters.studies.length} studies`);
        if (activeFilters.trends.length) summaryTexts.push(`${activeFilters.trends.length} trends`);

        summaryElem.textContent = summaryTexts.length ? `(${summaryTexts.join(', ')})` : '';
    }

    function toggleTreeNode(node) {
        node.classList.toggle('expanded');
        const content = node.querySelector('.tree-node-content');
        const toggle = node.querySelector('.tree-toggle');

        if (content) {
            content.style.display = node.classList.contains('expanded') ? 'block' : 'none';
        }

        if (toggle) {
            toggle.textContent = node.classList.contains('expanded') ? '▼' : '▶';
        }
    }

    // Search autocomplete functions
    function handleSearchInput(e) {
        const query = e.target.value.trim();
        if (query.length < 2) {
            hideSearchSuggestions();
            return;
        }

        // Get protein suggestions from current heatmap data
        if (currentHeatmapData && currentHeatmapData.proteins) {
            const suggestions = currentHeatmapData.proteins
                .filter(protein => protein.toLowerCase().includes(query.toLowerCase()))
                .slice(0, 5);

            showSearchSuggestions(suggestions, query);
        }
    }

    function handleSearchKeydown(e) {
        const suggestions = document.getElementById('search-suggestions');
        if (!suggestions || suggestions.style.display === 'none') return;

        const items = suggestions.querySelectorAll('.suggestion-item');
        let activeIndex = Array.from(items).findIndex(item => item.classList.contains('active'));

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                activeIndex = Math.min(activeIndex + 1, items.length - 1);
                updateActiveSuggestion(items, activeIndex);
                break;
            case 'ArrowUp':
                e.preventDefault();
                activeIndex = Math.max(activeIndex - 1, -1);
                updateActiveSuggestion(items, activeIndex);
                break;
            case 'Enter':
                e.preventDefault();
                if (activeIndex >= 0) {
                    selectSuggestion(items[activeIndex]);
                }
                break;
            case 'Escape':
                hideSearchSuggestions();
                break;
        }
    }

    function showSearchSuggestions(suggestions, query) {
        const container = document.getElementById('search-suggestions');
        if (suggestions.length === 0) {
            hideSearchSuggestions();
            return;
        }

        const html = suggestions.map((suggestion, index) => `
            <div class="suggestion-item ${index === 0 ? 'active' : ''}"
                 onclick="CompareDatasets.selectSearchSuggestion('${suggestion}')">
                ${highlightMatch(suggestion, query)}
            </div>
        `).join('');

        container.innerHTML = html;
        container.style.display = 'block';
    }

    function hideSearchSuggestions() {
        const container = document.getElementById('search-suggestions');
        if (container) {
            container.style.display = 'none';
        }
    }

    function updateActiveSuggestion(items, activeIndex) {
        items.forEach((item, index) => {
            item.classList.toggle('active', index === activeIndex);
        });
    }

    function selectSuggestion(item) {
        const suggestion = item.textContent.trim();
        document.getElementById('search-input').value = suggestion;
        hideSearchSuggestions();
        updateFilterDisplay();
        autoApplyIfEnabled();
        loadHeatmap();
    }

    function highlightMatch(text, query) {
        const regex = new RegExp(`(${query})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }

    // Public method for selecting suggestions (called from HTML onclick)
    function selectSearchSuggestion(suggestion) {
        document.getElementById('search-input').value = suggestion;
        hideSearchSuggestions();
        updateFilterDisplay();
        autoApplyIfEnabled();
        loadHeatmap();
    }

    return {
        init,
        selectSearchSuggestion
    };
})();

// Make available globally
window.CompareDatasets = CompareDatasets;
