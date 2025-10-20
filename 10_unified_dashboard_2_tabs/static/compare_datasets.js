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

        // Render filters in the left panel with enhanced UX
        const filtersContent = document.getElementById('filters-content');
        filtersContent.innerHTML = `
            <!-- Active Filters Display -->
            <div id="active-filters" class="active-filters-container collapsible collapsed" style="display: none;">
                <div class="active-filters-header">
                    <span class="active-filters-title">Active Filters</span>
                    <div class="active-filters-actions">
                        <button id="clear-all-active" class="clear-active-btn">Clear All</button>
                        <button id="toggle-active-filters" class="toggle-active-btn">▶</button>
                    </div>
                </div>
                <div id="active-filter-chips" class="active-filter-chips" style="display: none;"></div>
            </div>

            <!-- Compartments by Organ -->
            <div class="filter-group collapsible expanded">
                <div class="filter-group-header">
                    <h3>Compartments</h3>
                    <span class="filter-count" id="compartment-count" style="display: none;"></span>
                </div>
                <div class="filter-group-content">
                    <div id="compartment-filters"></div>
                </div>
            </div>

            <!-- Categories -->
            <div class="filter-group collapsible expanded">
                <div class="filter-group-header">
                    <h3>Matrisome Categories</h3>
                    <span class="filter-count" id="category-count" style="display: none;"></span>
                </div>
                <div class="filter-group-content">
                    <div id="category-filters"></div>
                </div>
            </div>

            <!-- Studies -->
            <div class="filter-group collapsible">
                <div class="filter-group-header">
                    <h3>Studies</h3>
                    <span class="filter-count" id="study-count" style="display: none;"></span>
                </div>
                <div class="filter-group-content">
                    <div id="study-filters"></div>
                </div>
            </div>

            <!-- Trend -->
            <div class="filter-group collapsible expanded">
                <div class="filter-group-header">
                    <h3>Aging Trend</h3>
                    <span class="filter-count" id="trend-count" style="display: none;"></span>
                </div>
                <div class="filter-group-content">
                    <div id="trend-filters">
                        <label class="trend-option"><input type="checkbox" value="up" class="trend-checkbox" /> <span class="trend-label up">Increased (&gt;+0.5)</span></label>
                        <label class="trend-option"><input type="checkbox" value="down" class="trend-checkbox" /> <span class="trend-label down">Decreased (&lt;-0.5)</span></label>
                        <label class="trend-option"><input type="checkbox" value="stable" class="trend-checkbox" /> <span class="trend-label stable">Stable (±0.5)</span></label>
                    </div>
                </div>
            </div>

            <!-- Filter Controls -->
            <div class="filter-controls">
                <div class="control-row">
                    <label class="control-option">
                        <input type="checkbox" id="auto-apply-toggle" checked />
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

        // Clear all filters
        document.getElementById('clear-filters').addEventListener('click', () => {
            clearAllFilters();
        });

        // Clear active filters
        document.getElementById('clear-all-active')?.addEventListener('click', () => {
            clearAllFilters();
        });

        // Toggle active filters
        document.getElementById('toggle-active-filters')?.addEventListener('click', () => {
            toggleActiveFilters();
        });

        // Auto-apply toggle
        document.getElementById('auto-apply-toggle').addEventListener('change', (e) => {
            const autoApply = e.target.checked;
            if (autoApply) {
                // Auto-apply immediately when changed
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

        // Organ checkbox toggle all compartments
        document.getElementById('compartment-filters').addEventListener('change', (e) => {
            if (e.target.classList.contains('organ-checkbox')) {
                const organ = e.target.dataset.organ;
                const checked = e.target.checked;
                document.querySelectorAll(`.compartment-checkbox[data-organ="${organ}"]`).forEach(cb => {
                    cb.checked = checked;
                });
                updateFilterDisplay();
                autoApplyIfEnabled();
            }
        });

        // Individual filter changes
        document.addEventListener('change', (e) => {
            if (e.target.matches('.compartment-checkbox, .category-checkbox, .study-checkbox, .trend-checkbox')) {
                updateFilterDisplay();
                autoApplyIfEnabled();
            }
        });

        // Collapsible filter groups
        document.addEventListener('click', (e) => {
            if (e.target.closest('.filter-group-header')) {
                const header = e.target.closest('.filter-group-header');
                const group = header.parentElement;
                toggleFilterGroup(group);
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
            }, 500);
        }
    }

    function updateFilterDisplay() {
        const activeFilters = getActiveFilters();
        updateActiveFilterChips(activeFilters);
        updateFilterCounts(activeFilters);
        updateActiveFiltersVisibility(activeFilters);
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

    function updateActiveFilterChips(filters) {
        const chipsContainer = document.getElementById('active-filter-chips');
        const chips = [];

        // Search chip
        if (filters.search) {
            chips.push(createFilterChip('search', `${filters.search}`, 'search'));
        }

        // Compartment chips
        filters.compartments.forEach(comp => {
            chips.push(createFilterChip('compartment', `${comp.label} (${comp.organ})`, comp.value, comp.organ));
        });

        // Category chips
        filters.categories.forEach(cat => {
            chips.push(createFilterChip('category', cat.label, cat.value));
        });

        // Study chips
        filters.studies.forEach(study => {
            chips.push(createFilterChip('study', study.label, study.value));
        });

        // Trend chips
        filters.trends.forEach(trend => {
            chips.push(createFilterChip('trend', trend.label, trend.value));
        });

        chipsContainer.innerHTML = chips.join('');
    }

    function createFilterChip(type, label, value, organ = null) {
        const colorClass = getChipColorClass(type);
        const removeHandler = organ ?
            `removeFilterChip('${type}', '${value}', '${organ}')` :
            `removeFilterChip('${type}', '${value}')`;

        return `
            <div class="filter-chip ${colorClass}" data-type="${type}" data-value="${value}" ${organ ? `data-organ="${organ}"` : ''}>
                <span class="chip-label">${label}</span>
                <button class="chip-remove" onclick="${removeHandler}" aria-label="Remove filter">×</button>
            </div>
        `;
    }

    function getChipColorClass(type) {
        const colors = {
            search: 'chip-search',
            compartment: 'chip-compartment',
            category: 'chip-category',
            study: 'chip-study',
            trend: 'chip-trend'
        };
        return colors[type] || 'chip-default';
    }

    function removeFilterChip(type, value, organ = null) {
        switch (type) {
            case 'search':
                document.getElementById('search-input').value = '';
                break;
            case 'compartment':
                const compartmentCheckbox = document.querySelector(`.compartment-checkbox[value="${value}"][data-organ="${organ}"]`);
                if (compartmentCheckbox) compartmentCheckbox.checked = false;
                break;
            case 'category':
                const categoryCheckbox = document.querySelector(`.category-checkbox[value="${value}"]`);
                if (categoryCheckbox) categoryCheckbox.checked = false;
                break;
            case 'study':
                const studyCheckbox = document.querySelector(`.study-checkbox[value="${value}"]`);
                if (studyCheckbox) studyCheckbox.checked = false;
                break;
            case 'trend':
                const trendCheckbox = document.querySelector(`.trend-checkbox[value="${value}"]`);
                if (trendCheckbox) trendCheckbox.checked = false;
                break;
        }

        updateFilterDisplay();
        autoApplyIfEnabled();
        loadHeatmap(); // Always apply when removing chips
    }

    function updateFilterCounts(filters) {
        // Update count badges for each filter group
        document.getElementById('search-count').textContent = filters.search ? '1' : '';
        document.getElementById('search-count').style.display = filters.search ? 'inline' : 'none';

        document.getElementById('compartment-count').textContent = filters.compartments.length || '';
        document.getElementById('compartment-count').style.display = filters.compartments.length ? 'inline' : 'none';

        document.getElementById('category-count').textContent = filters.categories.length || '';
        document.getElementById('category-count').style.display = filters.categories.length ? 'inline' : 'none';

        document.getElementById('study-count').textContent = filters.studies.length || '';
        document.getElementById('study-count').style.display = filters.studies.length ? 'inline' : 'none';

        document.getElementById('trend-count').textContent = filters.trends.length || '';
        document.getElementById('trend-count').style.display = filters.trends.length ? 'inline' : 'none';
    }

    function updateActiveFiltersVisibility(filters) {
        const activeContainer = document.getElementById('active-filters');
        const chipsContainer = document.getElementById('active-filter-chips');
        const toggleBtn = document.getElementById('toggle-active-filters');

        const hasActiveFilters = filters.search || filters.compartments.length || filters.categories.length ||
                                filters.studies.length || filters.trends.length;

        if (hasActiveFilters) {
            activeContainer.style.display = 'block';
            // Keep collapsed state (chips hidden) by default when first shown
            if (!activeContainer.classList.contains('expanded') && !activeContainer.classList.contains('collapsed')) {
                activeContainer.classList.add('collapsed');
                if (chipsContainer) chipsContainer.style.display = 'none';
                if (toggleBtn) toggleBtn.textContent = '▶';
            }
        } else {
            activeContainer.style.display = 'none';
        }
    }

    function toggleActiveFilters() {
        const activeContainer = document.getElementById('active-filters');
        const chipsContainer = document.getElementById('active-filter-chips');
        const toggleBtn = document.getElementById('toggle-active-filters');

        if (activeContainer.classList.contains('collapsed')) {
            // Expand
            activeContainer.classList.remove('collapsed');
            activeContainer.classList.add('expanded');
            chipsContainer.style.display = 'flex';
            toggleBtn.textContent = '▼';
        } else {
            // Collapse
            activeContainer.classList.remove('expanded');
            activeContainer.classList.add('collapsed');
            chipsContainer.style.display = 'none';
            toggleBtn.textContent = '▶';
        }
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

    function toggleFilterGroup(group) {
        group.classList.toggle('expanded');
        const content = group.querySelector('.filter-group-content');
        if (content) {
            content.style.display = group.classList.contains('expanded') ? 'block' : 'none';
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
