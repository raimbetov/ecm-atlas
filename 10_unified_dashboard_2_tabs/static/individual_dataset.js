// Individual Dataset Analysis Module

const IndividualDataset = (function() {
    let currentDataset = null;
    let currentCompartment = {};
    let datasetStats = null;

    function init() {
        console.log('Individual Dataset module initialized');
    }

    async function loadDataset(datasetName) {
        try {
            // Use per-chart placeholders only; no global overlay

            currentDataset = datasetName;

            // Load dataset stats
            const stats = await window.ECMAtlas.fetchAPI(`/api/dataset/${datasetName}/stats`);
            datasetStats = stats;

            // Display dataset info
            displayDatasetInfo(datasetName, stats);

            // Render content
            renderContent(stats);

            // Load initial visualizations for first compartment
            const firstCompartment = Object.keys(stats.compartments)[0];
            if (firstCompartment) {
                await loadAllVisualizations(firstCompartment);
            }

            
        } catch (error) {
            console.error('Error loading dataset:', error);
            alert(`Error loading dataset: ${error.message}`);
        }
    }

    function displayDatasetInfo(datasetName, stats) {
        const infoDiv = document.getElementById('dataset-info');
        const compartmentList = Object.keys(stats.compartments).map(comp => {
            const compStats = stats.compartments[comp];
            return `${comp} (${compStats.protein_count} proteins, ${compStats.ecm_count} ECM)`;
        }).join(', ');

        infoDiv.innerHTML = `
            <div class="dataset-info-section">
                <h3 class="dataset-title">${datasetName.replace(/_/g, ' ')}</h3>
                <div class="dataset-info-cards">
                    <div class="dataset-info-card">
                        <div class="dataset-info-label">Organ</div>
                        <div class="dataset-info-value">${stats.organ}</div>
                    </div>
                    <div class="dataset-info-card">
                        <div class="dataset-info-label">Total Proteins</div>
                        <div class="dataset-info-value">${stats.total_proteins.toLocaleString()}</div>
                    </div>
                    <div class="dataset-info-card">
                        <div class="dataset-info-label">ECM Proteins</div>
                        <div class="dataset-info-value">${stats.ecm_proteins.toLocaleString()}</div>
                    </div>
                </div>
                <div class="dataset-compartments-note">
                    <strong>Compartments:</strong> ${compartmentList}
                </div>
            </div>
        `;
    }

    function renderContent(stats) {
        const content = document.getElementById('individual-content');
        const compartments = Object.keys(stats.compartments);
        const hasMultipleCompartments = compartments.length > 1;

        content.innerHTML = `
            <!-- Statistics -->
            <div class="stats-container" id="dataset-stats"></div>

            <!-- Heatmap -->
            <div class="section">
                <h2>1. Heatmap: Top Aging-Associated Proteins</h2>
                <p>Top 100 proteins with largest absolute z-score changes</p>

                <!-- Z-Score Color Scale Legend -->
                <div style="margin: 30px 0; padding: 25px; background: linear-gradient(135deg, #f5f7fa 0%, #eef2f7 100%); border-radius: 12px; border-left: 6px solid #3b82f6; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);">
                    <strong style="display: block; margin-bottom: 20px; font-size: 18px; color: #1a1a1a;">Z-Score Color Scale</strong>
                    <div style="display: flex; flex-direction: column; gap: 20px;">
                        <!-- Color Bar -->
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <span style="display: inline-block; width: 100%; height: 50px; background: linear-gradient(to right, #0033ff, #0066ff, #66b3ff, #f7f7f7, #ffcc00, #ff6600, #ff0000); border-radius: 6px; border: 2px solid #ccc;"></span>
                        </div>
                        <!-- Legend Labels -->
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; padding: 0 10px;">
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                                <strong style="color: #0033ff; font-size: 18px;">-5</strong>
                                <span style="font-size: 13px; color: #555; font-weight: 600;">Decreased</span>
                            </div>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                                <strong style="color: #0066ff; font-size: 18px;">-2</strong>
                                <span style="font-size: 13px; color: #555;">Reduced</span>
                            </div>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                                <strong style="color: #808080; font-size: 18px;">0</strong>
                                <span style="font-size: 13px; color: #555; font-weight: 600;">No change</span>
                            </div>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                                <strong style="color: #ff6600; font-size: 18px;">+2</strong>
                                <span style="font-size: 13px; color: #555;">Increased</span>
                            </div>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px;">
                                <strong style="color: #ff0000; font-size: 18px;">+5</strong>
                                <span style="font-size: 13px; color: #555; font-weight: 600;">Highly Increased</span>
                            </div>
                        </div>
                    </div>
                </div>

                ${renderCompartmentTabs(compartments, 'heatmap')}
                <div id="heatmap-chart" class="chart-container tall">
                    <div class="loading">Select a compartment to view heatmap...</div>
                </div>
            </div>

            <!-- Volcano Plot -->
            <div class="section">
                <h2>2. Volcano Plot: Differential Expression</h2>
                <p>X-axis: Z-Score Change (Old - Young), Y-axis: -log10(Average Abundance)</p>
                ${renderCompartmentTabs(compartments, 'volcano')}
                <div id="volcano-chart" class="chart-container">
                    <div class="loading">Select a compartment to view volcano plot...</div>
                </div>
            </div>

            <!-- Scatter Plot -->
            <div class="section">
                <h2>3. Scatter Plot: Young vs Old Comparison</h2>
                <p>Direct comparison of z-scores between age groups. ECM proteins highlighted in red.</p>
                ${renderCompartmentTabs(compartments, 'scatter')}
                <div id="scatter-chart" class="chart-container">
                    <div class="loading">Select a compartment to view scatter plot...</div>
                </div>
            </div>

            <!-- Bar Chart -->
            <div class="section">
                <h2>4. Bar Chart: Top 20 Aging Markers</h2>
                <p>Top 10 increases and top 10 decreases with aging</p>
                ${renderCompartmentTabs(compartments, 'bars')}
                <div id="bars-chart" class="chart-container">
                    <div class="loading">Select a compartment to view bar chart...</div>
                </div>
            </div>

            <!-- Histograms -->
            <div class="section">
                <h2>5. Histograms: Distribution of Z-Score Changes</h2>
                <p>Overall distribution of aging-related changes</p>
                <div class="grid-${Math.min(compartments.length, 3)}">
                    ${compartments.map(comp => `
                        <div>
                            <h3 style="text-align: center; color: #667eea;">${comp}</h3>
                            <div id="histogram-${comp}" class="chart-container">
                                <div class="loading">Loading...</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>

            ${hasMultipleCompartments ? `
            <!-- Compartment Comparison -->
            <div class="section">
                <h2>6. Compartment Comparison</h2>
                <p>Correlation of aging changes between compartments</p>
                <div id="comparison-chart" class="chart-container">
                    <div class="loading">Loading comparison...</div>
                </div>
            </div>
            ` : ''}
        `;

        renderDatasetStats(stats);
        setupCompartmentTabs();
    }

    function renderCompartmentTabs(compartments, chartType) {
        return `
            <div class="compartment-tabs" data-chart="${chartType}">
                ${compartments.map((comp, idx) => `
                    <button class="tab ${idx === 0 ? 'active' : ''}"
                            data-compartment="${comp}"
                            data-chart="${chartType}">
                        ${comp}
                    </button>
                `).join('')}
            </div>
        `;
    }

    function setupCompartmentTabs() {
        document.querySelectorAll('.compartment-tabs .tab').forEach(button => {
            button.addEventListener('click', (e) => {
                const compartment = e.target.dataset.compartment;
                const chartType = e.target.dataset.chart;

                // Update active state
                window.ECMAtlas.updateActiveTab(e, 'tab');

                // Load visualization
                loadVisualization(chartType, compartment);
            });
        });
    }

    function renderDatasetStats(stats) {
        const container = document.getElementById('dataset-stats');
        const compartments = Object.keys(stats.compartments);

        container.innerHTML = `
            <div class="stats-grid">
                ${compartments.map(comp => {
                    const compStats = stats.compartments[comp];
                    return `
                        <div class="stat-item">
                            <span class="stat-number">${compStats.protein_count.toLocaleString()}</span>
                            <span class="stat-label">${comp} Proteins</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">${compStats.ecm_count.toLocaleString()}</span>
                            <span class="stat-label">${comp} ECM</span>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    async function loadAllVisualizations(compartment) {
        currentCompartment = {
            heatmap: compartment,
            volcano: compartment,
            scatter: compartment,
            bars: compartment
        };

        await Promise.all([
            loadHeatmap(compartment),
            loadVolcano(compartment),
            loadScatter(compartment),
            loadBars(compartment)
        ]);

        // Load histograms for all compartments
        const compartments = Object.keys(datasetStats.compartments);
        await Promise.all(
            compartments.map(comp => loadHistogram(comp))
        );

        // Load comparison if multiple compartments
        if (compartments.length > 1) {
            await loadComparison();
        }
    }

    async function loadVisualization(type, compartment) {
        switch (type) {
            case 'heatmap':
                await loadHeatmap(compartment);
                break;
            case 'volcano':
                await loadVolcano(compartment);
                break;
            case 'scatter':
                await loadScatter(compartment);
                break;
            case 'bars':
                await loadBars(compartment);
                break;
        }
    }

    async function loadHeatmap(compartment) {
        const container = document.getElementById('heatmap-chart');
        container.innerHTML = '<div class="loading">Loading heatmap...</div>';

        try {
            const data = await window.ECMAtlas.fetchAPI(
                `/api/dataset/${currentDataset}/heatmap/${compartment}?n=100`
            );

            const trace = {
                type: 'heatmap',
                z: data.genes.map((gene, i) => [data.zscore_young[i], data.zscore_old[i]]),
                y: data.genes,
                x: ['Young', 'Old'],
                colorscale: [
                    [0.0, '#0033ff'],  // Vivid blue (-5, downregulated)
                    [0.2, '#0066ff'],  // Bright blue (-3)
                    [0.35, '#66b3ff'], // Light blue (-1)
                    [0.5, '#f7f7f7'],  // Off-white (0, no change)
                    [0.65, '#ffcc00'], // Bright yellow (+1)
                    [0.8, '#ff6600'],  // Vibrant orange (+3)
                    [1.0, '#ff0000']   // Vivid red (+5, upregulated)
                ],
                zmid: 0,
                zmin: -5,
                zmax: 5,
                showscale: false,  // Remove side colorbar; use top legend instead
                hovertemplate: '<b>%{y}</b><br>Age: %{x}<br>Z-Score: %{z:.2f}<extra></extra>'
            };

            const layout = {
                title: `${data.compartment}: Top 100 Aging-Associated Proteins`,
                xaxis: { title: 'Age Group', side: 'bottom' },
                yaxis: { title: 'Gene Symbol', tickfont: { size: 8 } },
                height: 1200,
                margin: { l: 100, r: 100, t: 80, b: 80 }
            };

            // Clear placeholder and render
            container.innerHTML = '';
            Plotly.newPlot(container, [trace], layout, { responsive: true });
        } catch (error) {
            container.innerHTML = `<div class="error">Failed to load heatmap: ${error.message}</div>`;
        }
    }

    async function loadVolcano(compartment) {
        const container = document.getElementById('volcano-chart');
        container.innerHTML = '<div class="loading">Loading volcano plot...</div>';

        try {
            const data = await window.ECMAtlas.fetchAPI(
                `/api/dataset/${currentDataset}/volcano/${compartment}`
            );

            const trace = {
                type: 'scatter',
                mode: 'markers',
                x: data.zscore_delta,
                y: data.neglog_abundance,
                text: data.genes,
                marker: {
                    size: 6,
                    color: data.zscore_delta,
                    colorscale: [
                        [0.0, '#0033ff'],  // Vivid blue (downregulated)
                        [0.2, '#0066ff'],  // Bright blue
                        [0.35, '#66b3ff'], // Light blue
                        [0.5, '#f7f7f7'],  // Off-white (neutral)
                        [0.65, '#ffcc00'], // Bright yellow
                        [0.8, '#ff6600'],  // Vibrant orange
                        [1.0, '#ff0000']   // Vivid red (upregulated)
                    ],
                    showscale: false  // Remove side colorbar
                },
                hovertemplate: '<b>%{text}</b><br>ΔZ-Score: %{x:.2f}<br>-log10(Abundance): %{y:.2f}<extra></extra>'
            };

            const layout = {
                title: `${compartment}: Volcano Plot`,
                xaxis: {
                    title: 'Z-Score Change (Old - Young)',
                    zeroline: true,
                    zerolinewidth: 2,
                    zerolinecolor: 'gray'
                },
                yaxis: { title: '-log10(Average Abundance + 1)' },
                height: 600,
                hovermode: 'closest'
            };

            // Clear placeholder and render
            container.innerHTML = '';
            Plotly.newPlot(container, [trace], layout, { responsive: true });
        } catch (error) {
            container.innerHTML = `<div class="error">Failed to load volcano plot: ${error.message}</div>`;
        }
    }

    async function loadScatter(compartment) {
        const container = document.getElementById('scatter-chart');
        container.innerHTML = '<div class="loading">Loading scatter plot...</div>';

        try {
            const data = await window.ECMAtlas.fetchAPI(
                `/api/dataset/${currentDataset}/scatter/${compartment}`
            );

            // Separate ECM and non-ECM
            const ecmIndices = data.is_ecm.map((isEcm, i) => isEcm ? i : -1).filter(i => i !== -1);
            const nonEcmIndices = data.is_ecm.map((isEcm, i) => !isEcm ? i : -1).filter(i => i !== -1);

            const traces = [
                {
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Non-ECM',
                    x: nonEcmIndices.map(i => data.zscore_young[i]),
                    y: nonEcmIndices.map(i => data.zscore_old[i]),
                    text: nonEcmIndices.map(i => data.genes[i]),
                    marker: { size: 6, color: 'lightblue', opacity: 0.6 },
                    hovertemplate: '<b>%{text}</b><br>Young: %{x:.2f}<br>Old: %{y:.2f}<extra></extra>'
                },
                {
                    type: 'scatter',
                    mode: 'markers',
                    name: 'ECM Proteins',
                    x: ecmIndices.map(i => data.zscore_young[i]),
                    y: ecmIndices.map(i => data.zscore_old[i]),
                    text: ecmIndices.map(i => `${data.genes[i]} (${data.matrisome_category[i]})`),
                    marker: { size: 8, color: 'red', opacity: 0.8 },
                    hovertemplate: '<b>%{text}</b><br>Young: %{x:.2f}<br>Old: %{y:.2f}<extra></extra>'
                }
            ];

            // Diagonal reference line
            // Filter out null values before calculating min/max
            const allValues = [...data.zscore_young, ...data.zscore_old].filter(v => v !== null && v !== undefined && !isNaN(v));
            const minVal = Math.min(...allValues);
            const maxVal = Math.max(...allValues);
            traces.push({
                type: 'scatter',
                mode: 'lines',
                name: 'No Change',
                x: [minVal, maxVal],
                y: [minVal, maxVal],
                line: { color: 'gray', dash: 'dash', width: 2 },
                hoverinfo: 'skip'
            });

            const layout = {
                title: `${compartment}: Young vs Old Z-Scores`,
                xaxis: { title: 'Z-Score Young', zeroline: true },
                yaxis: { title: 'Z-Score Old', zeroline: true },
                height: 600,
                hovermode: 'closest'
            };

            // Clear placeholder and render
            container.innerHTML = '';
            Plotly.newPlot(container, traces, layout, { responsive: true });
        } catch (error) {
            container.innerHTML = `<div class="error">Failed to load scatter plot: ${error.message}</div>`;
        }
    }

    async function loadBars(compartment) {
        const container = document.getElementById('bars-chart');
        container.innerHTML = '<div class="loading">Loading bar chart...</div>';

        try {
            const data = await window.ECMAtlas.fetchAPI(
                `/api/dataset/${currentDataset}/bars/${compartment}`
            );

            const colors = data.zscore_delta.map(val => val > 0 ? '#d32f2f' : '#1976d2');

            const trace = {
                type: 'bar',
                orientation: 'h',
                y: data.genes,
                x: data.zscore_delta,
                marker: { color: colors },
                text: data.zscore_delta.map(v => v.toFixed(2)),
                textposition: 'auto',
                hovertemplate: '<b>%{y}</b><br>ΔZ-Score: %{x:.2f}<br>Category: %{customdata}<extra></extra>',
                customdata: data.matrisome_category
            };

            const layout = {
                title: `${compartment}: Top 20 Aging Markers`,
                xaxis: {
                    title: 'Z-Score Change (Old - Young)',
                    zeroline: true,
                    zerolinewidth: 2,
                    zerolinecolor: 'gray'
                },
                yaxis: { title: 'Gene Symbol' },
                height: 600,
                margin: { l: 100 }
            };

            // Clear placeholder and render
            container.innerHTML = '';
            Plotly.newPlot(container, [trace], layout, { responsive: true });
        } catch (error) {
            container.innerHTML = `<div class="error">Failed to load bar chart: ${error.message}</div>`;
        }
    }

    async function loadHistogram(compartment) {
        const containerId = `histogram-${compartment}`;
        const container = document.getElementById(containerId);

        if (!container) return;

        try {
            const data = await window.ECMAtlas.fetchAPI(
                `/api/dataset/${currentDataset}/histogram/${compartment}`
            );

            const trace = {
                type: 'histogram',
                x: data.zscore_delta,
                nbinsx: 50,
                marker: {
                    color: '#667eea',
                    line: { color: 'white', width: 1 }
                },
                hovertemplate: 'ΔZ-Score Range: %{x}<br>Count: %{y}<extra></extra>'
            };

            const layout = {
                title: `${compartment}: Distribution`,
                xaxis: { title: 'Z-Score Change', zeroline: true },
                yaxis: { title: 'Frequency' },
                height: 500,
                shapes: [{
                    type: 'line',
                    x0: data.mean_delta,
                    x1: data.mean_delta,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: { color: 'red', width: 2, dash: 'dash' }
                }],
                annotations: [{
                    x: data.mean_delta,
                    y: 0.95,
                    yref: 'paper',
                    text: `Mean: ${data.mean_delta.toFixed(3)}`,
                    showarrow: false,
                    bgcolor: 'rgba(255, 255, 255, 0.8)',
                    bordercolor: 'red',
                    borderwidth: 1
                }]
            };

            // Clear placeholder and render
            container.innerHTML = '';
            Plotly.newPlot(container, [trace], layout, { responsive: true });
        } catch (error) {
            container.innerHTML = `<div class="error">Failed to load histogram: ${error.message}</div>`;
        }
    }

    async function loadComparison() {
        const container = document.getElementById('comparison-chart');
        if (!container) return;

        try {
            const data = await window.ECMAtlas.fetchAPI(
                `/api/dataset/${currentDataset}/comparison`
            );

            // Group by matrisome category
            const categories = [...new Set(data.matrisome_category)];
            const colors = {
                'Non-ECM': 'lightgray',
                'Core matrisome': '#e53935',
                'Matrisome-associated': '#1e88e5',
                'ECM Glycoproteins': '#43a047',
                'Collagens': '#fdd835',
                'Proteoglycans': '#8e24aa'
            };

            const traces = categories.map(cat => {
                const indices = data.matrisome_category.map((c, i) => c === cat ? i : -1).filter(i => i !== -1);
                return {
                    type: 'scatter',
                    mode: 'markers',
                    name: cat,
                    x: indices.map(i => data.zscore_delta_comp1[i]),
                    y: indices.map(i => data.zscore_delta_comp2[i]),
                    text: indices.map(i => data.genes[i]),
                    marker: {
                        size: cat === 'Non-ECM' ? 4 : 8,
                        color: colors[cat] || 'gray',
                        opacity: cat === 'Non-ECM' ? 0.3 : 0.7
                    },
                    hovertemplate: `<b>%{text}</b><br>${data.compartment1}: %{x:.2f}<br>${data.compartment2}: %{y:.2f}<extra></extra>`
                };
            });

            // Diagonal reference line
            // Filter out null values before calculating min/max
            const allValues = [...data.zscore_delta_comp1, ...data.zscore_delta_comp2].filter(v => v !== null && v !== undefined && !isNaN(v));
            const minVal = Math.min(...allValues);
            const maxVal = Math.max(...allValues);
            traces.push({
                type: 'scatter',
                mode: 'lines',
                name: 'Perfect Correlation',
                x: [minVal, maxVal],
                y: [minVal, maxVal],
                line: { color: 'black', dash: 'dash', width: 2 },
                hoverinfo: 'skip'
            });

            const layout = {
                title: `Compartment Comparison: ${data.compartment1} vs ${data.compartment2}`,
                xaxis: {
                    title: `${data.compartment1} ΔZ-Score`,
                    zeroline: true,
                    zerolinewidth: 2,
                    zerolinecolor: 'gray'
                },
                yaxis: {
                    title: `${data.compartment2} ΔZ-Score`,
                    zeroline: true,
                    zerolinewidth: 2,
                    zerolinecolor: 'gray'
                },
                height: 600,
                hovermode: 'closest'
            };

            // Clear placeholder and render
            container.innerHTML = '';
            Plotly.newPlot(container, traces, layout, { responsive: true });
        } catch (error) {
            container.innerHTML = `<div class="error">Failed to load comparison: ${error.message}</div>`;
        }
    }

    return {
        init,
        loadDataset
    };
})();

// Make available globally
window.IndividualDataset = IndividualDataset;
