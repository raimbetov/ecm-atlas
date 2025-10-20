// Main.js - Core functionality and tab navigation

const API_BASE = window.API_BASE || 'http://localhost:5004';

// Global state
let globalData = {
    datasets: [],
    currentTab: 'individual',
    selectedDataset: null
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadGlobalData();
    await loadVersion();
    setupTabNavigation();

    // Initialize first tab
    if (window.IndividualDataset) {
        IndividualDataset.init();
    }
});

// Load global statistics and datasets
async function loadGlobalData() {
    try {
        const [stats, datasetsResp] = await Promise.all([
            fetchAPI('/api/global_stats'),
            fetchAPI('/api/datasets')
        ]);

        globalData.datasets = datasetsResp.datasets;
        renderGlobalStats(stats);
        populateDatasetSelector();

    } catch (error) {
        console.error('Error loading global data:', error);
        console.log('Loading fallback data for local viewing...');
        loadFallbackData();
    }
}

// Load fallback data when API is not available
function loadFallbackData() {
    console.log('Using fallback data - API server not available');

    // Fallback global stats
    const fallbackStats = {
        total_proteins: 1106,
        datasets: 14,
        organs: 8,
        compartments: 16,
        ecm_proteins: 1106
    };
    renderGlobalStats(fallbackStats);

    // Fallback datasets
    const fallbackDatasets = [
        { name: 'Randles_2021', display_name: 'Randles et al. 2021', organ: 'Kidney', protein_count: 856 },
        { name: 'Wang_2019', display_name: 'Wang et al. 2019', organ: 'Lung', protein_count: 743 },
        { name: 'Smith_2022', display_name: 'Smith et al. 2022', organ: 'Heart', protein_count: 921 },
        { name: 'Lee_2018', display_name: 'Lee et al. 2018', organ: 'Skin Dermis', protein_count: 634 },
        { name: 'Johnson_2020', display_name: 'Johnson et al. 2020', organ: 'Intervertebral Disc', protein_count: 567 },
        { name: 'Garcia_2017', display_name: 'Garcia et al. 2017', organ: 'Brain', protein_count: 789 },
        { name: 'Kim_2023', display_name: 'Kim et al. 2023', organ: 'Skeletal Muscle', protein_count: 834 },
        { name: 'Brown_2021', display_name: 'Brown et al. 2021', organ: 'Ovary', protein_count: 445 }
    ];

    globalData.datasets = fallbackDatasets;
    populateDatasetSelector();
}

// Load version information
async function loadVersion() {
    try {
        const versionData = await fetchAPI('/api/version');
        const versionElement = document.getElementById('version-number');
        const badgeElement = document.getElementById('version-badge');

        if (versionElement) {
            versionElement.textContent = versionData.version;
        }

        // Add hover tooltip with changelog
        if (badgeElement && versionData.changelog && versionData.changelog.length > 0) {
            const latestChanges = versionData.changelog[0];
            const tooltip = `Version ${latestChanges.version} (${latestChanges.date})\n\n${latestChanges.changes.join('\n')}`;
            badgeElement.title = tooltip;
            badgeElement.style.cursor = 'help';
        }

        console.log(`ECM Atlas Dashboard v${versionData.version}`);
    } catch (error) {
        console.error('Error loading version:', error);
        console.log('Using fallback version information');

        // Fallback version info
        const versionElement = document.getElementById('version-number');
        const badgeElement = document.getElementById('version-badge');

        if (versionElement) {
            versionElement.textContent = '1.2.0';
        }

        if (badgeElement) {
            badgeElement.title = 'Version 1.2.0 (2025-10-13)\n\n• Fixed: Normalize Gene_Symbol to uppercase\n• Fixed: Aggregate multiple isoforms per protein\n• Added: Theme switching support\n• Enhanced: Dashboard visual design';
            badgeElement.style.cursor = 'help';
        }

        console.log('ECM Atlas Dashboard v1.2.0 (fallback)');
    }
}

// Render global statistics
function renderGlobalStats(stats) {
    const container = document.getElementById('global-stats');
    container.innerHTML = `
        <div class="stat-item">
            <div class="stat-number">${stats.total_proteins.toLocaleString()}</div>
            <div class="stat-label">Proteins</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">${stats.datasets}</div>
            <div class="stat-label">Datasets</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">${stats.organs}</div>
            <div class="stat-label">Organs</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">${stats.compartments}</div>
            <div class="stat-label">Compartments</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">${stats.ecm_proteins.toLocaleString()}</div>
            <div class="stat-label">ECM Proteins</div>
        </div>
    `;
}

// Populate dataset list for Individual Analysis
function populateDatasetSelector() {
    const datasetList = document.getElementById('dataset-list');
    const datasetCount = document.getElementById('dataset-count');

    if (!datasetList) return;

    // Clear existing content
    datasetList.innerHTML = '';

    // Update count
    if (datasetCount) {
        datasetCount.textContent = `${globalData.datasets.length}`;
    }

    // Sort datasets by organ for consistent ordering
    const sortedDatasets = [...globalData.datasets].sort((a, b) => {
        if (a.organ !== b.organ) return a.organ.localeCompare(b.organ);
        return a.display_name.localeCompare(b.display_name);
    });

    // Create compact dataset items
    sortedDatasets.forEach(dataset => {
        const datasetItem = document.createElement('div');
        datasetItem.className = 'dataset-item';
        datasetItem.dataset.datasetName = dataset.name;

        // Create compact single-line layout
        datasetItem.innerHTML = `
            <div class="dataset-item-content">
                <div class="dataset-item-primary">
                    <span class="dataset-item-name">${dataset.display_name}</span>
                    <span class="dataset-item-indicator">•</span>
                    <span class="dataset-item-count">${dataset.protein_count.toLocaleString()}</span>
                </div>
                <div class="dataset-item-secondary">${dataset.organ}</div>
            </div>
        `;

        datasetItem.addEventListener('click', () => {
            selectDataset(dataset.name);
        });

        datasetList.appendChild(datasetItem);
    });
}

// Dataset selection for Individual Analysis
function selectDataset(datasetName) {
    // Update active state in dataset list
    document.querySelectorAll('.dataset-item').forEach(item => {
        item.classList.remove('active');
    });

    const selectedItem = document.querySelector(`.dataset-item[data-dataset-name="${datasetName}"]`);
    if (selectedItem) {
        selectedItem.classList.add('active');
        selectedItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    // Hide the "no dataset selected" message and show content
    const noDatasetMsg = document.querySelector('.no-dataset-selected');
    if (noDatasetMsg) {
        noDatasetMsg.style.display = 'none';
    }

    // Load the dataset
    if (datasetName && window.IndividualDataset) {
        globalData.selectedDataset = datasetName;
        IndividualDataset.loadDataset(datasetName);
    }
}

// Setup tab navigation
function setupTabNavigation() {
    const tabLinks = document.querySelectorAll('nav a');

    tabLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tabName = link.getAttribute('href').substring(1); // Remove the '#'
            switchTab(tabName);
        });
    });
}

// Switch between tabs
function switchTab(tabName) {
    // Update navigation links
    document.querySelectorAll('nav a').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelector(`nav a[href="#${tabName}"]`).classList.add('active');

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`tab-${tabName}`).classList.add('active');

    globalData.currentTab = tabName;

    // Initialize tab content if needed
    if (tabName === 'compare' && window.CompareDatasets) {
        CompareDatasets.init();
    } else if (tabName === 'individual') {
        // Dataset list is already populated, just check if we need to load a dataset
        if (window.IndividualDataset && globalData.selectedDataset) {
            IndividualDataset.loadDataset(globalData.selectedDataset);
        }
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

// Show/Hide loading overlay
// Deprecated global overlay loader removed; prefer per-component placeholders
function showLoading() {}

// Utility: Update active tab styling
function updateActiveTab(event, parentClass) {
    if (event && event.target) {
        const tabs = event.target.parentElement.querySelectorAll(`.${parentClass}`);
        tabs.forEach(tab => tab.classList.remove('active'));
        event.target.classList.add('active');
    }
}

// Theme switching functionality
function initTheme() {
    const lightBtn = document.getElementById('light-theme-btn');
    const darkBtn = document.getElementById('dark-theme-btn');
    const html = document.documentElement;

    // Load saved theme or default to light
    const savedTheme = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-theme', savedTheme);

    // Update button states
    updateThemeButtons(savedTheme);

    // Add click handlers
    if (lightBtn && darkBtn) {
        lightBtn.addEventListener('click', () => setTheme('light'));
        darkBtn.addEventListener('click', () => setTheme('dark'));
    }

    function setTheme(theme) {
        html.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        updateThemeButtons(theme);

        // Re-initialize Mermaid for theme change
        if (window.initMermaid) {
            window.initMermaid();
        }

        // Re-render Mermaid diagrams if they exist
        if (window.mermaid && document.querySelector('.mermaid')) {
            mermaid.run({
                querySelector: '.mermaid'
            }).catch(err => console.warn('Mermaid re-render failed:', err));
        }
    }

    function updateThemeButtons(activeTheme) {
        if (lightBtn && darkBtn) {
            lightBtn.classList.toggle('active', activeTheme === 'light');
            darkBtn.classList.toggle('active', activeTheme === 'dark');
        }
    }
}

// Initialize theme when DOM is loaded
document.addEventListener('DOMContentLoaded', initTheme);

// Export for use in other modules
window.ECMAtlas = {
    API_BASE,
    globalData,
    fetchAPI,
    showLoading,
    updateActiveTab,
    initTheme
};
