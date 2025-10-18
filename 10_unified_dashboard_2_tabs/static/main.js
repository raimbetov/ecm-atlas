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
        showLoading(true);

        const [stats, datasetsResp] = await Promise.all([
            fetchAPI('/api/global_stats'),
            fetchAPI('/api/datasets')
        ]);

        globalData.datasets = datasetsResp.datasets;
        renderGlobalStats(stats);
        populateDatasetSelector();

        showLoading(false);
    } catch (error) {
        console.error('Error loading global data:', error);
        showLoading(false);
        alert(`Error loading data. API is not responding at ${API_BASE}`);
    }
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
        const versionElement = document.getElementById('version-number');
        if (versionElement) {
            versionElement.textContent = 'unknown';
        }
    }
}

// Render global statistics
function renderGlobalStats(stats) {
    const container = document.getElementById('global-stats');
    container.innerHTML = `
        <div class="stat-item">📊 <strong>${stats.total_proteins}</strong> Proteins</div>
        <div class="stat-item">📚 <strong>${stats.datasets}</strong> Datasets</div>
        <div class="stat-item">🫀 <strong>${stats.organs}</strong> Organs</div>
        <div class="stat-item">🔬 <strong>${stats.compartments}</strong> Compartments</div>
        <div class="stat-item">🧬 <strong>${stats.ecm_proteins}</strong> ECM Proteins</div>
    `;
}

// Populate dataset selector
function populateDatasetSelector() {
    const select = document.getElementById('dataset-select');

    if (!select) return;

    select.innerHTML = '<option value="">Select a dataset...</option>';

    globalData.datasets.forEach(dataset => {
        const option = document.createElement('option');
        option.value = dataset.name;
        option.textContent = `${dataset.display_name} (${dataset.organ}) - ${dataset.protein_count} proteins`;
        select.appendChild(option);
    });

    select.addEventListener('change', (e) => {
        const datasetName = e.target.value;
        if (datasetName && window.IndividualDataset) {
            globalData.selectedDataset = datasetName;
            IndividualDataset.loadDataset(datasetName);
        }
    });
}

// Setup tab navigation
function setupTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-button');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

// Switch between tabs
function switchTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`tab-${tabName}`).classList.add('active');

    globalData.currentTab = tabName;

    // Initialize tab content if needed
    if (tabName === 'compare' && window.CompareDatasets) {
        CompareDatasets.init();
    } else if (tabName === 'individual' && window.IndividualDataset && globalData.selectedDataset) {
        IndividualDataset.loadDataset(globalData.selectedDataset);
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
function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (show) {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }
}

// Utility: Update active tab styling
function updateActiveTab(event, parentClass) {
    if (event && event.target) {
        const tabs = event.target.parentElement.querySelectorAll(`.${parentClass}`);
        tabs.forEach(tab => tab.classList.remove('active'));
        event.target.classList.add('active');
    }
}

// Export for use in other modules
window.ECMAtlas = {
    API_BASE,
    globalData,
    fetchAPI,
    showLoading,
    updateActiveTab
};
