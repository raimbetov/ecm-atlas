// About Tab Functionality

// Initialize Mermaid
mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
    }
});

// Load About content when page loads
async function loadAboutContent() {
    try {
        const response = await fetch('static/about_content.html');
        const html = await response.text();
        document.getElementById('about-content').innerHTML = html;

        // Render all Mermaid diagrams after content is loaded
        await mermaid.run({
            querySelector: '.mermaid'
        });

        console.log('About content loaded and Mermaid diagrams rendered');
    } catch (error) {
        console.error('Error loading about content:', error);
        document.getElementById('about-content').innerHTML = `
            <div class="error">
                <h3>Error loading content</h3>
                <p>Could not load About content. Please refresh the page.</p>
                <pre>${error.message}</pre>
            </div>
        `;
    }
}

// Load datasets list from API and add to About tab
async function loadDatasetsList() {
    try {
        const response = await fetch(`${window.API_BASE}/api/datasets`);
        const data = await response.json();

        // Create datasets list HTML
        const datasetsHTML = `
            <div class="section" style="margin-top: 30px;">
                <h2>📚 Available Datasets</h2>
                <p>Click on any dataset to view detailed processing information and original publication.</p>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                    ${data.datasets.map(dataset => `
                        <div class="dataset-card" style="background: white; border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; cursor: pointer; transition: all 0.3s;"
                             onclick="window.open('datasets/${dataset.study_id}.html', '_blank')">
                            <h3 style="margin: 0 0 10px 0; color: #667eea; font-size: 1.2em;">${dataset.study_id}</h3>
                            <p style="margin: 5px 0; font-size: 0.9em;"><strong>Tissue:</strong> ${dataset.tissue}</p>
                            <p style="margin: 5px 0; font-size: 0.9em;"><strong>Species:</strong> ${dataset.species}</p>
                            <p style="margin: 5px 0; font-size: 0.9em;"><strong>Samples:</strong> ${dataset.total_samples || 'N/A'}</p>
                            ${dataset.compartments ? `<p style="margin: 5px 0; font-size: 0.9em;"><strong>Compartments:</strong> ${dataset.compartments.join(', ')}</p>` : ''}
                            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e0e0e0; color: #667eea; font-weight: 600; font-size: 0.9em;">
                                View Details →
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        // Append to about content
        document.getElementById('about-content').innerHTML += datasetsHTML;

        console.log(`Loaded ${data.datasets.length} datasets`);
    } catch (error) {
        console.error('Error loading datasets list:', error);
    }
}

// Initialize About tab when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    await loadAboutContent();
    await loadDatasetsList();
});

// Add hover effects for dataset cards
document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = `
        .dataset-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2) !important;
            border-color: #667eea !important;
        }
    `;
    document.head.appendChild(style);
});
