// About Tab Functionality

// Initialize Mermaid with theme support
function initMermaid() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    mermaid.initialize({
        startOnLoad: false,
        theme: isDark ? 'dark' : 'default',
        flowchart: {
            useMaxWidth: true,
            htmlLabels: true,
            curve: 'basis'
        },
        themeVariables: isDark ? {
            background: '#1a1a1a',
            primaryColor: '#667eea',
            primaryTextColor: '#e5e5e5',
            primaryBorderColor: '#333333',
            lineColor: '#b0b0b0',
            secondaryColor: '#4caf50',
            tertiaryColor: '#ff9800'
        } : {}
    });
}

// Initialize Mermaid on load
initMermaid();

// Make initMermaid globally available for theme switching
window.initMermaid = initMermaid;

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

        // Generate table of contents after content is loaded
        generateTableOfContents();
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

// Load datasets list from API and populate the table in About tab
async function loadDatasetsList() {
    try {
        const response = await fetch(`${window.API_BASE}/api/datasets`);
        const data = await response.json();

        const tableBody = document.getElementById('datasets-table-body');
        if (!tableBody) return;

        // Create table rows for each dataset
        const datasetsHTML = data.datasets.map(dataset => `
            <tr onclick="window.open('datasets/${dataset.study_id}.html', '_blank')" style="cursor: pointer;">
                <td><a href="datasets/${dataset.study_id}.html" style="color: var(--accent); text-decoration: none; font-weight: 600;">${dataset.study_id}</a></td>
                <td>${dataset.year || 'N/A'}</td>
                <td>${dataset.tissue}</td>
                <td>${dataset.species}</td>
                <td>${dataset.compartments ? dataset.compartments.join(', ') : 'N/A'}</td>
                <td>${dataset.total_samples || 'N/A'}</td>
            </tr>
        `).join('');

        tableBody.innerHTML = datasetsHTML;

        // Update sidebar stats with real data
        updateSidebarStats(data.datasets);

        console.log(`Loaded ${data.datasets.length} datasets into table`);
    } catch (error) {
        console.error('Error loading datasets list:', error);
        const tableBody = document.getElementById('datasets-table-body');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">Error loading datasets</td></tr>';
        }
    }
}

// Generate Table of Contents from sections
function generateTableOfContents() {
    const sections = document.querySelectorAll('#about-content .section');
    const tocList = document.getElementById('toc-list');

    if (!tocList) return;

    tocList.innerHTML = '';

    sections.forEach(section => {
        const h2 = section.querySelector('h2');
        if (h2 && section.id) {
            const id = section.id;
            const title = h2.textContent.trim().replace('📊', '').replace('🔬', '').replace('⚡', '').replace('🧬', '').trim();

            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = `#${id}`;
            a.textContent = title;
            a.className = 'nav-link';

            // Attach click handler directly to the link
            a.addEventListener('click', (e) => {
                e.preventDefault();
                smoothScrollToSection(id);
                return false;
            });

            li.appendChild(a);
            tocList.appendChild(li);
        }
    });

    // Initialize scroll spy after TOC is generated
    initScrollSpy();
}

// Smooth scroll to section
function smoothScrollToSection(sectionId) {
    // Small delay to ensure DOM is fully updated
    setTimeout(() => {
        const section = document.getElementById(sectionId);
        if (section) {
            // Try scrollIntoView first, then fallback to manual scroll
            try {
                section.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });

                // Adjust for header after a brief delay
                setTimeout(() => {
                    const rect = section.getBoundingClientRect();
                    if (rect.top < 120) { // If section is too close to top
                        const additionalOffset = 120 - rect.top;
                        window.scrollBy({
                            top: -additionalOffset,
                            behavior: 'smooth'
                        });
                    }
                }, 300);
            } catch (e) {
                // Fallback to manual scroll
                const headerOffset = 120;
                const elementPosition = section.offsetTop;
                const offsetPosition = elementPosition - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }

            // Update URL hash without triggering scroll
            history.replaceState(null, null, `#${sectionId}`);
        }
    }, 100);
}

// Initialize scroll spy for TOC highlighting
function initScrollSpy() {
    const sections = document.querySelectorAll('#about-content .section');
    const tocList = document.getElementById('toc-list');

    function updateActiveTOC() {
        const scrollPosition = window.scrollY + 120; // Offset for header

        let currentSection = null;
        sections.forEach(section => {
            const rect = section.getBoundingClientRect();
            const sectionTop = rect.top + window.scrollY;
            const sectionHeight = section.offsetHeight;

            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                currentSection = section.id;
            }
        });

        // Update active class
        const tocLinks = tocList.querySelectorAll('a');
        tocLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSection}`) {
                link.classList.add('active');
            }
        });
    }

    // Throttle scroll events for better performance
    let scrollTimer;
    window.addEventListener('scroll', () => {
        clearTimeout(scrollTimer);
        scrollTimer = setTimeout(updateActiveTOC, 10);
    });

    // Initial update
    updateActiveTOC();
}

// Update sidebar stats from API data
function updateSidebarStats(datasets) {
    if (!datasets || !datasets.length) return;

    // Update counts
    const datasetsCount = datasets.length;
    const proteinsCount = 1106; // From existing data
    const organsCount = [...new Set(datasets.map(d => d.tissue))].length;
    const compartmentsCount = [...new Set(datasets.flatMap(d => d.compartments || []))].length;

    // Update sidebar elements
    const datasetsEl = document.getElementById('sidebar-datasets-count');
    const proteinsEl = document.getElementById('sidebar-proteins-count');
    const organsEl = document.getElementById('sidebar-organs-count');
    const compartmentsEl = document.getElementById('sidebar-compartments-count');

    if (datasetsEl) datasetsEl.textContent = datasetsCount.toLocaleString();
    if (proteinsEl) proteinsEl.textContent = proteinsCount.toLocaleString();
    if (organsEl) organsEl.textContent = organsCount;
    if (compartmentsEl) compartmentsEl.textContent = compartmentsCount;
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
