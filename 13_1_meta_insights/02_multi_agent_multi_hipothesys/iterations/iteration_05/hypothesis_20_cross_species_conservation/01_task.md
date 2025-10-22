# H20 – Cross-Species Conservation: Validating Universal vs Species-Specific Aging Mechanisms

## Scientific Question
Are the top discoveries from H01-H19 (S100→crosslinking pathway, metabolic-mechanical transition, SERPINE1 centrality) conserved across species (mouse, rat, C. elegans), or are they human-specific artifacts?

## Background & Rationale

**Critical Validation Question:**
- ALL findings H01-H15 based on HUMAN and MOUSE data (13 studies)
- **Risk:** Mechanisms may be species-specific, limiting:
  - Preclinical mouse model validity for human trials
  - Evolutionary understanding of ECM aging
  - Drug target generalizability (does SERPINE1 inhibitor work in mice?)

**Why Cross-Species Matters:**
1. **Evolutionary Conservation:** If S100→LOX conserved in worms/flies → ancient mechanism, high confidence
2. **Preclinical Translation:** If mouse findings replicate in rats → robust animal model
3. **Drug Development:** FDA requires animal validation; species conservation de-risks trials
4. **Fundamental Biology:** Species-specific = niche adaptation; conserved = core aging process

**Expected Patterns:**
- **Universal Mechanisms:** Conserved across ≥3 species (human, mouse, C. elegans)
  - Examples: TGM2 crosslinking, S100 calcium signaling (ancient pathways)
- **Mammal-Specific:** Present in human/mouse/rat, absent in worms/flies
  - Examples: Complex collagens (COL1A1 gene duplications)
- **Human-Specific:** Only in our data, not in mouse
  - Examples: SERPINE1 centrality (if mouse uses different serpin)

**Available Resources:**
- Mouse Aging Proteomics: Mouse Aging Cell Atlas, NIA Interventions Testing Program
- Rat Proteomics: Sprague-Dawley aging studies
- C. elegans: WormBase, aging intervention screens
- Drosophila: FlyBase, lifespan GWAS

**Clinical Impact:**
If S100 pathway conserved → validates mouse models, accelerates trials. If human-specific → need human organoids/tissues for testing.

## Objectives

### Primary Objective
Download and analyze mouse/rat/C. elegans aging proteomics datasets, test conservation of (1) S100→crosslinking pathway (H08), (2) metabolic-mechanical transition (H12), (3) SERPINE1 centrality (H14), identifying universal vs species-specific mechanisms.

### Secondary Objectives
1. Search databases for mouse, rat, C. elegans, Drosophila aging proteomics
2. Download ≥2 species datasets (preferably mouse + C. elegans for evolutionary span)
3. Ortholog mapping (human genes → mouse/worm equivalents)
4. Test H08 S100→LOX/TGM2 correlations in each species
5. Test H12 Phase I/II transition (velocity threshold conservation)
6. Test H14 SERPINE1 centrality (network hub status)
7. Evolutionary rate analysis (conserved proteins evolve slowly)

## Hypotheses to Test

### H20.1: S100 Pathway Conservation (H08)
S100A9, S100A10, S100B correlate with TGM2/LOX in mouse (ρ>0.60) and orthologous pathways in C. elegans (ρ>0.40), confirming ancient calcium→crosslinking mechanism.

### H20.2: Metabolic-Mechanical Transition Conservation (H12)
Mouse exhibits similar velocity threshold (v_mouse=1.5-2.0) separating metabolic from mechanical phases; C. elegans shows NO transition (worms lack complex ECM).

### H20.3: SERPINE1 Centrality Conservation (H14)
Mouse serpin ortholog (Serpine1) shows high eigenvector centrality (>0.80); C. elegans serpin (sri-40) does NOT (centrality <0.50), indicating mammal-specific network role.

### H20.4: Evolutionary Rate Correlation
Conserved proteins (present in all species) have lower dN/dS ratios (<0.5), evolving slowly due to functional constraint; species-specific proteins have dN/dS>1.0 (rapid evolution).

## Required Analyses

### 1. DATABASE SEARCH & SPECIES SELECTION

**Mouse Aging Proteomics:**
```python
# PRIDE database search
from pyteomics import pride

# Search for mouse aging studies
mouse_studies = pride.search(
    query="mouse aging proteomics ECM",
    species="Mus musculus"
)

print(f"Found {len(mouse_studies)} mouse aging proteomics studies")

# Target datasets:
# - Mouse Aging Cell Atlas (Tabula Muris Senis)
# - NIA Interventions Testing Program
# - Specific tissues: heart, liver, muscle (match human data)
```

**C. elegans Proteomics:**
```python
# WormBase, PRIDE
worm_studies = pride.search(
    query="C. elegans aging proteomics",
    species="Caenorhabditis elegans"
)

# Alternative: Search PeptideAtlas
# C. elegans has ~20,000 genes, many with human orthologs
```

**Rat Proteomics:**
```python
rat_studies = pride.search(
    query="rat aging proteomics",
    species="Rattus norvegicus"
)
```

**Success Criteria:**
- ≥1 mouse dataset with ≥2 tissues matching human (heart, liver, muscle)
- ≥1 C. elegans dataset (whole organism acceptable, no tissue separation)
- Optional: Rat dataset for mammalian robustness

### 2. ORTHOLOG MAPPING

**Human → Mouse Gene Mapping:**
```python
from biomart import BiomartServer

server = BiomartServer("http://www.ensembl.org/biomart")
dataset = server.datasets['hsapiens_gene_ensembl']

# Get orthologs for all 648 ECM genes
human_genes = ['S100A9', 'S100A10', 'S100B', 'LOX', 'TGM2', 'COL1A1', 'SERPINE1', ...]  # All 648

orthologs_human_mouse = {}

for gene in human_genes:
    response = dataset.search({
        'filters': {'external_gene_name': gene},
        'attributes': [
            'external_gene_name',
            'mmusculus_homolog_associated_gene_name',
            'mmusculus_homolog_orthology_type',
            'mmusculus_homolog_perc_id'
        ]
    })

    for line in response.iter_lines():
        data = line.decode('utf-8').split('\t')
        if len(data) >= 4:
            orthologs_human_mouse[gene] = {
                'mouse_gene': data[1],
                'orthology_type': data[2],
                'percent_identity': float(data[3]) if data[3] else 0
            }

# Save mapping
df_orthologs = pd.DataFrame(orthologs_human_mouse).T
df_orthologs.to_csv(f'orthologs_human_mouse_{agent}.csv')

print(f"Mapped {len(orthologs_human_mouse)}/{len(human_genes)} genes to mouse orthologs")
# Expected: ~90% mapping (some human genes have no mouse ortholog, e.g., recent duplications)
```

**Human → C. elegans Mapping:**
```python
# Use WormBase orthology
# Example: S100A9 → no direct ortholog (S100 is vertebrate-specific)
#          TGM2 → tgm-1 (transglutaminase ortholog)
#          LOX → lox-1, lox-2, lox-3

# Lower expected mapping: ~50% (worms lack many mammalian ECM genes)
```

**Success Criteria:**
- Human→Mouse: ≥85% genes mapped (1:1 orthologs)
- Human→C. elegans: ≥40% mapped (ancient core genes)

### 3. S100 PATHWAY CONSERVATION TEST (H08)

**Mouse Data:**
```python
# Load mouse proteomics (preprocessed like human)
df_mouse = pd.read_csv('mouse_aging_proteomics_zscore.csv')

# Map to mouse gene names
s100_mouse_genes = [orthologs_human_mouse.get(g, {}).get('mouse_gene') for g in s100_genes]
s100_mouse_genes = [g for g in s100_mouse_genes if g is not None]  # Remove unmapped

lox_mouse_genes = ['Lox', 'Loxl1', 'Loxl2', 'Loxl3', 'Loxl4']
tgm_mouse_genes = ['Tgm2', 'Tgm3']

# Test correlations (same as H08 analysis)
from scipy.stats import spearmanr

# S100A9 (mouse: S100a9) → TGM2 (mouse: Tgm2)
s100a9_mouse = df_mouse[df_mouse['Gene_Symbol'] == 'S100a9']['Z_score']
tgm2_mouse = df_mouse[df_mouse['Gene_Symbol'] == 'Tgm2']['Z_score']

# Align samples
merged_mouse = pd.merge(
    df_mouse[df_mouse['Gene_Symbol'] == 'S100a9'][['Sample_ID', 'Z_score']].rename(columns={'Z_score': 'S100a9'}),
    df_mouse[df_mouse['Gene_Symbol'] == 'Tgm2'][['Sample_ID', 'Z_score']].rename(columns={'Z_score': 'Tgm2'}),
    on='Sample_ID'
)

rho_mouse, p_mouse = spearmanr(merged_mouse['S100a9'], merged_mouse['Tgm2'])

print(f"Mouse S100a9→Tgm2: ρ={rho_mouse:.3f}, p={p_mouse:.4f}")
# Compare to human: ρ=0.79 (H08 Claude)

# Repeat for all S100→LOX/TGM pairs
```

**C. elegans Data:**
```python
# C. elegans may lack S100 (vertebrate innovation)
# Test ancestral crosslinking pathway: TGM → collagens

df_worm = pd.read_csv('celegans_aging_proteomics_zscore.csv')

# tgm-1 (TGM2 ortholog) → col-19 (COL1A1 ortholog)
tgm1_worm = df_worm[df_worm['Gene_Symbol'] == 'tgm-1']['Z_score']
col19_worm = df_worm[df_worm['Gene_Symbol'] == 'col-19']['Z_score']

rho_worm, p_worm = spearmanr(tgm1_worm, col19_worm)
print(f"C. elegans tgm-1→col-19: ρ={rho_worm:.3f}, p={p_worm:.4f}")
```

**Conservation Summary:**
```python
s100_conservation = {
    'Human': {'S100A9→TGM2': 0.79, 'S100B→LOX': 0.74},  # From H08
    'Mouse': {'S100a9→Tgm2': rho_mouse, 'S100b→Lox': rho_s100b_lox_mouse},
    'C. elegans': {'tgm-1→col-19': rho_worm}  # S100 absent
}

df_s100_conservation = pd.DataFrame(s100_conservation).T
df_s100_conservation.to_csv(f's100_pathway_conservation_{agent}.csv')
```

**Success Criteria:**
- Mouse: ≥2/3 S100→LOX/TGM correlations ρ>0.60 (conserved)
- C. elegans: TGM→collagen ρ>0.40 (ancestral crosslinking), S100 absent

### 4. METABOLIC-MECHANICAL TRANSITION (H12)

**Mouse Tissue Velocities:**
```python
# Calculate mouse velocities (same method as H03)
mouse_tissues = df_mouse['Tissue'].unique()

mouse_velocities = {}

for tissue in mouse_tissues:
    tissue_data = df_mouse[df_mouse['Tissue'] == tissue]
    velocity = tissue_data['Z_score'].abs().mean()
    mouse_velocities[tissue] = velocity

# Sort
mouse_velocities_sorted = dict(sorted(mouse_velocities.items(), key=lambda x: x[1], reverse=True))
print("Mouse tissue velocities:")
for tissue, v in mouse_velocities_sorted.items():
    print(f"  {tissue}: {v:.2f}")
```

**Test for Transition Threshold:**
```python
# H12 human transition: v=1.65-2.17
# Does mouse have similar threshold?

# Method 1: Visual inspection (plot Phase I vs Phase II markers)
# Method 2: Changepoint detection

from ruptures import Pelt

# Fit changepoint model to sorted velocities
velocities_array = np.array(list(mouse_velocities_sorted.values()))
model = Pelt(model="rbf").fit(velocities_array)
changepoints = model.predict(pen=3)  # Penalty parameter

print(f"Mouse velocity changepoints: {changepoints}")
# Expected: ~1-2 changepoints (Phase I/transition/Phase II)

mouse_transition_zone = velocities_array[changepoints[0]-1:changepoints[0]+1] if changepoints else None
print(f"Mouse transition zone: v={mouse_transition_zone}")
# Compare to human: v=1.65-2.17
```

**C. elegans (NO Transition Expected):**
```python
# C. elegans has minimal ECM (no bones, cartilage, complex connective tissues)
# Expect: NO clear Phase I/Phase II separation

# If dataset is whole-organism (no tissues), cannot compute velocity
# Alternative: check if collagen/crosslinking markers show distinct aging phases

# Plot collagen abundance vs age
worm_collagens = ['col-19', 'col-34', 'col-99', 'col-120']
for col in worm_collagens:
    col_data = df_worm[df_worm['Gene_Symbol'] == col]
    plt.scatter(col_data['Age'], col_data['Z_score'], label=col)

plt.xlabel('Age (days)')
plt.ylabel('Z-score')
plt.title('C. elegans Collagen Abundance vs Age')
plt.legend()
plt.savefig(f'visualizations_{agent}/worm_collagen_aging_{agent}.png', dpi=300)

# Expected: Linear/monotonic increase (no transition)
```

**Success Criteria:**
- Mouse: Transition zone detected at v=1.4-2.0 (similar to human 1.65-2.17)
- C. elegans: NO transition (monotonic aging, no Phase I/II separation)

### 5. SERPINE1 CENTRALITY CONSERVATION (H14)

**Mouse Network:**
```python
# Build mouse protein-protein network (same as H05)
from scipy.stats import spearmanr

# Correlation network
mouse_proteins = df_mouse['Gene_Symbol'].unique()
n_mouse_proteins = len(mouse_proteins)

mouse_corr_matrix = np.zeros((n_mouse_proteins, n_mouse_proteins))

for i, gene1 in enumerate(mouse_proteins):
    for j, gene2 in enumerate(mouse_proteins):
        if i >= j:
            continue

        vals1 = df_mouse[df_mouse['Gene_Symbol'] == gene1]['Z_score']
        vals2 = df_mouse[df_mouse['Gene_Symbol'] == gene2]['Z_score']

        # Align samples
        merged = pd.merge(
            df_mouse[df_mouse['Gene_Symbol'] == gene1][['Sample_ID', 'Z_score']].rename(columns={'Z_score': 'val1'}),
            df_mouse[df_mouse['Gene_Symbol'] == gene2][['Sample_ID', 'Z_score']].rename(columns={'Z_score': 'val2'}),
            on='Sample_ID'
        )

        if len(merged) >= 10:
            rho, p = spearmanr(merged['val1'], merged['val2'])
            if abs(rho) > 0.6:  # Threshold from H05
                mouse_corr_matrix[i, j] = rho
                mouse_corr_matrix[j, i] = rho

# Network centrality
import networkx as nx

G_mouse = nx.from_numpy_array(mouse_corr_matrix)
G_mouse = nx.relabel_nodes(G_mouse, {i: gene for i, gene in enumerate(mouse_proteins)})

# Eigenvector centrality (H14 validated metric)
mouse_centrality = nx.eigenvector_centrality(G_mouse, max_iter=1000)

# Check Serpine1
serpine1_centrality_mouse = mouse_centrality.get('Serpine1', 0)
print(f"Mouse Serpine1 centrality: {serpine1_centrality_mouse:.3f}")
# Compare to human: 0.891 (H14)

# Top 10 mouse central proteins
top_mouse_central = sorted(mouse_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 mouse central proteins:")
for gene, cent in top_mouse_central:
    print(f"  {gene}: {cent:.3f}")
```

**C. elegans Network:**
```python
# Build C. elegans network
# Check sri-40 (serpin ortholog)

worm_centrality = nx.eigenvector_centrality(G_worm, max_iter=1000)
sri40_centrality_worm = worm_centrality.get('sri-40', 0)

print(f"C. elegans sri-40 centrality: {sri40_centrality_worm:.3f}")
# Expected: Lower than mammalian serpins (worms use different coagulation/ECM regulation)
```

**Conservation Table:**
```python
centrality_conservation = {
    'Species': ['Human', 'Mouse', 'C. elegans'],
    'Serpin_Gene': ['SERPINE1', 'Serpine1', 'sri-40'],
    'Eigenvector_Centrality': [0.891, serpine1_centrality_mouse, sri40_centrality_worm],
    'Rank': [1, rank_serpine1_mouse, rank_sri40_worm]  # Rank among all proteins
}

df_centrality_conservation = pd.DataFrame(centrality_conservation)
df_centrality_conservation.to_csv(f'serpine1_centrality_conservation_{agent}.csv')
```

**Success Criteria:**
- Mouse Serpine1: Centrality >0.80, rank in top 5 (conserved hub status)
- C. elegans sri-40: Centrality <0.50 (not a hub, different mechanism)

### 6. EVOLUTIONARY RATE ANALYSIS (dN/dS)

**Conserved vs Species-Specific Proteins:**
```python
# dN/dS (non-synonymous / synonymous substitution rate)
# Low dN/dS (<0.5) → strong purifying selection (conserved function)
# High dN/dS (>1.0) → positive selection or relaxed constraint

# Use Ensembl Compara API
import requests

def get_dnds(human_gene, mouse_gene):
    url = f"https://rest.ensembl.org/homology/symbol/human/{human_gene}"
    params = {'target_species': 'mouse', 'content-type': 'application/json'}

    response = requests.get(url, params=params)
    if response.ok:
        data = response.json()
        for homology in data['data'][0]['homologies']:
            if homology['target']['id'] == mouse_gene:
                dnds = homology.get('dn_ds')
                return dnds
    return None

# Test conserved proteins (S100A9, TGM2, LOX)
conserved_proteins = {
    'S100A9': 'S100a9',
    'TGM2': 'Tgm2',
    'LOX': 'Lox',
    'COL1A1': 'Col1a1',
    'SERPINE1': 'Serpine1'
}

dnds_values = {}

for human_gene, mouse_gene in conserved_proteins.items():
    dnds = get_dnds(human_gene, mouse_gene)
    dnds_values[human_gene] = dnds
    print(f"{human_gene}: dN/dS = {dnds}")

# Expected: S100A9, TGM2, LOX have dN/dS < 0.5 (conserved)
#           Species-specific proteins have dN/dS > 1.0
```

**Success Criteria:**
- Conserved proteins (present in all 3 species): dN/dS <0.5
- Species-specific proteins: dN/dS >1.0

### 7. COMPARATIVE SUMMARY

**Universal vs Species-Specific Table:**
```python
summary_conservation = {
    'Mechanism': [
        'S100→TGM2/LOX pathway',
        'Metabolic-Mechanical Transition',
        'SERPINE1 Centrality',
        'Collagen Accumulation',
        'Crosslinking Enzymes (LOX/TGM)'
    ],
    'Human': ['YES (ρ=0.79)', 'YES (v=1.65-2.17)', 'YES (cent=0.89)', 'YES', 'YES'],
    'Mouse': [
        f"{'YES' if rho_mouse > 0.6 else 'NO'} (ρ={rho_mouse:.2f})",
        f"{'YES' if mouse_transition_zone else 'NO'}",
        f"{'YES' if serpine1_centrality_mouse > 0.8 else 'NO'} (cent={serpine1_centrality_mouse:.2f})",
        'YES',
        'YES'
    ],
    'C. elegans': [
        f"PARTIAL (no S100, TGM conserved)",
        'NO (no transition)',
        f"NO (sri-40 cent={sri40_centrality_worm:.2f})",
        'YES (collagens present)',
        'YES (tgm-1)'
    ],
    'Classification': [
        'Mammal-specific (S100) + Ancient (TGM)',
        'Mammal-specific (complex ECM)',
        'Mammal-specific (serpin hub)',
        'Universal',
        'Universal'
    ]
}

df_conservation_summary = pd.DataFrame(summary_conservation)
df_conservation_summary.to_csv(f'conservation_summary_{agent}.csv')
```

## Deliverables

### Code & Models
- `species_database_search_{agent}.py` — Search PRIDE, WormBase for mouse/worm datasets
- `ortholog_mapping_{agent}.py` — Human→mouse→C. elegans gene mapping
- `s100_pathway_cross_species_{agent}.py` — Test H08 correlations in each species
- `transition_cross_species_{agent}.py` — Test H12 Phase I/II in mouse, worm
- `centrality_cross_species_{agent}.py` — Test H14 SERPINE1 hub status
- `evolutionary_rate_{agent}.py` — dN/dS analysis for conserved proteins

### Data Tables
- `species_datasets_{agent}.csv` — Downloaded datasets (species, n_samples, tissues)
- `orthologs_human_mouse_{agent}.csv` — Gene mapping (648 human → mouse)
- `orthologs_human_worm_{agent}.csv` — Gene mapping (human → C. elegans)
- `s100_pathway_conservation_{agent}.csv` — Correlation ρ for S100→LOX/TGM per species
- `mouse_velocities_{agent}.csv` — Tissue velocities, transition zone
- `serpine1_centrality_conservation_{agent}.csv` — Centrality per species
- `dnds_conserved_proteins_{agent}.csv` — Evolutionary rates
- `conservation_summary_{agent}.csv` — Universal vs species-specific mechanisms

### Visualizations
- `visualizations_{agent}/species_comparison_{agent}.png` — Heatmap (mechanisms × species)
- `visualizations_{agent}/s100_pathway_correlation_{agent}.png` — Side-by-side ρ (human vs mouse vs worm)
- `visualizations_{agent}/velocity_transition_{agent}.png` — Human vs mouse transition zones
- `visualizations_{agent}/centrality_barplot_{agent}.png` — SERPINE1/Serpine1/sri-40 centrality comparison
- `visualizations_{agent}/dnds_distribution_{agent}.png` — Histogram (conserved vs species-specific)
- `visualizations_{agent}/evolutionary_tree_{agent}.png` — Phylogenetic tree with conserved proteins highlighted

### Report
- `90_results_{agent}.md` — CRITICAL findings:
  - **Dataset Success:** Which species datasets downloaded? Sample sizes?
  - **S100 Pathway:** Conserved in mouse? Absent in worms (vertebrate innovation)?
  - **Metabolic Transition:** Mouse has similar threshold (v~1.5-2.0)? Worms lack transition?
  - **SERPINE1 Hub:** Mammal-specific central role? Worms use different mechanism?
  - **Universal Mechanisms:** Which proteins/pathways conserved across all species?
  - **Species-Specific:** Which mechanisms unique to mammals or humans?
  - **FINAL VERDICT:** Can we trust mouse models? Which targets are evolutionarily ancient (high confidence)?
  - **Recommendations:** Focus drug development on UNIVERSAL targets (TGM, LOX)? Human-specific targets (SERPINE1) need human organoids?

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Species datasets downloaded | ≥2 (mouse + 1 other) | PRIDE, WormBase |
| Human→mouse ortholog mapping | ≥85% | Ensembl Compara |
| Human→worm ortholog mapping | ≥40% | WormBase |
| Mouse S100→TGM correlation | ρ>0.60 (conserved) | Spearman test |
| Mouse transition zone | v=1.4-2.0 (similar to human) | Changepoint detection |
| Mouse Serpine1 centrality | >0.80 (conserved hub) | Network analysis |
| C. elegans TGM→collagen | ρ>0.40 (ancestral) | Spearman test |
| Conserved proteins dN/dS | <0.5 | Ensembl API |
| Overall CONSERVATION | ≥6/8 criteria met | Comprehensive |

## Expected Outcomes

### Scenario 1: STRONG CONSERVATION (Mouse Models Valid)
- Mouse: S100→TGM2 ρ=0.71, transition v=1.52-1.98, Serpine1 centrality=0.85
- All mammalian mechanisms conserved
- C. elegans: TGM→collagen ρ=0.48 (ancestral crosslinking), S100 absent (vertebrate innovation)
- dN/dS: S100A9=0.32, TGM2=0.41, LOX=0.38 (strong constraint)
- **Action:** Validate mouse models, proceed with SERPINE1 inhibitor mouse trials, high confidence translation to humans

### Scenario 2: PARTIAL CONSERVATION (Mixed)
- Mouse: S100 pathway YES, transition YES, but Serpine1 centrality=0.52 (NOT a hub in mice)
- Some mechanisms conserved, others mouse-specific
- **Action:** Use mouse for S100/TGM testing, but SERPINE1 requires human organoids

### Scenario 3: POOR CONSERVATION (Species-Specific)
- Mouse shows DIFFERENT mechanisms (S100→TGM ρ=0.18, no transition)
- Human findings may be artifacts or human-specific adaptations
- **Action:** Re-evaluate all H01-H19 with skepticism, prioritize human cell/organoid models

### Scenario 4: DATA UNAVAILABLE
- No mouse aging ECM proteomics found, only serum/plasma
- **Action:** Generate new mouse tissue proteomics (expensive, 12-18 months) or use existing databases with limited overlap

## Dataset

**Primary (Human):**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**External (to download):**
- PRIDE: Mouse aging proteomics (https://www.ebi.ac.uk/pride/)
- WormBase: C. elegans proteomics (https://wormbase.org/)
- FlyBase: Drosophila (optional) (https://flybase.org/)
- Ensembl Compara: Ortholog/dN/dS data (https://www.ensembl.org/)

**Reference:**
- H03 Velocities: `/iterations/iteration_01/hypothesis_03_tissue_specific_inflammation/`
- H08 S100 Pathway: `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/`
- H12 Transition: `/iterations/iteration_04/hypothesis_12_metabolic_mechanical_transition/`
- H14 SERPINE1 Centrality: `/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/`

## References

1. **Mouse Aging Cell Atlas**: Tabula Muris Senis Consortium, Nature 2020
2. **C. elegans Aging Proteomics**: Walther et al., Molecular Systems Biology 2015
3. **Evolutionary Rates**: Yang & Nielsen, Molecular Biology & Evolution 2000
4. **Ensembl Compara**: Herrero et al., Database 2016
5. **WormBase**: Harris et al., Nucleic Acids Research 2020
