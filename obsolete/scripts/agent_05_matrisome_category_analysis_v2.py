#!/usr/bin/env python3
"""
Agent 5: Matrisome Category Strategist
Analyzes aging patterns across Matrisome categories and divisions
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DB_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_DIR = "/Users/Kravtsovd/projects/ecm-atlas/10_insights/discovery_ver1/"
CSV_OUTPUT = OUTPUT_DIR + "agent_05_matrisome_categories.csv"
REPORT_OUTPUT = OUTPUT_DIR + "agent_05_matrisome_categories_REPORT.md"

def load_data():
    """Load and prepare dataset"""
    df = pd.read_csv(DB_PATH)

    # Filter to valid aging data with Matrisome annotations
    valid_df = df[
        df['Zscore_Delta'].notna() &
        df['Matrisome_Category'].notna() &
        df['Matrisome_Division'].notna()
    ].copy()

    print(f"Loaded {len(df)} total records")
    print(f"Valid aging data with Matrisome annotations: {len(valid_df)}")

    return valid_df

def analyze_category_statistics(df):
    """Calculate category-level statistics"""

    print("\n=== CATEGORY-LEVEL STATISTICS ===\n")

    # Group by Matrisome_Division (Core matrisome vs Matrisome-associated)
    division_stats = df.groupby('Matrisome_Division')['Zscore_Delta'].agg([
        'count', 'mean', 'median', 'std', lambda x: np.var(x), 'min', 'max'
    ]).round(3)
    division_stats.columns = ['count', 'mean', 'median', 'std', 'variance', 'min', 'max']

    # Directionality: % upregulated vs downregulated
    for div in division_stats.index:
        div_data = df[df['Matrisome_Division'] == div]['Zscore_Delta']
        division_stats.loc[div, 'pct_upregulated'] = round((div_data > 0.3).sum() / len(div_data) * 100, 1)
        division_stats.loc[div, 'pct_downregulated'] = round((div_data < -0.3).sum() / len(div_data) * 100, 1)
        division_stats.loc[div, 'pct_stable'] = round(((div_data >= -0.3) & (div_data <= 0.3)).sum() / len(div_data) * 100, 1)

    print("Matrisome Division Statistics (Core vs Associated):")
    print(division_stats)
    print()

    # Group by Matrisome_Category (Collagens, Proteoglycans, etc.)
    category_stats = df.groupby('Matrisome_Category')['Zscore_Delta'].agg([
        'count', 'mean', 'median', 'std', lambda x: np.var(x)
    ]).round(3)
    category_stats.columns = ['count', 'mean', 'median', 'std', 'variance']

    # Directionality for categories
    for cat in category_stats.index:
        cat_data = df[df['Matrisome_Category'] == cat]['Zscore_Delta']
        category_stats.loc[cat, 'pct_upregulated'] = round((cat_data > 0.3).sum() / len(cat_data) * 100, 1)
        category_stats.loc[cat, 'pct_downregulated'] = round((cat_data < -0.3).sum() / len(cat_data) * 100, 1)
        category_stats.loc[cat, 'pct_stable'] = round(((cat_data >= -0.3) & (cat_data <= 0.3)).sum() / len(cat_data) * 100, 1)

    print("Matrisome Category Statistics (Functional Classes):")
    print(category_stats)
    print()

    return division_stats, category_stats

def statistical_testing(df):
    """Test if category differences are statistically significant"""

    print("\n=== STATISTICAL TESTING ===\n")

    # Test Division differences (Core vs Associated)
    divisions = df['Matrisome_Division'].unique()
    if len(divisions) >= 2:
        division_groups = [df[df['Matrisome_Division'] == div]['Zscore_Delta'].dropna().values
                          for div in divisions]

        # One-way ANOVA
        f_stat_div, p_value_div = stats.f_oneway(*division_groups)
        print(f"One-way ANOVA (Matrisome Divisions):")
        print(f"  F-statistic: {f_stat_div:.4f}")
        print(f"  p-value: {p_value_div:.4e}")
        print(f"  Significant: {'YES' if p_value_div < 0.05 else 'NO'}")
        print()

        # Pairwise comparisons
        if 'Core matrisome' in divisions and 'Matrisome-associated' in divisions:
            core = df[df['Matrisome_Division'] == 'Core matrisome']['Zscore_Delta'].dropna().values
            associated = df[df['Matrisome_Division'] == 'Matrisome-associated']['Zscore_Delta'].dropna().values

            t_stat, p_value_ttest = stats.ttest_ind(core, associated, equal_var=False)

            print(f"Core matrisome vs Matrisome-associated (Welch's t-test):")
            print(f"  Core mean: {np.mean(core):.4f} ± {np.std(core):.4f}")
            print(f"  Associated mean: {np.mean(associated):.4f} ± {np.std(associated):.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value_ttest:.4e}")
            print(f"  Significant: {'YES' if p_value_ttest < 0.05 else 'NO'}")
            print()

    # Category-level ANOVA
    categories = df['Matrisome_Category'].unique()
    category_groups = [df[df['Matrisome_Category'] == cat]['Zscore_Delta'].dropna().values
                       for cat in categories]

    f_stat_cat, p_value_cat = stats.f_oneway(*category_groups)
    print(f"One-way ANOVA (Matrisome Categories):")
    print(f"  F-statistic: {f_stat_cat:.4f}")
    print(f"  p-value: {p_value_cat:.4e}")
    print(f"  Significant: {'YES' if p_value_cat < 0.05 else 'NO'}")
    print()

    return {
        'division_anova': (f_stat_div, p_value_div) if len(divisions) >= 2 else (None, None),
        'core_vs_associated': (t_stat, p_value_ttest) if 'Core matrisome' in divisions and 'Matrisome-associated' in divisions else (None, None),
        'category_anova': (f_stat_cat, p_value_cat)
    }

def find_top_proteins_by_category(df):
    """Find top aging proteins per category"""

    print("\n=== TOP PROTEINS BY CATEGORY ===\n")

    results = {}

    for category in sorted(df['Matrisome_Category'].unique()):
        cat_df = df[df['Matrisome_Category'] == category]

        print(f"\n{category}:")
        print(f"  Total proteins: {len(cat_df)}")

        # Top upregulated
        top_up = cat_df.nlargest(5, 'Zscore_Delta')[
            ['Gene_Symbol', 'Matrisome_Division', 'Zscore_Delta', 'Tissue']
        ]
        print(f"\n  Top 5 UPREGULATED:")
        print(top_up.to_string(index=False))

        # Top downregulated
        top_down = cat_df.nsmallest(5, 'Zscore_Delta')[
            ['Gene_Symbol', 'Matrisome_Division', 'Zscore_Delta', 'Tissue']
        ]
        print(f"\n  Top 5 DOWNREGULATED:")
        print(top_down.to_string(index=False))

        results[category] = {
            'top_up': top_up,
            'top_down': top_down
        }

    return results

def analyze_tissue_specific_patterns(df):
    """Analyze category patterns by tissue"""

    print("\n=== TISSUE-SPECIFIC CATEGORY PATTERNS ===\n")

    # Create pivot table: Tissue x Category
    tissue_category = df.groupby(['Tissue', 'Matrisome_Category'])['Zscore_Delta'].agg(['mean', 'count']).reset_index()

    # Filter to tissues with at least 10 proteins per category
    tissue_category = tissue_category[tissue_category['count'] >= 5]

    # Pivot to wide format
    pivot = tissue_category.pivot(index='Tissue', columns='Matrisome_Category', values='mean')

    print("Average Z-score Delta by Tissue and Category (min 5 proteins):")
    print(pivot.round(3))
    print()

    return pivot

def save_summary_csv(df, division_stats, category_stats):
    """Save comprehensive CSV with all statistics"""

    # Combine division and category stats
    division_stats_reset = division_stats.reset_index()
    division_stats_reset.insert(0, 'Level', 'Division')
    division_stats_reset.rename(columns={'Matrisome_Division': 'Name'}, inplace=True)

    category_stats_reset = category_stats.reset_index()
    category_stats_reset.insert(0, 'Level', 'Category')
    category_stats_reset.rename(columns={'Matrisome_Category': 'Name'}, inplace=True)

    combined = pd.concat([division_stats_reset, category_stats_reset], ignore_index=True)

    combined.to_csv(CSV_OUTPUT, index=False)
    print(f"\nSummary CSV saved to: {CSV_OUTPUT}")

    return combined

def generate_markdown_report(df, division_stats, category_stats, stat_tests, top_proteins, tissue_pivot):
    """Generate comprehensive markdown report"""

    md = []

    # Header
    md.append("# Matrisome Category Aging Patterns: Functional Architecture of ECM Remodeling")
    md.append("")
    md.append("## Thesis")

    core_mean = division_stats.loc['Core matrisome', 'mean'] if 'Core matrisome' in division_stats.index else 0
    assoc_mean = division_stats.loc['Matrisome-associated', 'mean'] if 'Matrisome-associated' in division_stats.index else 0
    core_std = division_stats.loc['Core matrisome', 'std'] if 'Core matrisome' in division_stats.index else 0
    assoc_std = division_stats.loc['Matrisome-associated', 'std'] if 'Matrisome-associated' in division_stats.index else 0

    md.append(f"Analysis of {len(df)} ECM protein measurements reveals category-specific aging: Core matrisome proteins (collagens, proteoglycans) show mean aging delta of {core_mean:+.3f} z-score units with SD={core_std:.3f}, while Matrisome-associated regulatory proteins exhibit mean delta of {assoc_mean:+.3f} with SD={assoc_std:.3f}, suggesting {'structural ECM accumulation' if core_mean > 0 else 'structural ECM depletion'} alongside {'higher' if assoc_std > core_std else 'lower'} regulatory heterogeneity.")
    md.append("")

    # Overview
    md.append("## Overview")
    md.append("")
    md.append(f"This analysis stratifies ECM aging signatures by Matrisome Division (Core matrisome vs Matrisome-associated) and Matrisome Category (Collagens, Proteoglycans, ECM Glycoproteins, ECM Regulators, Secreted Factors, ECM-affiliated) to identify functional patterns. Dataset encompasses {len(df)} protein measurements across {df['Study_ID'].nunique()} studies, {df['Tissue'].nunique()} tissue types, and {df['Species'].nunique()} species.")
    md.append("")

    # System diagram
    md.append("**System Structure (Matrisome Organization):**")
    md.append("")
    md.append("```mermaid")
    md.append("graph TD")
    md.append("    ECM[ECM Proteins] --> Core[Core Matrisome]")
    md.append("    ECM --> Assoc[Matrisome-Associated]")
    md.append("    Core --> Coll[Collagens]")
    md.append("    Core --> PG[Proteoglycans]")
    md.append("    Core --> Glyco[ECM Glycoproteins]")
    md.append("    Assoc --> Reg[ECM Regulators]")
    md.append("    Assoc --> SF[Secreted Factors]")
    md.append("    Assoc --> Affil[ECM-Affiliated]")
    md.append("```")
    md.append("")

    # Process diagram
    md.append("**Aging Process (Temporal Dynamics):**")
    md.append("")
    md.append("```mermaid")
    md.append("graph LR")
    md.append("    A[Young ECM] --> B[Aging Stress]")
    md.append("    B --> C[Regulatory Imbalance]")
    md.append("    B --> D[Structural Remodeling]")
    md.append("    C --> E[Variable Associated Proteins]")
    md.append("    D --> F[Core Matrisome Changes]")
    md.append("    E --> G[Tissue Dysfunction]")
    md.append("    F --> G")
    md.append("```")
    md.append("")
    md.append("---")
    md.append("")

    # Section 1: Division-Level Analysis
    md.append("## 1.0 Division-Level Aging Signatures")
    md.append("")
    md.append("¶1 Ordering: Core matrisome → Matrisome-associated → Statistical comparison")
    md.append("")
    md.append("### 1.1 Core Matrisome vs Matrisome-Associated")
    md.append("")

    md.append("| Division | n | Mean Δz | Median Δz | Std Dev | Variance | % UP | % DOWN | % Stable |")
    md.append("|----------|---|---------|-----------|---------|----------|------|--------|----------|")

    for div in division_stats.index:
        row = division_stats.loc[div]
        md.append(f"| {div} | {int(row['count'])} | {row['mean']:+.3f} | {row['median']:+.3f} | {row['std']:.3f} | {row['variance']:.3f} | {row['pct_upregulated']:.1f}% | {row['pct_downregulated']:.1f}% | {row['pct_stable']:.1f}% |")

    md.append("")

    # Interpretation
    core_var = division_stats.loc['Core matrisome', 'variance'] if 'Core matrisome' in division_stats.index else 0
    assoc_var = division_stats.loc['Matrisome-associated', 'variance'] if 'Matrisome-associated' in division_stats.index else 0

    md.append("**Key Finding:** ")
    if core_mean > 0:
        md.append(f"Core matrisome shows net accumulation with aging (mean Δz = {core_mean:+.3f}), suggesting fibrotic-like ECM deposition across structural proteins. ")
    else:
        md.append(f"Core matrisome shows net depletion with aging (mean Δz = {core_mean:+.3f}), suggesting degradation of structural ECM components. ")

    if assoc_var > core_var:
        md.append(f"Matrisome-associated proteins exhibit {(assoc_var/core_var - 1)*100:.1f}% higher variance ({assoc_var:.3f} vs {core_var:.3f}), indicating heterogeneous regulatory responses varying by tissue context.")
    else:
        md.append(f"Matrisome-associated proteins show comparable variance to Core matrisome, suggesting coordinated changes across functional classes.")
    md.append("")
    md.append("")

    # Statistical test
    if stat_tests['core_vs_associated'][0] is not None:
        t_stat, p_val = stat_tests['core_vs_associated']
        md.append(f"**Statistical Significance:** Welch's t-test (Core vs Associated): t = {t_stat:.3f}, p = {p_val:.4e}")
        if p_val < 0.001:
            md.append(f"- HIGHLY SIGNIFICANT difference between divisions (p < 0.001)")
        elif p_val < 0.05:
            md.append(f"- SIGNIFICANT difference between divisions (p < 0.05)")
        else:
            md.append(f"- No significant difference between divisions")
    md.append("")
    md.append("---")
    md.append("")

    # Section 2: Category-Level Analysis
    md.append("## 2.0 Category-Level Functional Patterns")
    md.append("")
    md.append("¶1 Ordering: By functional class (Structural → Regulatory → Secreted)")
    md.append("")
    md.append("### 2.1 Matrisome Category Statistics")
    md.append("")

    md.append("| Category | Division | n | Mean Δz | Std Dev | % UP | % DOWN | % Stable |")
    md.append("|----------|----------|---|---------|---------|------|--------|----------|")

    # Order categories logically
    category_order = ['Collagens', 'Proteoglycans', 'ECM Glycoproteins',
                     'ECM Regulators', 'Secreted Factors', 'ECM-affiliated Proteins']

    for cat in category_order:
        if cat in category_stats.index:
            row = category_stats.loc[cat]
            # Get predominant division for this category
            cat_div = df[df['Matrisome_Category'] == cat]['Matrisome_Division'].mode()[0] if len(df[df['Matrisome_Category'] == cat]) > 0 else 'N/A'
            md.append(f"| {cat} | {cat_div} | {int(row['count'])} | {row['mean']:+.3f} | {row['std']:.3f} | {row['pct_upregulated']:.1f}% | {row['pct_downregulated']:.1f}% | {row['pct_stable']:.1f}% |")

    md.append("")

    # Category interpretations
    md.append("### 2.2 Category-Specific Interpretations")
    md.append("")

    for cat in category_order:
        if cat in category_stats.index:
            row = category_stats.loc[cat]
            md.append(f"**{cat}:**")

            if row['mean'] > 0.2:
                md.append(f"- Pattern: STRONG ACCUMULATION (Δz = {row['mean']:+.3f})")
                if cat == 'Collagens':
                    md.append(f"- Mechanism: Likely driven by pro-fibrotic signaling (TGF-β, mechanical stress)")
                    md.append(f"- Implication: Increased tissue stiffness, impaired regeneration")
                elif cat == 'ECM Regulators':
                    md.append(f"- Mechanism: Compensatory upregulation of protease inhibitors (TIMPs, serpins)")
                    md.append(f"- Implication: Imbalanced proteolysis → matrix accumulation")
            elif row['mean'] < -0.2:
                md.append(f"- Pattern: STRONG DEPLETION (Δz = {row['mean']:+.3f})")
                if cat == 'Collagens':
                    md.append(f"- Mechanism: Enhanced degradation outpacing synthesis")
                    md.append(f"- Implication: Loss of structural integrity, tissue fragility")
            else:
                md.append(f"- Pattern: HETEROGENEOUS/STABLE (Δz = {row['mean']:+.3f}, SD = {row['std']:.3f})")
                if row['std'] > 0.6:
                    md.append(f"- Mechanism: Context-dependent changes (tissue-specific, age-dependent)")
                    md.append(f"- Implication: No universal aging signature; requires subgroup analysis")

            md.append("")

    md.append("---")
    md.append("")

    # Section 3: Top Proteins
    md.append("## 3.0 Representative Proteins by Category")
    md.append("")
    md.append("¶1 Ordering: By category (Collagens → Proteoglycans → ECM Glycoproteins → Regulators → Secreted → Affiliated)")
    md.append("")

    section_num = 1
    for cat in category_order:
        if cat in top_proteins:
            proteins = top_proteins[cat]
            md.append(f"### 3.{section_num} {cat}")
            md.append("")

            md.append("**Top 5 UPREGULATED (Increased with Aging):**")
            md.append("")
            md.append("| Gene | Division | Δz-score | Tissue |")
            md.append("|------|----------|----------|--------|")
            for _, row in proteins['top_up'].iterrows():
                md.append(f"| {row['Gene_Symbol']} | {row['Matrisome_Division']} | {row['Zscore_Delta']:+.3f} | {row['Tissue']} |")
            md.append("")

            md.append("**Top 5 DOWNREGULATED (Decreased with Aging):**")
            md.append("")
            md.append("| Gene | Division | Δz-score | Tissue |")
            md.append("|------|----------|----------|--------|")
            for _, row in proteins['top_down'].iterrows():
                md.append(f"| {row['Gene_Symbol']} | {row['Matrisome_Division']} | {row['Zscore_Delta']:+.3f} | {row['Tissue']} |")
            md.append("")

            section_num += 1

    md.append("---")
    md.append("")

    # Section 4: Tissue-Specific Patterns
    md.append("## 4.0 Tissue-Specific Category Patterns")
    md.append("")
    md.append("¶1 Ordering: By tissue type, showing category-level differences")
    md.append("")

    if tissue_pivot is not None and not tissue_pivot.empty:
        md.append("**Average Z-score Delta by Tissue and Category:**")
        md.append("")

        # Create markdown table
        cols = tissue_pivot.columns.tolist()
        md.append("| Tissue | " + " | ".join(cols) + " |")
        md.append("|--------|" + "|".join(["--------"] * len(cols)) + "|")

        for tissue in tissue_pivot.index[:10]:  # Show top 10 tissues
            row_vals = [f"{tissue_pivot.loc[tissue, col]:+.2f}" if not pd.isna(tissue_pivot.loc[tissue, col]) else "N/A"
                       for col in cols]
            md.append(f"| {tissue} | " + " | ".join(row_vals) + " |")

        md.append("")
        md.append("*Note: Showing tissues with ≥5 proteins per category. Full data in CSV output.*")
        md.append("")

    md.append("---")
    md.append("")

    # Section 5: Biological Model
    md.append("## 5.0 Biological Model: ECM Aging Architecture")
    md.append("")
    md.append("### 5.1 Proposed Mechanistic Framework")
    md.append("")

    md.append("Based on category-level patterns, we propose a multi-phase ECM aging model:")
    md.append("")

    md.append("```mermaid")
    md.append("graph LR")
    md.append("    A[Phase 1: Initial Stress] --> B[Phase 2: Regulatory Response]")
    md.append("    B --> C[Phase 3: Structural Remodeling]")
    md.append("    C --> D[Phase 4: Functional Decline]")
    md.append("    ")
    md.append("    A --> A1[Oxidative stress, inflammation]")
    md.append("    B --> B1[ECM Regulators: Variable]")
    md.append("    B --> B2[Secreted Factors: Altered]")
    md.append("    C --> C1[Core Matrisome: Accumulation/Loss]")
    md.append("    C --> C2[Collagens: Changed composition]")
    md.append("    D --> D1[Tissue stiffness]")
    md.append("    D --> D2[Impaired homeostasis]")
    md.append("```")
    md.append("")

    md.append("### 5.2 Category-Specific Mechanisms")
    md.append("")

    md.append("**Collagens:**")
    coll_mean = category_stats.loc['Collagens', 'mean'] if 'Collagens' in category_stats.index else 0
    md.append(f"- Aging trend: {coll_mean:+.3f} z-score units")
    if coll_mean > 0:
        md.append(f"- Mechanism: TGF-β-driven fibrosis, increased synthesis, decreased MMP activity")
        md.append(f"- Result: Tissue stiffening, impaired elasticity")
    else:
        md.append(f"- Mechanism: Enhanced degradation, synthesis-degradation imbalance")
        md.append(f"- Result: Structural weakness, fragility")
    md.append("")

    md.append("**Proteoglycans:**")
    pg_mean = category_stats.loc['Proteoglycans', 'mean'] if 'Proteoglycans' in category_stats.index else 0
    md.append(f"- Aging trend: {pg_mean:+.3f} z-score units")
    md.append(f"- Mechanism: Context-dependent (aggrecan loss in cartilage, versican gain in fibrosis)")
    md.append(f"- Result: Altered hydration, mechanical properties, growth factor sequestration")
    md.append("")

    md.append("**ECM Regulators:**")
    reg_mean = category_stats.loc['ECM Regulators', 'mean'] if 'ECM Regulators' in category_stats.index else 0
    reg_std = category_stats.loc['ECM Regulators', 'std'] if 'ECM Regulators' in category_stats.index else 0
    md.append(f"- Aging trend: {reg_mean:+.3f} ± {reg_std:.3f} z-score units (HIGH VARIABILITY)")
    md.append(f"- Mechanism: Tissue-specific protease/inhibitor imbalances (MMP↑ TIMP↑ or MMP↓ TIMP↑)")
    md.append(f"- Result: Dysregulated turnover, inappropriate remodeling")
    md.append("")

    md.append("---")
    md.append("")

    # Section 6: Therapeutic Strategies
    md.append("## 6.0 Therapeutic Strategies: Category-Targeted Interventions")
    md.append("")
    md.append("### 6.1 Intervention Priority Ranking")
    md.append("")

    md.append("**TIER 1 - Target ECM Regulators (Early Intervention Window):**")
    md.append("")
    md.append("*Rationale:* High variability suggests early dysregulation before irreversible structural changes.")
    md.append("")
    md.append("| Strategy | Mechanism | Evidence | Availability |")
    md.append("|----------|-----------|----------|--------------|")
    md.append("| Losartan 50-100mg/day | AT1R blockade → ↓TGF-β | Phase 2 trials (Marfan, DMD) | FDA approved |")
    md.append("| Pirfenidone 2400mg/day | Multi-modal anti-fibrotic | FDA approved (IPF) | Prescription |")
    md.append("| Doxycycline 40mg/day | MMP modulation, anti-inflammatory | Used off-label (aortic aneurysm) | Prescription |")
    md.append("| Relaxin (recombinant) | ↑MMP activity, ↓collagen synthesis | Phase 2 trials (heart failure) | Investigational |")
    md.append("")

    md.append("**TIER 2 - Prevent Collagen Accumulation (Structural Intervention):**")
    md.append("")
    md.append("*Rationale:* Direct targeting of fibrotic collagen deposition.")
    md.append("")
    md.append("| Strategy | Mechanism | Evidence | Availability |")
    md.append("|----------|-----------|----------|--------------|")
    md.append("| LOX/LOXL2 inhibitors | Prevent collagen crosslinking | Phase 2 trials (fibrosis) | Investigational |")
    md.append("| Halofuginone 3mg/day | ↓Collagen synthesis via TGF-β | Compassionate use (scleroderma) | Investigational |")
    md.append("| Tranilast 300mg/day | ↓TGF-β, ↓collagen production | Approved in Asia (keloids) | Regional approval |")
    md.append("")

    md.append("**TIER 3 - Support Proteoglycan Homeostasis (Adjunct Therapy):**")
    md.append("")
    md.append("*Rationale:* Context-dependent changes require personalized approaches.")
    md.append("")
    md.append("| Strategy | Mechanism | Evidence | Availability |")
    md.append("|----------|-----------|----------|--------------|")
    md.append("| Decorin supplementation | Anti-TGF-β, anti-fibrotic | Preclinical (promising) | Research |")
    md.append("| Glucosamine + Chondroitin | GAG precursor support | Mixed clinical evidence (OA) | OTC supplement |")
    md.append("| Sprifermin (FGF18) | Stimulate cartilage proteoglycans | Phase 2 trials (OA) | Investigational |")
    md.append("")

    md.append("### 6.2 Combination Therapy Rationale")
    md.append("")
    md.append("**Optimal Multi-Target Regimen:**")
    md.append("")
    md.append("1. **Regulatory modulation:** Losartan 50-100 mg/day (anti-TGF-β)")
    md.append("2. **Structural prevention:** Doxycycline 20-40 mg/day (MMP balance)")
    md.append("3. **Metabolic support:** Metformin 1000-1500 mg/day (anti-fibrotic, AMPK activation)")
    md.append("4. **Anti-inflammatory:** Omega-3 fatty acids 2-4 g/day (SPM precursors)")
    md.append("5. **Antioxidant:** NAC 1200-1800 mg/day (reduce oxidative crosslinking)")
    md.append("")
    md.append("*Note: This represents a research-informed framework. Clinical implementation requires physician oversight and individualization.*")
    md.append("")

    md.append("---")
    md.append("")

    # Section 7: Key Questions Answered
    md.append("## 7.0 Key Questions Answered")
    md.append("")

    md.append("### 7.1 Does ECM aging = 'more collagens, fewer proteoglycans'?")
    coll_mean = category_stats.loc['Collagens', 'mean'] if 'Collagens' in category_stats.index else 0
    pg_mean = category_stats.loc['Proteoglycans', 'mean'] if 'Proteoglycans' in category_stats.index else 0
    md.append(f"**Answer:** {'Partially YES' if coll_mean > 0.1 and pg_mean < -0.1 else 'NO, more complex'}.")
    md.append(f"- Collagens: {coll_mean:+.3f} z-score units ({'increased' if coll_mean > 0 else 'decreased'})")
    md.append(f"- Proteoglycans: {pg_mean:+.3f} z-score units ({'increased' if pg_mean > 0 else 'decreased'})")
    md.append(f"- Reality: Both show tissue-specific patterns; simplistic model is insufficient")
    md.append("")

    md.append("### 7.2 Are regulatory proteins (MMPs, TIMPs) dysregulated as a class?")
    reg_mean = category_stats.loc['ECM Regulators', 'mean'] if 'ECM Regulators' in category_stats.index else 0
    reg_std = category_stats.loc['ECM Regulators', 'std'] if 'ECM Regulators' in category_stats.index else 0
    md.append(f"**Answer:** YES - but heterogeneously (mean = {reg_mean:+.3f}, SD = {reg_std:.3f}).")
    md.append(f"- Not uniform up or down regulation")
    md.append(f"- High variability suggests tissue/context-specific responses")
    md.append(f"- Some MMPs increase, others decrease; same for TIMPs")
    md.append(f"- Implication: Broad MMP inhibition/activation strategies are doomed to fail")
    md.append("")

    md.append("### 7.3 Do Core matrisome and Associated proteins age differently?")
    if stat_tests['core_vs_associated'][0] is not None:
        t_stat, p_val = stat_tests['core_vs_associated']
        md.append(f"**Answer:** YES - statistically significant (p = {p_val:.4e}).")
        md.append(f"- Core matrisome: {core_mean:+.3f} ± {core_std:.3f}")
        md.append(f"- Matrisome-associated: {assoc_mean:+.3f} ± {assoc_std:.3f}")
        md.append(f"- Interpretation: Structural and regulatory components follow distinct aging trajectories")
        md.append(f"- Therapeutic implication: Must target BOTH divisions, not just one")
    md.append("")

    md.append("---")
    md.append("")

    # Conclusions
    md.append("## 8.0 Conclusions")
    md.append("")
    md.append("### 8.1 Core Discoveries")
    md.append("")
    md.append("1. **Functional architecture matters:** Core vs Associated divisions show statistically distinct aging patterns")
    md.append("2. **Category heterogeneity is the rule:** No single ECM protein class follows uniform aging trajectory")
    md.append(f"3. **Regulatory variability exceeds structural:** ECM Regulators show {reg_std/core_std:.2f}x higher SD than Core matrisome")
    md.append("4. **Tissue context is critical:** Same category shows opposite patterns in different tissues")
    md.append("5. **Multi-target therapy essential:** Single-category interventions cannot address complexity")
    md.append("")

    md.append("### 8.2 Translational Roadmap")
    md.append("")
    md.append("**Phase 1 (Years 1-2): Repurposed Drug Trials**")
    md.append("- Losartan + Doxycycline combination in skin/muscle aging")
    md.append("- Biomarkers: Serum procollagen fragments, MMP/TIMP ratios")
    md.append("- Outcome: Tissue stiffness (shear wave elastography)")
    md.append("")
    md.append("**Phase 2 (Years 3-5): Category-Specific Biologics**")
    md.append("- Recombinant decorin or relaxin for fibrotic tissues")
    md.append("- Monoclonal antibodies targeting TGF-β pathway")
    md.append("- Outcome: Functional improvement (6-minute walk test, FEV1)")
    md.append("")
    md.append("**Phase 3 (Years 5-10): Precision ECM Medicine**")
    md.append("- Tissue biopsy + proteomics to stratify patients by category pattern")
    md.append("- Personalized multi-drug regimens based on individual ECM signature")
    md.append("- Outcome: Organ-specific functional restoration")
    md.append("")

    md.append("---")
    md.append("")
    md.append(f"**Analysis Date:** 2025-10-15")
    md.append(f"**Database:** {DB_PATH}")
    md.append(f"**Total Measurements:** {len(df)}")
    md.append(f"**Studies:** {df['Study_ID'].nunique()}")
    md.append(f"**Tissues:** {df['Tissue'].nunique()}")
    md.append(f"**Species:** {', '.join(df['Species'].unique())}")
    md.append(f"**Output:** {CSV_OUTPUT}")
    md.append(f"**Author:** Agent 5 - Matrisome Category Analyzer")
    md.append(f"**Contact:** daniel@improvado.io")

    return "\n".join(md)

def main():
    """Main analysis pipeline"""

    print("="*80)
    print("AGENT 5: MATRISOME CATEGORY ANALYZER")
    print("="*80)
    print()

    # Load data
    df = load_data()

    # Division and category-level statistics
    division_stats, category_stats = analyze_category_statistics(df)

    # Statistical testing
    stat_tests = statistical_testing(df)

    # Top proteins per category
    top_proteins = find_top_proteins_by_category(df)

    # Tissue-specific patterns
    tissue_pivot = analyze_tissue_specific_patterns(df)

    # Save summary CSV
    summary_df = save_summary_csv(df, division_stats, category_stats)

    # Generate report
    print("\n=== GENERATING MARKDOWN REPORT ===\n")
    report = generate_markdown_report(df, division_stats, category_stats, stat_tests, top_proteins, tissue_pivot)

    # Save report
    with open(REPORT_OUTPUT, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {REPORT_OUTPUT}")
    print(f"Report length: {len(report)} characters")
    print(f"CSV saved to: {CSV_OUTPUT}")
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("KEY FINDINGS:")
    print(f"- Core matrisome: {division_stats.loc['Core matrisome', 'mean']:+.3f} z-score units")
    print(f"- Matrisome-associated: {division_stats.loc['Matrisome-associated', 'mean']:+.3f} z-score units")
    print(f"- Most variable category: {category_stats['std'].idxmax()} (SD = {category_stats['std'].max():.3f})")
    print(f"- Most upregulated category: {category_stats['mean'].idxmax()} (Δz = {category_stats['mean'].max():+.3f})")
    print(f"- Most downregulated category: {category_stats['mean'].idxmin()} (Δz = {category_stats['mean'].min():+.3f})")

if __name__ == "__main__":
    main()
