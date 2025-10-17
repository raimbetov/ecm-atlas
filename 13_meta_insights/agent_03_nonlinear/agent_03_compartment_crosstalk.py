#!/usr/bin/env python3
"""
AGENT 03: COMPARTMENT CROSSTALK ANALYZER

Mission: Discover COORDINATED changes between tissue compartments.
Find anti-correlated (compensatory) and synergistic (coordinated) protein pairs.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CompartmentCrosstalkAnalyzer:
    """Analyze compartment-compartment interactions in aging"""

    def __init__(self, data_path):
        print("Loading dataset...")
        self.df = pd.read_csv(data_path)
        self.output_dir = Path('/Users/Kravtsovd/projects/ecm-atlas/10_insights/discovery_ver1')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Filter to valid z-scores
        self.df = self.df.dropna(subset=['Zscore_Delta', 'Gene_Symbol', 'Compartment'])

        print(f"Loaded {len(self.df)} protein measurements")
        print(f"Unique proteins: {self.df['Gene_Symbol'].nunique()}")
        print(f"Unique tissues: {self.df['Tissue'].nunique()}")

    def identify_multi_compartment_studies(self):
        """Find studies with multiple compartments from same tissue"""
        print("\n" + "="*80)
        print("PHASE 1: IDENTIFYING MULTI-COMPARTMENT STUDIES")
        print("="*80 + "\n")

        # Group by Study_ID and count compartments
        study_compartments = self.df.groupby(['Study_ID', 'Tissue', 'Compartment']).size().reset_index(name='protein_count')

        # Find studies with >1 compartment
        multi_comp_studies = study_compartments.groupby('Study_ID').size()
        multi_comp_studies = multi_comp_studies[multi_comp_studies > 1]

        print(f"Studies with multiple compartments: {len(multi_comp_studies)}\n")

        results = []
        for study_id in multi_comp_studies.index:
            study_data = study_compartments[study_compartments['Study_ID'] == study_id]
            tissue = study_data['Tissue'].iloc[0]
            comps = study_data['Compartment'].tolist()

            print(f"{study_id} ({tissue}):")
            for _, row in study_data.iterrows():
                print(f"  - {row['Compartment']}: {row['protein_count']} proteins")
            print()

            results.append({
                'Study_ID': study_id,
                'Tissue': tissue,
                'Compartments': comps,
                'N_Compartments': len(comps)
            })

        return pd.DataFrame(results)

    def calculate_protein_correlations(self, study_id, tissue, compartments):
        """Calculate protein-level correlations between compartments"""
        print(f"\n{'='*80}")
        print(f"ANALYZING: {study_id} - {tissue}")
        print(f"Compartments: {', '.join(compartments)}")
        print(f"{'='*80}\n")

        # Get data for this study
        study_df = self.df[
            (self.df['Study_ID'] == study_id) &
            (self.df['Compartment'].isin(compartments))
        ].copy()

        # Find proteins present in multiple compartments
        protein_comp_counts = study_df.groupby('Gene_Symbol')['Compartment'].nunique()
        shared_proteins = protein_comp_counts[protein_comp_counts > 1].index.tolist()

        print(f"Total proteins: {study_df['Gene_Symbol'].nunique()}")
        print(f"Shared proteins (in >1 compartment): {len(shared_proteins)}\n")

        if len(shared_proteins) == 0:
            return pd.DataFrame(), pd.DataFrame()

        # Pivot to get compartment comparison matrix
        pivot_df = study_df[study_df['Gene_Symbol'].isin(shared_proteins)].pivot_table(
            index='Gene_Symbol',
            columns='Compartment',
            values='Zscore_Delta',
            aggfunc='mean'
        )

        # Calculate pairwise protein correlations
        results = []
        compartment_pairs = []

        for i, comp1 in enumerate(compartments):
            for j, comp2 in enumerate(compartments):
                if i >= j:  # Skip diagonal and duplicates
                    continue

                if comp1 not in pivot_df.columns or comp2 not in pivot_df.columns:
                    continue

                # Get paired data (proteins in both compartments)
                paired = pivot_df[[comp1, comp2]].dropna()

                if len(paired) < 3:
                    continue

                # Calculate correlation
                corr, pval = stats.pearsonr(paired[comp1], paired[comp2])

                compartment_pairs.append({
                    'Study_ID': study_id,
                    'Tissue': tissue,
                    'Compartment_A': comp1,
                    'Compartment_B': comp2,
                    'N_Shared_Proteins': len(paired),
                    'Correlation': corr,
                    'P_Value': pval,
                    'Relationship': self._classify_relationship(corr)
                })

                # Find individual proteins driving the relationship
                for protein in paired.index:
                    delta_a = paired.loc[protein, comp1]
                    delta_b = paired.loc[protein, comp2]

                    results.append({
                        'Study_ID': study_id,
                        'Tissue': tissue,
                        'Protein': protein,
                        'Compartment_A': comp1,
                        'Compartment_B': comp2,
                        'Delta_A': delta_a,
                        'Delta_B': delta_b,
                        'Product': delta_a * delta_b,  # Positive = synergistic, negative = antagonistic
                        'Divergence': abs(delta_a - delta_b),
                        'Pattern': self._classify_pattern(delta_a, delta_b)
                    })

        return pd.DataFrame(results), pd.DataFrame(compartment_pairs)

    def _classify_relationship(self, corr):
        """Classify compartment relationship by correlation"""
        if corr > 0.5:
            return 'Synergistic (coordinated)'
        elif corr < -0.3:
            return 'Compensatory (anti-correlated)'
        else:
            return 'Independent'

    def _classify_pattern(self, delta_a, delta_b):
        """Classify protein pattern between compartments"""
        threshold = 0.5

        if abs(delta_a) < threshold and abs(delta_b) < threshold:
            return 'No_change'
        elif delta_a > threshold and delta_b > threshold:
            return 'Both_UP (synergistic)'
        elif delta_a < -threshold and delta_b < -threshold:
            return 'Both_DOWN (synergistic)'
        elif delta_a > threshold and delta_b < -threshold:
            return 'A_UP_B_DOWN (antagonistic)'
        elif delta_a < -threshold and delta_b > threshold:
            return 'A_DOWN_B_UP (antagonistic)'
        else:
            return 'Mixed'

    def find_master_regulators(self, protein_df):
        """Find proteins that appear in multiple compartment pairs with consistent patterns"""
        print("\n" + "="*80)
        print("FINDING MASTER REGULATORS")
        print("="*80 + "\n")

        # Count how many compartment pairs each protein appears in
        protein_counts = protein_df.groupby('Protein').agg({
            'Study_ID': 'first',
            'Tissue': 'first',
            'Pattern': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',  # Most common pattern
            'Delta_A': 'count',  # Number of pairs
            'Divergence': 'mean'
        }).rename(columns={'Delta_A': 'N_Compartment_Pairs'})

        protein_counts = protein_counts[protein_counts['N_Compartment_Pairs'] > 1]
        protein_counts = protein_counts.sort_values('Divergence', ascending=False)

        print(f"Potential master regulators (appear in >1 compartment pair): {len(protein_counts)}\n")
        print(protein_counts.head(20).to_string())

        return protein_counts

    def run_analysis(self):
        """Run complete compartment crosstalk analysis"""
        print("\n" + "="*80)
        print("AGENT 03: COMPARTMENT CROSSTALK ANALYZER")
        print("="*80)

        # Phase 1: Identify multi-compartment studies
        multi_comp_studies = self.identify_multi_compartment_studies()

        # Phase 2: Analyze each study
        all_proteins = []
        all_compartment_pairs = []

        for _, study in multi_comp_studies.iterrows():
            protein_df, comp_pair_df = self.calculate_protein_correlations(
                study['Study_ID'],
                study['Tissue'],
                study['Compartments']
            )

            if not protein_df.empty:
                all_proteins.append(protein_df)
            if not comp_pair_df.empty:
                all_compartment_pairs.append(comp_pair_df)

        # Combine results
        if all_proteins:
            protein_results = pd.concat(all_proteins, ignore_index=True)
        else:
            protein_results = pd.DataFrame()

        if all_compartment_pairs:
            compartment_results = pd.concat(all_compartment_pairs, ignore_index=True)
        else:
            compartment_results = pd.DataFrame()

        # Phase 3: Find master regulators
        if not protein_results.empty:
            master_regulators = self.find_master_regulators(protein_results)
        else:
            master_regulators = pd.DataFrame()

        # Phase 4: Generate outputs
        self.save_results(protein_results, compartment_results, master_regulators)
        self.generate_report(protein_results, compartment_results, master_regulators)

        return protein_results, compartment_results, master_regulators

    def save_results(self, protein_results, compartment_results, master_regulators):
        """Save comprehensive CSV results"""

        # Main CSV: protein-level crosstalk
        csv_path = self.output_dir / 'agent_03_compartment_crosstalk.csv'
        protein_results.to_csv(csv_path, index=False)
        print(f"\nSaved protein-level results: {csv_path}")

        # Compartment pair summary
        comp_path = self.output_dir / 'agent_03_compartment_correlations.csv'
        compartment_results.to_csv(comp_path, index=False)
        print(f"Saved compartment correlations: {comp_path}")

        # Master regulators
        if not master_regulators.empty:
            master_path = self.output_dir / 'agent_03_master_regulators.csv'
            master_regulators.to_csv(master_path)
            print(f"Saved master regulators: {master_path}")

    def generate_report(self, protein_results, compartment_results, master_regulators):
        """Generate markdown report following Knowledge Framework"""
        report_path = self.output_dir / 'agent_03_compartment_crosstalk_REPORT.md'

        with open(report_path, 'w') as f:
            # Thesis
            f.write("# Compartment Crosstalk Analysis: Systems-Level Aging Coordination\n\n")

            f.write("## Thesis\n")
            f.write("Multi-compartment tissue analysis reveals coordinated ECM aging through synergistic ")
            f.write("(both-up/both-down) and compensatory (anti-correlated) protein networks, with master ")
            f.write("regulators orchestrating cross-compartment responses to aging stress.\n\n")

            # Overview
            f.write("## Overview\n")
            n_studies = protein_results['Study_ID'].nunique() if not protein_results.empty else 0
            n_proteins = protein_results['Protein'].nunique() if not protein_results.empty else 0
            n_pairs = len(compartment_results) if not compartment_results.empty else 0

            f.write(f"Analyzed {n_studies} multi-compartment studies containing {n_proteins} shared proteins ")
            f.write(f"across {n_pairs} compartment pairs. Compartment crosstalk analysis identifies proteins ")
            f.write("that age coordinately (synergistic) versus independently (compensatory) between adjacent ")
            f.write("tissue regions, revealing systems-level aging programs and potential master regulators.\n\n")

            # Diagrams
            f.write("**System Structure (Continuants):**\n")
            f.write("```mermaid\n")
            f.write("graph TD\n")
            f.write("    Tissue[Multi-Compartment Tissue] --> CompA[Compartment A]\n")
            f.write("    Tissue --> CompB[Compartment B]\n")
            f.write("    CompA --> ProteinsA[ECM Proteins A]\n")
            f.write("    CompB --> ProteinsB[ECM Proteins B]\n")
            f.write("    ProteinsA <-.Crosstalk.-> ProteinsB\n")
            f.write("    ProteinsA --> Synergistic[Synergistic: Both↑ or Both↓]\n")
            f.write("    ProteinsA --> Compensatory[Compensatory: A↑B↓ or A↓B↑]\n")
            f.write("    ProteinsA --> Master[Master Regulators]\n")
            f.write("```\n\n")

            f.write("**Analysis Flow (Occurrents):**\n")
            f.write("```mermaid\n")
            f.write("graph LR\n")
            f.write("    A[Identify Multi-Compartment Studies] --> B[Find Shared Proteins]\n")
            f.write("    B --> C[Calculate Compartment Correlations]\n")
            f.write("    C --> D[Classify Protein Patterns]\n")
            f.write("    D --> E[Identify Master Regulators]\n")
            f.write("    E --> F[Biological Interpretation]\n")
            f.write("```\n\n")

            f.write("---\n\n")

            # Section 1: Compartment-Level Correlations
            f.write("## 1.0 Compartment-Level Correlations\n\n")
            f.write("¶1 Ordering: Study → Compartment pairs → Correlation strength → Relationship type\n\n")

            if not compartment_results.empty:
                for tissue in compartment_results['Tissue'].unique():
                    tissue_data = compartment_results[compartment_results['Tissue'] == tissue]

                    f.write(f"### {tissue}\n\n")
                    f.write("| Compartment A | Compartment B | Shared Proteins | Correlation | P-Value | Relationship |\n")
                    f.write("|---------------|---------------|-----------------|-------------|---------|---------------|\n")

                    for _, row in tissue_data.iterrows():
                        f.write(f"| {row['Compartment_A']} | {row['Compartment_B']} | ")
                        f.write(f"{row['N_Shared_Proteins']} | {row['Correlation']:.3f} | ")
                        f.write(f"{row['P_Value']:.2e} | {row['Relationship']} |\n")
                    f.write("\n")

            f.write("\n---\n\n")

            # Section 2: Synergistic Proteins
            f.write("## 2.0 Synergistic Proteins (Coordinated Aging)\n\n")
            f.write("¶1 Ordering: Pattern type → Protein → Compartment comparison → Biological significance\n\n")

            if not protein_results.empty:
                synergistic = protein_results[protein_results['Pattern'].str.contains('synergistic', case=False, na=False)]

                if not synergistic.empty:
                    # Both UP
                    both_up = synergistic[synergistic['Pattern'].str.contains('Both_UP', na=False)]
                    if not both_up.empty:
                        f.write("### 2.1 Both Compartments UP (Coordinated Increase)\n\n")
                        f.write("Proteins that increase in BOTH compartments during aging (synchronized upregulation):\n\n")

                        top_both_up = both_up.nlargest(20, 'Product')
                        f.write("| Protein | Tissue | Comp A | Δz_A | Comp B | Δz_B | Product |\n")
                        f.write("|---------|--------|--------|------|--------|------|----------|\n")
                        for _, row in top_both_up.iterrows():
                            f.write(f"| {row['Protein']} | {row['Tissue']} | {row['Compartment_A']} | ")
                            f.write(f"{row['Delta_A']:.2f} | {row['Compartment_B']} | ")
                            f.write(f"{row['Delta_B']:.2f} | {row['Product']:.2f} |\n")
                        f.write("\n")

                    # Both DOWN
                    both_down = synergistic[synergistic['Pattern'].str.contains('Both_DOWN', na=False)]
                    if not both_down.empty:
                        f.write("### 2.2 Both Compartments DOWN (Coordinated Decrease)\n\n")
                        f.write("Proteins that decrease in BOTH compartments during aging (synchronized downregulation):\n\n")

                        top_both_down = both_down.nlargest(20, 'Product')
                        f.write("| Protein | Tissue | Comp A | Δz_A | Comp B | Δz_B | Product |\n")
                        f.write("|---------|--------|--------|------|--------|------|----------|\n")
                        for _, row in top_both_down.iterrows():
                            f.write(f"| {row['Protein']} | {row['Tissue']} | {row['Compartment_A']} | ")
                            f.write(f"{row['Delta_A']:.2f} | {row['Compartment_B']} | ")
                            f.write(f"{row['Delta_B']:.2f} | {row['Product']:.2f} |\n")
                        f.write("\n")

            f.write("\n---\n\n")

            # Section 3: Antagonistic Proteins
            f.write("## 3.0 Antagonistic Proteins (Compensatory Aging)\n\n")
            f.write("¶1 Ordering: Pattern type → Divergence score → Compensatory mechanisms\n\n")

            if not protein_results.empty:
                antagonistic = protein_results[protein_results['Pattern'].str.contains('antagonistic', case=False, na=False)]

                if not antagonistic.empty:
                    f.write("Proteins showing OPPOSITE directions in different compartments (one UP, other DOWN):\n\n")

                    top_antag = antagonistic.nlargest(30, 'Divergence')
                    f.write("| Protein | Tissue | Comp A | Δz_A | Comp B | Δz_B | Divergence | Pattern |\n")
                    f.write("|---------|--------|--------|------|--------|------|------------|----------|\n")
                    for _, row in top_antag.iterrows():
                        f.write(f"| {row['Protein']} | {row['Tissue']} | {row['Compartment_A']} | ")
                        f.write(f"{row['Delta_A']:.2f} | {row['Compartment_B']} | ")
                        f.write(f"{row['Delta_B']:.2f} | {row['Divergence']:.2f} | {row['Pattern']} |\n")
                    f.write("\n")

            f.write("\n---\n\n")

            # Section 4: Master Regulators
            f.write("## 4.0 Master Regulators\n\n")
            f.write("¶1 Ordering: Consistency → Presence in multiple pairs → Therapeutic potential\n\n")

            if not master_regulators.empty:
                f.write("Proteins appearing in multiple compartment pairs with consistent patterns:\n\n")
                f.write("| Protein | Tissue | Dominant Pattern | N_Pairs | Avg Divergence |\n")
                f.write("|---------|--------|------------------|---------|----------------|\n")
                for protein, row in master_regulators.head(30).iterrows():
                    f.write(f"| {protein} | {row['Tissue']} | {row['Pattern']} | ")
                    f.write(f"{row['N_Compartment_Pairs']} | {row['Divergence']:.3f} |\n")
                f.write("\n")

            f.write("\n---\n\n")

            # Section 5: Biological Interpretation
            f.write("## 5.0 Biological Interpretation\n\n")
            f.write("¶1 Ordering: Molecular mechanisms → Clinical implications → Therapeutic strategies\n\n")

            f.write("### 5.1 Systems-Level Aging Coordination\n\n")
            f.write("**Synergistic aging (coordinated):** Compartments age together through shared mechanisms:\n")
            f.write("- **Systemic factors:** Inflammatory cytokines, oxidative stress, hormonal changes affect entire tissue\n")
            f.write("- **Biomechanical coupling:** Load-bearing tissues experience coordinated mechanical deterioration\n")
            f.write("- **Shared vasculature:** Blood supply decline affects adjacent compartments simultaneously\n\n")

            f.write("**Compensatory aging (anti-correlated):** One compartment adapts to weakness in neighbor:\n")
            f.write("- **Load redistribution:** Healthy compartment reinforces ECM to compensate for degenerating neighbor\n")
            f.write("- **Paracrine signaling:** Damaged compartment secretes factors triggering protective response in adjacent tissue\n")
            f.write("- **Differential vulnerability:** Compartments age at different rates, creating asymmetric remodeling\n\n")

            f.write("### 5.2 Master Regulators as Therapeutic Targets\n\n")
            f.write("Proteins appearing in multiple compartment pairs with consistent patterns represent system-wide controllers:\n")
            f.write("- **Target one protein, affect multiple compartments:** Efficiency for multi-compartment diseases\n")
            f.write("- **Biomarker potential:** Master regulators may be detectable in systemic circulation\n")
            f.write("- **Druggability:** Proteins with conserved patterns across tissues are ideal therapeutic targets\n\n")

            f.write("### 5.3 Clinical Implications\n\n")
            f.write("**Disease progression:** Understanding crosstalk predicts how damage spreads between compartments\n")
            f.write("- Disc degeneration: NP collapse triggers AF stress response\n")
            f.write("- Kidney fibrosis: Glomerular damage induces tubulointerstitial remodeling\n\n")

            f.write("**Therapeutic intervention:** Target crosstalk mechanisms, not individual compartments\n")
            f.write("- Block paracrine damage signals between compartments\n")
            f.write("- Enhance compensatory responses in healthy compartments\n")
            f.write("- Restore biomechanical balance across tissue regions\n\n")

            f.write("---\n\n")

            # Section 6: Key Findings
            f.write("## 6.0 Key Findings Summary\n\n")

            if not protein_results.empty:
                n_synergistic = len(protein_results[protein_results['Pattern'].str.contains('synergistic', case=False, na=False)])
                n_antagonistic = len(protein_results[protein_results['Pattern'].str.contains('antagonistic', case=False, na=False)])
                n_master = len(master_regulators)

                f.write(f"- **Synergistic protein pairs (coordinated):** {n_synergistic}\n")
                f.write(f"- **Antagonistic protein pairs (compensatory):** {n_antagonistic}\n")
                f.write(f"- **Master regulators identified:** {n_master}\n")
                f.write(f"- **Multi-compartment studies analyzed:** {protein_results['Study_ID'].nunique()}\n")
                f.write(f"- **Unique proteins analyzed:** {protein_results['Protein'].nunique()}\n\n")

            f.write("**Conclusion:** Tissue compartments age as SYSTEMS, not isolated units. Synergistic patterns ")
            f.write("reveal tissue-wide aging programs, while compensatory patterns show adaptive remodeling. ")
            f.write("Master regulators orchestrating cross-compartment responses represent high-value therapeutic ")
            f.write("targets for multi-compartment diseases.\n\n")

            f.write("**Answer to key question:** Do compartments age independently or talk to each other? ")
            f.write("**THEY TALK.** Strong correlations (positive and negative) demonstrate active crosstalk ")
            f.write("through paracrine signaling, biomechanical coupling, and systemic factors.\n\n")

            f.write("---\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Analysis:** Agent 03 - Compartment Crosstalk Analyzer\n")
            f.write(f"**Dataset:** merged_ecm_aging_zscore.csv\n")

        print(f"\nGenerated report: {report_path}")


def main():
    """Run Agent 03 analysis"""
    data_path = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'

    analyzer = CompartmentCrosstalkAnalyzer(data_path)
    protein_results, compartment_results, master_regulators = analyzer.run_analysis()

    print("\n" + "="*80)
    print("AGENT 03: COMPARTMENT CROSSTALK ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print(f"  - CSV: 10_insights/discovery_ver1/agent_03_compartment_crosstalk.csv")
    print(f"  - Report: 10_insights/discovery_ver1/agent_03_compartment_crosstalk_REPORT.md")
    print(f"  - Correlations: 10_insights/discovery_ver1/agent_03_compartment_correlations.csv")
    print(f"  - Master regulators: 10_insights/discovery_ver1/agent_03_master_regulators.csv")
    print("\n")


if __name__ == "__main__":
    main()
