# Plan - Codex Agent

## Objective
Complete Tier 0 "knowledge-first" analysis for the 11 ECM proteomic datasets before any parsing code is written.

## Steps
1. **Dataset Reconnaissance**
   - Enumerate all raw files under `data_raw/` per study.
   - Capture file metadata (format, sheet names, row/column counts).
   - Draft `knowledge_base/00_dataset_inventory.md` with consolidated table.

2. **Per-Study Paper & Data Analysis**
   - For each of the 11 studies, extract study metadata (title, PMID, tissues, species, age groups).
   - Review associated raw files to confirm structure and relevant columns.
   - Populate `knowledge_base/01_paper_analysis/[Study]_analysis.md` using the provided template.

3. **Cross-Study Strategy Documents**
   - Derive consistent column mappings across datasets; document in `knowledge_base/02_column_mapping_strategy.md`.
   - Determine abundance normalization assumptions; document in `knowledge_base/03_normalization_strategy.md`.

4. **Implementation Roadmap**
   - After consolidating findings from Steps 1-3, outline the parsing approach, modular design, configs, and validation hooks in `knowledge_base/04_implementation_plan.md`.

5. **Progress & Evidence Logging**
   - Record key observations and outstanding questions for traceability.
   - Prepare final analysis summary for `codex_cli/90_results_codex.md`.

## Notes
- No parsing code until Steps 1-3 are fully complete.
- Prioritize accuracy of age/abundance reasoning and traceable citations.
