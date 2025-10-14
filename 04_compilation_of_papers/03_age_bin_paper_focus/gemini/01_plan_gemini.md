# Gemini Agent Plan for Age Bin Normalization

## 1. Workspace Setup

- Create the necessary directories:
  - `03_age_bin_paper_focus/gemini/`
  - `03_age_bin_paper_focus/gemini/paper_analyses_updated/`

## 2. LFQ Study Identification

- Read all 11 paper analysis files located in `knowledge_base/01_paper_analysis/`.
- Identify the 6 LFQ-compatible studies based on the provided criteria.
- Exclude the 5 non-LFQ studies.

## 3. Age Bin and Column Mapping Analysis (6 LFQ Studies)

For each of the 6 LFQ studies, create a detailed analysis file: `03_age_bin_paper_focus/gemini/{StudyName}_age_bin_analysis.md`.

Each analysis will contain:
- **Method Verification:** Confirm the study uses an LFQ-compatible method.
- **Age Group Analysis:** Document the current age groups.
- **Species-Specific Cutoffs:** Apply the correct age cutoffs for the species.
- **Age Bin Mapping:** Define the mapping to "young" and "old" bins, excluding intermediate groups.
- **Column Mapping Verification:** Create a 13-column schema mapping table, identifying the source for each column and noting any gaps.

## 4. Update Paper Analyses

- Copy all 11 paper analysis files from `knowledge_base/01_paper_analysis/` into the `03_age_bin_paper_focus/gemini/paper_analyses_updated/` directory.
- For each of the 11 copied files, add a new "Section 6" that documents the age bin normalization strategy.
  - For the 6 LFQ studies, this section will detail the young/old mapping.
  - For the 5 non-LFQ studies, this section will state that the study is excluded from the current analysis.

## 5. Create Summary and Self-Evaluation

- **Cross-Study Summary:** Generate `03_age_bin_paper_focus/gemini/00_cross_study_age_bin_summary.md` to summarize the results for all 11 studies.
- **Self-Evaluation:** Complete the task by creating the `90_results_gemini.md` file, which will include a self-evaluation based on the 13 success criteria.
