#!/usr/bin/env Rscript
# ComBat Batch Effect Correction for ECM-Atlas
#
# Purpose: Apply empirical Bayes batch correction to remove study-specific
#          technical variation while preserving biological age effects
#
# Input: ../../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
# Output: combat_corrected.csv, combat_metadata.json, diagnostic plots
#
# Author: Exploratory Batch Correction Analysis
# Date: 2025-10-18

# =============================================================================
# 1. SETUP
# =============================================================================

cat("=== ComBat Batch Correction Pipeline ===\n\n")

# Load required packages
required_packages <- c("sva", "tidyverse", "jsonlite", "lme4")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing package: %s\n", pkg))
    if (pkg == "sva") {
      if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
      BiocManager::install("sva", update = FALSE)
    } else {
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
    library(pkg, character.only = TRUE)
  }
}

# Set working directory
setwd(dirname(sys.frame(1)$ofile))

# Create output directories
dir.create("../diagnostics", showWarnings = FALSE, recursive = TRUE)

cat("Libraries loaded successfully.\n\n")

# =============================================================================
# 2. LOAD AND PREPARE DATA
# =============================================================================

cat("Loading merged ECM dataset...\n")

# Load data
data_path <- "../../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

if (!file.exists(data_path)) {
  stop(sprintf("Data file not found: %s", data_path))
}

df <- read_csv(data_path, show_col_types = FALSE)

cat(sprintf("Loaded %d rows, %d columns\n", nrow(df), ncol(df)))
cat(sprintf("Unique proteins: %d\n", length(unique(df$Protein_ID))))
cat(sprintf("Unique studies: %d\n", length(unique(df$Study_ID))))
cat(sprintf("Age groups: %s\n\n", paste(unique(df$Age_Group), collapse = ", ")))

# =============================================================================
# 3. CREATE EXPRESSION MATRIX
# =============================================================================

cat("Creating expression matrix for ComBat...\n")

# ComBat requires: rows = features (proteins), columns = samples
# Create unique sample identifiers
df <- df %>%
  mutate(Sample_ID = paste(Study_ID, Tissue_Compartment, Age_Group,
                           row_number(), sep = "_"))

# Pivot to wide format (proteins × samples)
expr_matrix <- df %>%
  select(Protein_ID, Sample_ID, Abundance) %>%
  pivot_wider(names_from = Sample_ID, values_from = Abundance) %>%
  column_to_rownames("Protein_ID") %>%
  as.matrix()

cat(sprintf("Expression matrix: %d proteins × %d samples\n",
            nrow(expr_matrix), ncol(expr_matrix)))

# Handle missing values
na_count <- sum(is.na(expr_matrix))
na_percent <- (na_count / (nrow(expr_matrix) * ncol(expr_matrix))) * 100

cat(sprintf("Missing values: %d (%.1f%%)\n", na_count, na_percent))

# ComBat can't handle NAs - for now, use complete cases only
# Alternative: impute with protein-specific median
proteins_complete <- rowSums(is.na(expr_matrix)) == 0
samples_complete <- colSums(is.na(expr_matrix)) == 0

cat(sprintf("Proteins with complete data: %d / %d (%.1f%%)\n",
            sum(proteins_complete), length(proteins_complete),
            mean(proteins_complete) * 100))

cat(sprintf("Samples with complete data: %d / %d (%.1f%%)\n\n",
            sum(samples_complete), length(samples_complete),
            mean(samples_complete) * 100))

# For this exploratory analysis: use median imputation
cat("Applying median imputation for missing values...\n")
for (i in 1:nrow(expr_matrix)) {
  row_median <- median(expr_matrix[i, ], na.rm = TRUE)
  expr_matrix[i, is.na(expr_matrix[i, ])] <- row_median
}

cat("Imputation complete.\n\n")

# =============================================================================
# 4. CREATE METADATA
# =============================================================================

cat("Extracting sample metadata...\n")

# Extract metadata for each sample
metadata <- df %>%
  select(Sample_ID, Study_ID, Age_Group, Tissue_Compartment) %>%
  distinct() %>%
  arrange(Sample_ID)

# Ensure same order as expression matrix
metadata <- metadata[match(colnames(expr_matrix), metadata$Sample_ID), ]

# Verify alignment
stopifnot(all(metadata$Sample_ID == colnames(expr_matrix)))

cat(sprintf("Metadata extracted for %d samples\n", nrow(metadata)))
cat(sprintf("Batch variable (Study_ID): %d levels\n",
            length(unique(metadata$Study_ID))))
cat(sprintf("Age groups: %s\n",
            paste(unique(metadata$Age_Group), collapse = ", ")))
cat(sprintf("Tissues: %d compartments\n\n",
            length(unique(metadata$Tissue_Compartment))))

# =============================================================================
# 5. CREATE MODEL MATRIX (Covariates to Preserve)
# =============================================================================

cat("Creating model matrix for biological covariates...\n")

# Preserve age effect and tissue-specific differences
# Model: ~ Age_Group + Tissue_Compartment
mod <- model.matrix(~ Age_Group + Tissue_Compartment, data = metadata)

cat(sprintf("Model matrix: %d samples × %d coefficients\n",
            nrow(mod), ncol(mod)))
cat("Covariates preserved:\n")
cat(sprintf("  - Age_Group (2 levels)\n"))
cat(sprintf("  - Tissue_Compartment (%d levels)\n\n",
            length(unique(metadata$Tissue_Compartment))))

# =============================================================================
# 6. APPLY COMBAT
# =============================================================================

cat("=== RUNNING COMBAT BATCH CORRECTION ===\n")
cat("This may take several minutes...\n\n")

start_time <- Sys.time()

# Run ComBat
# par.prior = TRUE: Use empirical Bayes for batch effect estimation
# ref.batch = NULL: No reference batch (adjust all batches)
combat_corrected <- ComBat(
  dat = expr_matrix,
  batch = metadata$Study_ID,
  mod = mod,
  par.prior = TRUE,
  prior.plots = FALSE,
  ref.batch = NULL
)

end_time <- Sys.time()
runtime <- difftime(end_time, start_time, units = "secs")

cat(sprintf("\nComBat completed in %.1f seconds\n\n", as.numeric(runtime)))

# =============================================================================
# 7. BACK-TRANSFORM TO LONG FORMAT
# =============================================================================

cat("Converting corrected matrix back to long format...\n")

# Convert to data frame
combat_df <- as.data.frame(combat_corrected) %>%
  rownames_to_column("Protein_ID") %>%
  pivot_longer(
    cols = -Protein_ID,
    names_to = "Sample_ID",
    values_to = "Abundance_Corrected"
  )

# Merge with original metadata
df_corrected <- df %>%
  left_join(combat_df, by = c("Protein_ID", "Sample_ID")) %>%
  mutate(
    Abundance_Original = Abundance,
    Abundance = Abundance_Corrected
  ) %>%
  select(-Abundance_Corrected)

cat(sprintf("Corrected data: %d rows\n\n", nrow(df_corrected)))

# =============================================================================
# 8. RECALCULATE Z-SCORES ON CORRECTED DATA
# =============================================================================

cat("Recalculating z-scores on batch-corrected abundances...\n")

# Within-study z-score normalization (same as original pipeline)
df_corrected <- df_corrected %>%
  group_by(Study_ID, Tissue_Compartment) %>%
  mutate(
    Z_score_Original = Z_score,
    Z_score = scale(Abundance)[, 1]
  ) %>%
  ungroup()

cat("Z-scores recalculated.\n\n")

# =============================================================================
# 9. VALIDATION METRICS
# =============================================================================

cat("=== VALIDATION METRICS ===\n\n")

# 9.1 Calculate ICC before and after correction
cat("Calculating ICC (Intraclass Correlation)...\n")

# Function to calculate ICC
calculate_icc <- function(data, value_col, group_col) {
  # Fit random intercept model: value ~ 1 + (1 | group)
  formula_str <- sprintf("%s ~ 1 + (1 | %s)", value_col, group_col)
  model <- lmer(as.formula(formula_str), data = data, REML = TRUE)

  # Extract variance components
  var_components <- as.data.frame(VarCorr(model))
  sigma2_between <- var_components$vcov[var_components$grp == group_col]
  sigma2_within <- var_components$vcov[var_components$grp == "Residual"]

  # ICC = sigma2_between / (sigma2_between + sigma2_within)
  icc <- sigma2_between / (sigma2_between + sigma2_within)

  return(icc)
}

# Sample subset for ICC calculation (to reduce computation)
set.seed(42)
sample_proteins <- sample(unique(df$Protein_ID), 200)

icc_data_before <- df %>%
  filter(Protein_ID %in% sample_proteins) %>%
  select(Protein_ID, Study_ID, Z_score_Original) %>%
  rename(Z_score = Z_score_Original)

icc_data_after <- df_corrected %>%
  filter(Protein_ID %in% sample_proteins) %>%
  select(Protein_ID, Study_ID, Z_score)

tryCatch({
  icc_before <- calculate_icc(icc_data_before, "Z_score", "Study_ID")
  icc_after <- calculate_icc(icc_data_after, "Z_score", "Study_ID")

  cat(sprintf("ICC before correction: %.3f\n", icc_before))
  cat(sprintf("ICC after correction:  %.3f\n", icc_after))
  cat(sprintf("ICC improvement: %+.3f\n\n", icc_after - icc_before))
}, error = function(e) {
  cat("ICC calculation failed (likely due to data structure).\n")
  cat("Skipping ICC validation.\n\n")
  icc_before <- NA
  icc_after <- NA
})

# 9.2 Variance summary
cat("Variance statistics:\n")
var_before <- var(df$Z_score, na.rm = TRUE)
var_after <- var(df_corrected$Z_score, na.rm = TRUE)

cat(sprintf("Variance before: %.3f\n", var_before))
cat(sprintf("Variance after:  %.3f\n\n", var_after))

# 9.3 Effect size preservation (age effect correlation)
cat("Checking biological signal preservation...\n")

# Calculate mean age difference per protein (before and after)
age_effects_before <- df %>%
  group_by(Protein_ID) %>%
  summarize(
    Delta_Z_Before = mean(Z_score[Age_Group == "Old"], na.rm = TRUE) -
                     mean(Z_score[Age_Group == "Young"], na.rm = TRUE),
    .groups = "drop"
  )

age_effects_after <- df_corrected %>%
  group_by(Protein_ID) %>%
  summarize(
    Delta_Z_After = mean(Z_score[Age_Group == "Old"], na.rm = TRUE) -
                    mean(Z_score[Age_Group == "Young"], na.rm = TRUE),
    .groups = "drop"
  )

age_effects_comparison <- age_effects_before %>%
  inner_join(age_effects_after, by = "Protein_ID")

# Correlation of effect sizes
effect_corr <- cor(age_effects_comparison$Delta_Z_Before,
                   age_effects_comparison$Delta_Z_After,
                   use = "complete.obs")

cat(sprintf("Age effect correlation (before vs after): r = %.3f\n", effect_corr))

if (effect_corr > 0.7) {
  cat("✓ Biological signal well preserved\n\n")
} else if (effect_corr > 0.5) {
  cat("⚠ Biological signal partially preserved\n\n")
} else {
  cat("✗ WARNING: Biological signal may be distorted\n\n")
}

# =============================================================================
# 10. SAVE OUTPUTS
# =============================================================================

cat("=== SAVING OUTPUTS ===\n\n")

# 10.1 Save corrected data
output_file <- "combat_corrected.csv"
write_csv(df_corrected, output_file)
cat(sprintf("✓ Corrected data saved: %s\n", output_file))
cat(sprintf("  Size: %.2f MB\n", file.size(output_file) / 1e6))

# 10.2 Save metadata
metadata_output <- list(
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  runtime_seconds = as.numeric(runtime),
  n_proteins = nrow(expr_matrix),
  n_samples = ncol(expr_matrix),
  n_studies = length(unique(metadata$Study_ID)),
  n_age_groups = length(unique(metadata$Age_Group)),
  n_tissues = length(unique(metadata$Tissue_Compartment)),
  missing_imputed_percent = na_percent,
  icc_before = ifelse(exists("icc_before"), icc_before, NA),
  icc_after = ifelse(exists("icc_after"), icc_after, NA),
  variance_before = var_before,
  variance_after = var_after,
  age_effect_correlation = effect_corr,
  combat_parameters = list(
    par.prior = TRUE,
    ref.batch = NULL,
    covariates = "Age_Group + Tissue_Compartment"
  )
)

metadata_file <- "combat_metadata.json"
write_json(metadata_output, metadata_file, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("✓ Metadata saved: %s\n\n", metadata_file))

# =============================================================================
# 11. GENERATE DIAGNOSTIC PLOTS
# =============================================================================

cat("Generating diagnostic plots...\n")

# 11.1 PCA Before/After
cat("  - PCA comparison...\n")

library(ggplot2)

# Function to run PCA
run_pca <- function(expr_mat) {
  pca_result <- prcomp(t(expr_mat), scale. = FALSE, center = TRUE)
  pca_df <- as.data.frame(pca_result$x[, 1:5])
  pca_df$Sample_ID <- rownames(pca_result$x)

  var_explained <- summary(pca_result)$importance[2, 1:5] * 100

  return(list(df = pca_df, var_explained = var_explained))
}

# PCA before
pca_before <- run_pca(expr_matrix)
pca_before$df <- pca_before$df %>%
  left_join(metadata, by = "Sample_ID")

# PCA after
pca_after <- run_pca(combat_corrected)
pca_after$df <- pca_after$df %>%
  left_join(metadata, by = "Sample_ID")

# Plot
p_before <- ggplot(pca_before$df, aes(x = PC1, y = PC2, color = Study_ID)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(
    title = "Before ComBat Correction",
    x = sprintf("PC1 (%.1f%% variance)", pca_before$var_explained[1]),
    y = sprintf("PC2 (%.1f%% variance)", pca_before$var_explained[2])
  ) +
  theme_minimal() +
  theme(legend.position = "right")

p_after <- ggplot(pca_after$df, aes(x = PC1, y = PC2, color = Study_ID)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(
    title = "After ComBat Correction",
    x = sprintf("PC1 (%.1f%% variance)", pca_after$var_explained[1]),
    y = sprintf("PC2 (%.1f%% variance)", pca_after$var_explained[2])
  ) +
  theme_minimal() +
  theme(legend.position = "right")

# Save combined plot
png("../diagnostics/combat_before_after_pca.png", width = 1400, height = 600, res = 100)
gridExtra::grid.arrange(p_before, p_after, ncol = 2)
dev.off()

cat("  ✓ PCA plot saved\n")

# 11.2 Effect size correlation plot
cat("  - Effect size preservation plot...\n")

p_effect <- ggplot(age_effects_comparison,
                   aes(x = Delta_Z_Before, y = Delta_Z_After)) +
  geom_point(alpha = 0.3, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  labs(
    title = "Biological Signal Preservation",
    subtitle = sprintf("Correlation: r = %.3f", effect_corr),
    x = "Age Effect Before Correction (Δz)",
    y = "Age Effect After Correction (Δz)"
  ) +
  theme_minimal() +
  coord_fixed()

ggsave("../diagnostics/combat_effect_preservation.png", p_effect,
       width = 8, height = 6, dpi = 100)

cat("  ✓ Effect preservation plot saved\n\n")

# =============================================================================
# 12. SUMMARY
# =============================================================================

cat("=== COMBAT CORRECTION SUMMARY ===\n\n")

cat("Input:\n")
cat(sprintf("  - Dataset: %s\n", data_path))
cat(sprintf("  - Proteins: %d\n", nrow(expr_matrix)))
cat(sprintf("  - Samples: %d\n", ncol(expr_matrix)))
cat(sprintf("  - Studies (batches): %d\n", length(unique(metadata$Study_ID))))
cat("\n")

cat("Correction:\n")
cat(sprintf("  - Method: ComBat (empirical Bayes)\n"))
cat(sprintf("  - Covariates preserved: Age_Group, Tissue_Compartment\n"))
cat(sprintf("  - Missing values: %.1f%% (median imputed)\n", na_percent))
cat(sprintf("  - Runtime: %.1f seconds\n", as.numeric(runtime)))
cat("\n")

cat("Validation:\n")
if (!is.na(icc_before) && !is.na(icc_after)) {
  cat(sprintf("  - ICC before: %.3f (poor)\n", icc_before))
  cat(sprintf("  - ICC after: %.3f ", icc_after))
  if (icc_after > 0.5) {
    cat("(moderate - SUCCESS)\n")
  } else if (icc_after > 0.4) {
    cat("(fair - PARTIAL SUCCESS)\n")
  } else {
    cat("(poor - LIMITED SUCCESS)\n")
  }
} else {
  cat("  - ICC: Not calculated\n")
}
cat(sprintf("  - Age effect correlation: r = %.3f ", effect_corr))
if (effect_corr > 0.7) {
  cat("(preserved)\n")
} else {
  cat("(WARNING: signal distortion)\n")
}
cat("\n")

cat("Outputs:\n")
cat(sprintf("  - Corrected data: %s\n", output_file))
cat(sprintf("  - Metadata: %s\n", metadata_file))
cat("  - Diagnostics: ../diagnostics/\n")
cat("\n")

cat("Next Steps:\n")
cat("  1. Review PCA plot: ../diagnostics/combat_before_after_pca.png\n")
cat("  2. Check effect preservation: ../diagnostics/combat_effect_preservation.png\n")
cat("  3. Run validation/01_calculate_metrics.py for comprehensive metrics\n")
cat("  4. Re-test hypotheses with validation/02_retest_hypotheses.py\n")
cat("\n")

cat("=== COMBAT CORRECTION COMPLETE ===\n")
