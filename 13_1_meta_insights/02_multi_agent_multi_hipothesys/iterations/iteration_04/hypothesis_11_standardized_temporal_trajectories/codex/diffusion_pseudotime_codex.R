#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(destiny)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript diffusion_pseudotime_codex.R <input_matrix_csv> <output_csv>")
}

input_path <- args[[1]]
output_path <- args[[2]]

if (!file.exists(input_path)) {
  stop(paste("Input file does not exist:", input_path))
}

mat <- read.csv(input_path, row.names = 1, check.names = FALSE)
mat_matrix <- as.matrix(mat)

if (nrow(mat_matrix) < 3) {
  stop("Diffusion pseudotime requires at least 3 observations.")
}

sce <- DiffusionMap(mat_matrix, sigma = 'local')
pt <- destiny::dpt(sce)

result <- data.frame(
  Tissue = rownames(mat_matrix),
  pseudo_time_score = as.numeric(pt)
)

result <- result[order(result$pseudo_time_score), ]
write.csv(result, output_path, row.names = FALSE)
