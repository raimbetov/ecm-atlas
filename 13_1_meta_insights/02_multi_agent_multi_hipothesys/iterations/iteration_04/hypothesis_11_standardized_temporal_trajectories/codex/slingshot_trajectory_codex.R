#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(slingshot)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript slingshot_trajectory_codex.R <input_matrix_csv> <output_csv>")
}

input_path <- args[[1]]
output_path <- args[[2]]

if (!file.exists(input_path)) {
  stop(paste("Input file does not exist:", input_path))
}

mat <- read.csv(input_path, row.names = 1, check.names = FALSE)
if (nrow(mat) < 3) {
  stop("Slingshot requires at least 3 observations (tissues).")
}

mat_matrix <- as.matrix(mat)
mat_scaled <- scale(mat_matrix, center = TRUE, scale = TRUE)

pca <- prcomp(mat_scaled, center = FALSE, scale. = FALSE)
num_components <- min(3, ncol(pca$x))
reduced <- pca$x[, seq_len(num_components), drop = FALSE]

k <- min(4, nrow(reduced))
set.seed(42)
clusters <- kmeans(reduced, centers = k)$cluster

sce <- SingleCellExperiment(assays = list(counts = t(mat_matrix)))
reducedDims(sce)$PCA <- reduced
colData(sce)$cluster <- as.factor(clusters)

sce <- slingshot(sce, clusterLabels = "cluster", reducedDim = "PCA")
pt <- slingPseudotime(sce)
if (is.list(pt)) {
  pt <- do.call(cbind, pt)
}

pt_numeric <- apply(pt, 1, function(x) mean(x, na.rm = TRUE))
result <- data.frame(
  Tissue = rownames(mat_matrix),
  pseudo_time_score = as.numeric(pt_numeric)
)

result <- result[order(result$pseudo_time_score), ]
write.csv(result, output_path, row.names = FALSE)
