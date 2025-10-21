import os
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = "/Users/Kravtsovd/projects/ecm-atlas"
OUTPUT_DIR = os.path.join(BASE_DIR, "13_1_meta_insights", "codex")
V1_PATH = os.path.join(BASE_DIR, "08_merged_ecm_dataset", "merged_ecm_aging_zscore.csv")
V2_PATH = os.path.join(
    BASE_DIR,
    "14_exploratory_batch_correction",
    "multi_agents_ver1_for_batch_cerection",
    "step2_batch",
    "codex",
    "merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv",
)

INSIGHTS = [
    ("G1", "GOLD", "Universal markers are rare"),
    ("G2", "GOLD", "PCOLCE quality paradigm"),
    ("G3", "GOLD", "Batch effects dominate biology"),
    ("G4", "GOLD", "Weak signals compound"),
    ("G5", "GOLD", "Entropy transitions"),
    ("G6", "GOLD", "Compartment antagonistic remodeling"),
    ("G7", "GOLD", "Species divergence"),
    ("S1", "SILVER", "Fibrinogen coagulation cascade"),
    ("S2", "SILVER", "Temporal intervention windows"),
    ("S3", "SILVER", "TIMP3 lock-in"),
    ("S4", "SILVER", "Tissue-specific TSI"),
    ("S5", "SILVER", "Biomarker panel"),
]

pd.set_option("display.max_columns", None)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_universality(df: pd.DataFrame) -> pd.DataFrame:
    df_valid = df.dropna(subset=["Zscore_Delta"]).copy()
    gene_col = "Gene_Symbol" if "Gene_Symbol" in df_valid.columns else "Canonical_Gene_Symbol"
    df_valid[gene_col] = df_valid[gene_col].astype(str)
    tissue_total = df_valid["Tissue_Compartment"].nunique()
    records: List[Dict[str, float]] = []

    for gene, gene_data in df_valid.groupby(gene_col):
        zscores = clean_numeric(gene_data["Zscore_Delta"]).dropna()
        if zscores.empty:
            continue
        n_tissues = gene_data["Tissue_Compartment"].nunique()
        n_measurements = len(zscores)
        n_up = (zscores > 0).sum()
        n_down = (zscores < 0).sum()
        direction_consistency = max(n_up, n_down) / len(zscores)
        predominant_direction = "UP" if n_up >= n_down else "DOWN"
        abs_mean_delta = zscores.abs().mean()
        median_delta = float(zscores.median())
        std_delta = float(zscores.std()) if len(zscores) > 1 else float("nan")
        n_strong = (zscores.abs() > 0.5).sum()
        strong_rate = n_strong / len(zscores)
        if len(zscores) >= 3:
            t_stat, p_value = stats.ttest_1samp(zscores, 0)
        else:
            t_stat, p_value = float("nan"), float("nan")
        tissue_fraction = (n_tissues / tissue_total) if tissue_total else 0
        effect_component = min(abs_mean_delta / 2.0, 1.0)
        if not math.isnan(p_value):
            significance_component = 1 - min(p_value, 1.0)
        else:
            significance_component = 0.0
        records.append(
            {
                "Gene_Symbol": gene,
                "N_Tissues": n_tissues,
                "N_Measurements": n_measurements,
                "Direction_Consistency": direction_consistency,
                "Predominant_Direction": predominant_direction,
                "Abs_Mean_Zscore_Delta": abs_mean_delta,
                "Median_Zscore_Delta": median_delta,
                "Std_Zscore_Delta": std_delta,
                "N_Strong_Effects": n_strong,
                "Strong_Effect_Rate": strong_rate,
                "Universality_Score": (
                    tissue_fraction * 0.3
                    + direction_consistency * 0.3
                    + effect_component * 0.2
                    + significance_component * 0.2
                ),
            }
        )

    return pd.DataFrame(records)


def summarize_universality(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "universal_count": 0,
            "total_genes": 0,
            "universal_pct": 0.0,
            "top_markers": [],
        }
    universal = df[
        (df["N_Tissues"] >= 3) & (df["Direction_Consistency"] >= 0.7)
    ].copy()
    total_genes = df.shape[0]
    universal_pct = (len(universal) / total_genes * 100) if total_genes else 0.0
    top_markers = (
        universal.sort_values("Universality_Score", ascending=False)
        .head(5)
        .assign(
            Summary=lambda x: x["Gene_Symbol"].astype(str)
            + " (" + x["Universality_Score"].round(3).astype(str) + ")"
        )["Summary"].tolist()
    )
    return {
        "universal_count": int(len(universal)),
        "total_genes": int(total_genes),
        "universal_pct": universal_pct,
        "top_markers": top_markers,
    }


def extract_gene_metrics(df: pd.DataFrame, gene: str) -> Dict[str, float]:
    subset = df[df["Canonical_Gene_Symbol"].str.upper() == gene.upper()].copy()
    subset = subset.dropna(subset=["Zscore_Delta"])
    if subset.empty:
        return {
            "n_rows": 0,
            "n_tissues": 0,
            "n_studies": 0,
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "std_delta": float("nan"),
            "direction_consistency": float("nan"),
        }
    zscores = clean_numeric(subset["Zscore_Delta"]).dropna()
    if zscores.empty:
        return {
            "n_rows": int(len(subset)),
            "n_tissues": subset["Tissue_Compartment"].nunique(),
            "n_studies": subset["Study_ID"].nunique(),
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "std_delta": float("nan"),
            "direction_consistency": float("nan"),
        }
    n_up = (zscores > 0).sum()
    n_down = (zscores < 0).sum()
    direction_consistency = max(n_up, n_down) / len(zscores)
    return {
        "n_rows": int(len(subset)),
        "n_tissues": subset["Tissue_Compartment"].nunique(),
        "n_studies": subset["Study_ID"].nunique(),
        "mean_delta": float(zscores.mean()),
        "median_delta": float(zscores.median()),
        "std_delta": float(zscores.std()) if len(zscores) > 1 else float("nan"),
        "direction_consistency": direction_consistency,
    }


def build_sample_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, object]] = []
    for (study, tissue, compartment), group in df.groupby(
        ["Study_ID", "Tissue", "Tissue_Compartment"]
    ):
        base_id = f"{study}_{tissue}_{compartment}"
        for age_label, column in (("Old", "Zscore_Old"), ("Young", "Zscore_Young")):
            age_values = group[["Canonical_Gene_Symbol", column]].dropna()
            if age_values.empty:
                continue
            record = {
                "Sample_ID": f"{base_id}_{age_label}",
                "Study_ID": study,
                "Tissue": tissue,
                "Tissue_Compartment": compartment,
                "Age_Group": age_label,
            }
            record.update(
                dict(zip(age_values["Canonical_Gene_Symbol"], age_values[column]))
            )
            records.append(record)
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    matrix = pd.DataFrame(records)
    metadata = matrix[[
        "Sample_ID",
        "Study_ID",
        "Tissue",
        "Tissue_Compartment",
        "Age_Group",
    ]].copy()
    features = matrix.drop(columns=metadata.columns)
    features = features.fillna(0.0)
    return metadata, features


def compute_pca_signals(df: pd.DataFrame) -> Dict[str, float]:
    metadata, features = build_sample_matrix(df)
    if metadata.empty or features.empty:
        return {
            "n_samples": 0,
            "n_features": 0,
            "silhouette_age": float("nan"),
            "silhouette_study": float("nan"),
            "age_minus_study": float("nan"),
        }
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    n_components = min(10, features_scaled.shape[0], features_scaled.shape[1])
    if n_components < 2:
        return {
            "n_samples": features_scaled.shape[0],
            "n_features": features_scaled.shape[1],
            "silhouette_age": float("nan"),
            "silhouette_study": float("nan"),
            "age_minus_study": float("nan"),
        }
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(features_scaled)
    base = pcs[:, : min(5, pcs.shape[1])]
    le_age = LabelEncoder()
    le_study = LabelEncoder()
    age_labels = le_age.fit_transform(metadata["Age_Group"])
    study_labels = le_study.fit_transform(metadata["Study_ID"])
    try:
        sil_age = float(silhouette_score(base, age_labels))
    except Exception:
        sil_age = float("nan")
    try:
        sil_study = float(silhouette_score(base, study_labels))
    except Exception:
        sil_study = float("nan")
    return {
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "silhouette_age": sil_age,
        "silhouette_study": sil_study,
        "age_minus_study": sil_age - sil_study
        if not (math.isnan(sil_age) or math.isnan(sil_study))
        else float("nan"),
    }


def compute_weak_signal_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df_valid = df.dropna(subset=["Zscore_Delta"]).copy()
    stats_rows: List[Dict[str, object]] = []
    for gene, gene_data in df_valid.groupby("Canonical_Gene_Symbol"):
        zscores = clean_numeric(gene_data["Zscore_Delta"]).dropna()
        if len(zscores) < 2:
            continue
        mean_delta = float(zscores.mean())
        abs_mean_delta = float(zscores.abs().mean())
        std_delta = float(zscores.std()) if len(zscores) > 1 else 0.0
        n_measurements = len(zscores)
        n_up = (zscores > 0).sum()
        n_down = (zscores < 0).sum()
        direction_consistency = max(n_up, n_down) / len(zscores)
        dominant_direction = "increase" if n_up >= n_down else "decrease"
        cumulative_effect = float(zscores.sum())
        if len(zscores) >= 3:
            p_values = []
            for z in zscores:
                if dominant_direction == "increase":
                    p = 1 - stats.norm.cdf(z)
                else:
                    p = stats.norm.cdf(z)
                p_values.append(max(p, 1e-16))
            chi2_stat, combined_p = stats.combine_pvalues(p_values, method="fisher")
            _ = chi2_stat
        else:
            combined_p = 1.0
        is_strict = (
            0.3 <= abs_mean_delta <= 0.8
            and n_measurements >= 8
            and direction_consistency >= 0.70
            and std_delta < 0.3
        )
        is_moderate = (
            0.3 <= abs_mean_delta <= 1.0
            and n_measurements >= 6
            and direction_consistency >= 0.65
            and std_delta < 0.4
        )
        stats_rows.append(
            {
                "Gene_Symbol": gene,
                "N_Measurements": n_measurements,
                "N_Studies": gene_data["Study_ID"].nunique(),
                "Mean_Zscore_Delta": mean_delta,
                "Abs_Mean_Zscore_Delta": abs_mean_delta,
                "Std_Zscore_Delta": std_delta,
                "Direction_Consistency": direction_consistency,
                "Dominant_Direction": dominant_direction,
                "Cumulative_Effect": cumulative_effect,
                "Combined_P_Value": combined_p,
                "Is_Weak_Signal": is_strict,
                "Is_Moderate_Weak": is_moderate,
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    metrics = {
        "strict_count": int(stats_df["Is_Weak_Signal"].sum()),
        "moderate_count": int(stats_df["Is_Moderate_Weak"].sum()),
        "median_abs_delta": float(stats_df["Abs_Mean_Zscore_Delta"].median())
        if not stats_df.empty
        else float("nan"),
    }
    return stats_df, metrics


def calculate_entropy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    df_valid = df.copy()
    for protein, prot_data in df_valid.groupby("Canonical_Gene_Symbol"):
        abundances = pd.concat(
            [
                clean_numeric(prot_data["Abundance_Old"]).dropna(),
                clean_numeric(prot_data["Abundance_Young"]).dropna(),
            ]
        )
        if abundances.empty:
            continue
        shifted = abundances - abundances.min() + 1
        probs = shifted / shifted.sum()
        shannon = float(-(probs * np.log2(probs + 1e-10)).sum())
        vals = clean_numeric(prot_data["Zscore_Delta"]).dropna()
        if len(vals) < 2:
            predictability = float("nan")
            direction = "insufficient"
        else:
            pos = (vals > 0).sum()
            neg = (vals < 0).sum()
            total = len(vals)
            predictability = max(pos, neg) / total if total else float("nan")
            direction = "increase" if pos >= neg else "decrease"
        old_vals = clean_numeric(prot_data["Abundance_Old"]).dropna()
        young_vals = clean_numeric(prot_data["Abundance_Young"]).dropna()
        cv_old = old_vals.std() / abs(old_vals.mean()) if len(old_vals) >= 2 and old_vals.mean() else float("nan")
        cv_young = young_vals.std() / abs(young_vals.mean()) if len(young_vals) >= 2 and young_vals.mean() else float("nan")
        if math.isnan(cv_old) or math.isnan(cv_young):
            transition = float("nan")
        else:
            transition = abs(cv_old - cv_young)
        records.append(
            {
                "Protein": protein,
                "N_Studies": prot_data["Study_ID"].nunique(),
                "N_Tissues": prot_data["Tissue"].nunique(),
                "Matrisome_Category": prot_data["Matrisome_Category"].mode()[0]
                if not prot_data["Matrisome_Category"].mode().empty
                else "Unknown",
                "Matrisome_Division": prot_data["Matrisome_Division"].mode()[0]
                if not prot_data["Matrisome_Division"].mode().empty
                else "Unknown",
                "Shannon_Entropy": shannon,
                "Entropy_Transition": transition,
                "Predictability_Score": predictability,
                "Aging_Direction": direction,
                "Mean_Zscore_Delta": float(vals.mean()) if len(vals) else float("nan"),
                "N_Observations": len(prot_data),
            }
        )
    return pd.DataFrame(records)


def identify_entropy_switchers(df_entropy: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df_entropy[df_entropy["Entropy_Transition"] >= threshold].copy()


def compute_antagonistic_events(df: pd.DataFrame) -> pd.DataFrame:
    df_valid = df.dropna(subset=["Zscore_Delta"]).copy()
    events: List[Dict[str, object]] = []
    for organ, organ_df in df_valid.groupby("Organ"):
        compartments = organ_df["Tissue_Compartment"].unique()
        if len(compartments) < 2:
            continue
        for gene, gene_df in organ_df.groupby("Canonical_Gene_Symbol"):
            comp_means = (
                gene_df.groupby("Tissue_Compartment")["Zscore_Delta"].mean()
            )
            if comp_means.shape[0] < 2:
                continue
            for i, comp_a in enumerate(compartments):
                if comp_a not in comp_means:
                    continue
                for comp_b in compartments[i + 1 :]:
                    if comp_b not in comp_means:
                        continue
                    delta_a = comp_means[comp_a]
                    delta_b = comp_means[comp_b]
                    if delta_a == 0 or delta_b == 0:
                        continue
                    if delta_a * delta_b >= 0:
                        continue
                    divergence = abs(delta_a - delta_b)
                    if divergence < 1.0:
                        continue
                    events.append(
                        {
                            "Organ": organ,
                            "Gene_Symbol": gene,
                            "Compartment_A": comp_a,
                            "Compartment_B": comp_b,
                            "Delta_A": float(delta_a),
                            "Delta_B": float(delta_b),
                            "Divergence": divergence,
                        }
                    )
    return pd.DataFrame(events)


def compute_species_metrics(df: pd.DataFrame) -> Dict[str, object]:
    df_valid = df.dropna(subset=["Zscore_Delta"]).copy()
    species_counts = (
        df_valid.groupby("Species")["Canonical_Gene_Symbol"].nunique().to_dict()
    )
    total_genes = df_valid["Canonical_Gene_Symbol"].nunique()
    species_per_gene = (
        df_valid.groupby("Canonical_Gene_Symbol")["Species"].nunique()
    )
    multispecies_genes = species_per_gene[species_per_gene > 1]
    shared_genes = multispecies_genes.index.tolist()
    gene_species_delta = (
        df_valid.groupby(["Canonical_Gene_Symbol", "Species"])["Zscore_Delta"].mean()
    )
    human_mouse = []
    for gene in shared_genes:
        try:
            human_delta = gene_species_delta.loc[(gene, "Homo sapiens")]
            mouse_delta = gene_species_delta.loc[(gene, "Mus musculus")]
            human_mouse.append((human_delta, mouse_delta, gene))
        except KeyError:
            continue
    if len(human_mouse) >= 2:
        human_vals, mouse_vals, genes = zip(*human_mouse)
        correlation = np.corrcoef(human_vals, mouse_vals)[0, 1]
    else:
        correlation = float("nan")
        genes = []
    return {
        "species_counts": species_counts,
        "total_genes": int(total_genes),
        "shared_gene_count": int(len(shared_genes)),
        "human_mouse_genes": list(genes),
        "human_mouse_corr": float(correlation),
    }


def compute_coagulation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    targets = ["FGA", "FGB", "FGG", "SERPINC1", "F2", "PLG", "F12"]
    rows: List[Dict[str, object]] = []
    df_valid = df.dropna(subset=["Zscore_Delta"]).copy()
    for gene in targets:
        subset = df_valid[df_valid["Canonical_Gene_Symbol"].str.upper() == gene]
        if subset.empty:
            continue
        rows.append(
            {
                "Gene_Symbol": gene,
                "Mean_Delta": float(clean_numeric(subset["Zscore_Delta"]).mean()),
                "Median_Delta": float(clean_numeric(subset["Zscore_Delta"]).median()),
                "N_Tissues": subset["Tissue"].nunique(),
                "Max_Tissue": subset.loc[
                    subset["Zscore_Delta"].abs().idxmax(), "Tissue"
                ]
                if subset["Zscore_Delta"].abs().any()
                else None,
                "Max_Compartment": subset.loc[
                    subset["Zscore_Delta"].abs().idxmax(), "Tissue_Compartment"
                ]
                if subset["Zscore_Delta"].abs().any()
                else None,
            }
        )
    return pd.DataFrame(rows)


def compute_temporal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df_valid = df.dropna(subset=["Zscore_Delta"]).copy()
    records: List[Dict[str, object]] = []
    for protein, prot_data in df_valid.groupby("Canonical_Gene_Symbol"):
        if len(prot_data) < 2:
            continue
        mean_delta = float(clean_numeric(prot_data["Zscore_Delta"]).mean())
        std_delta = float(clean_numeric(prot_data["Zscore_Delta"]).std())
        consistency = 1 - (std_delta / (abs(mean_delta) + 0.1))
        if abs(mean_delta) > 0.5 and consistency > 0.5:
            pattern = "Late_Increase" if mean_delta > 0 else "Late_Decrease"
        elif abs(mean_delta) > 0.2:
            pattern = "Chronic_Change"
        else:
            pattern = "Stable"
        records.append(
            {
                "Protein": protein,
                "Mean_Zscore_Delta": mean_delta,
                "Std_Zscore_Delta": std_delta,
                "Consistency": consistency,
                "Temporal_Pattern": pattern,
                "Matrisome_Category": prot_data["Matrisome_Category"].mode()[0]
                if not prot_data["Matrisome_Category"].mode().empty
                else "Unknown",
            }
        )
    return pd.DataFrame(records)


def compute_timp3_metrics(df: pd.DataFrame) -> Dict[str, object]:
    df_valid = df.dropna(subset=["Zscore_Delta"]).copy()
    metrics = extract_gene_metrics(df_valid, "TIMP3")
    mmps = df_valid[df_valid["Canonical_Gene_Symbol"].str.match(r"^MMP", na=False)]
    vegf = df_valid[df_valid["Canonical_Gene_Symbol"].str.contains("VEGF", na=False)]
    mean_mmp = float(clean_numeric(mmps["Zscore_Delta"]).mean()) if not mmps.empty else float("nan")
    mean_vegf = float(clean_numeric(vegf["Zscore_Delta"]).mean()) if not vegf.empty else float("nan")
    if not math.isnan(metrics.get("mean_delta", float("nan"))):
        deficiency = (mean_mmp + mean_vegf) / (metrics["mean_delta"] + 0.1)
    else:
        deficiency = float("nan")
    metrics.update(
        {
            "mean_mmp_delta": mean_mmp,
            "mean_vegf_delta": mean_vegf,
            "deficiency_index": deficiency,
        }
    )
    return metrics


def compute_tsi_markers(df: pd.DataFrame) -> pd.DataFrame:
    df_valid = df.dropna(subset=["Zscore_Delta", "Canonical_Gene_Symbol", "Tissue"]).copy()
    grouped = (
        df_valid.groupby(["Canonical_Gene_Symbol", "Tissue"])["Zscore_Delta"].mean().reset_index()
    )
    records: List[Dict[str, object]] = []
    for gene, gene_df in grouped.groupby("Canonical_Gene_Symbol"):
        if len(gene_df) < 2:
            continue
        for _, row in gene_df.iterrows():
            tissue = row["Tissue"]
            tissue_z = abs(row["Zscore_Delta"])
            others = gene_df[gene_df["Tissue"] != tissue]["Zscore_Delta"].abs()
            if others.empty:
                continue
            mean_other = others.mean()
            if mean_other > 0.1:
                tsi = tissue_z / mean_other
            else:
                tsi = tissue_z / 0.1
            records.append(
                {
                    "Gene_Symbol": gene,
                    "Tissue": tissue,
                    "Tissue_Z": tissue_z,
                    "Mean_Others": mean_other,
                    "TSI": tsi,
                }
            )
    tsi_df = pd.DataFrame(records)
    markers = tsi_df[
        (tsi_df["Tissue_Z"] > 2.0)
        & (tsi_df["Mean_Others"] < 0.5)
        & (tsi_df["TSI"] > 3.0)
    ].copy()
    return markers


def compute_biomarker_panel(df: pd.DataFrame) -> pd.DataFrame:
    df_valid = df.dropna(subset=["Zscore_Delta"]).copy()
    rows: List[Dict[str, object]] = []
    for gene, gene_df in df_valid.groupby("Canonical_Gene_Symbol"):
        protein_name = gene_df["Protein_Name"].iloc[0]
        matrisome_cat = gene_df["Matrisome_Category"].mode()[0] if not gene_df["Matrisome_Category"].mode().empty else "Unknown"
        feasibility = 0
        detection = "Mass spectrometry"
        sample = "Blood"
        if matrisome_cat in ["Secreted Factors", "ECM Regulators"]:
            feasibility += 2
            detection = "ELISA"
        if matrisome_cat == "ECM Glycoproteins":
            feasibility += 1
        if gene.startswith("COL4"):
            feasibility += 2
            sample = "Urine"
            detection = "ELISA"
        elif gene.startswith("COL"):
            feasibility += 1
            detection = "ELISA"
        if gene.startswith("MMP") or gene.startswith("TIMP"):
            feasibility += 2
            detection = "ELISA"
        if gene in ["F2", "F9", "F10", "F12", "PLG", "FGA", "FGB", "FGG", "VWF", "VTN"]:
            feasibility += 2
            detection = "ELISA"
        if isinstance(protein_name, str) and any(
            kw.lower() in protein_name.lower()
            for kw in [
                "fibrinogen",
                "vitronectin",
                "hemopexin",
                "plasminogen",
                "prothrombin",
                "antithrombin",
                "fibronectin",
                "thrombospondin",
                "periostin",
                "tenascin",
                "versican",
            ]
        ):
            feasibility += 1
        feasibility = min(feasibility, 5)
        zscores = clean_numeric(gene_df["Zscore_Delta"]).dropna()
        if zscores.empty:
            continue
        mean_delta = float(zscores.mean())
        abs_mean = float(zscores.abs().mean())
        std_delta = float(zscores.std()) if len(zscores) > 1 else 0.0
        direction_consistency = max((zscores > 0).sum(), (zscores < 0).sum()) / len(zscores)
        if len(zscores) >= 3:
            t_stat, p_value = stats.ttest_1samp(zscores, 0)
        else:
            t_stat, p_value = float("nan"), float("nan")
        score = (
            feasibility * 0.4
            + abs_mean * 0.3
            + direction_consistency * 0.2
            + (1 - min(p_value, 1.0) if not math.isnan(p_value) else 0) * 0.1
        )
        rows.append(
            {
                "Gene_Symbol": gene,
                "Protein_Name": protein_name,
                "Feasibility": feasibility,
                "Sample": sample,
                "Detection": detection,
                "Mean_Delta": mean_delta,
                "Abs_Mean_Delta": abs_mean,
                "Direction_Consistency": direction_consistency,
                "Composite_Score": score,
            }
        )
    return pd.DataFrame(rows)


def classify_status(baseline: float, current: float, direction: str = "increase") -> str:
    if math.isnan(baseline) or math.isnan(current):
        return "MODIFIED"
    if direction == "increase":
        if current >= baseline * 0.9:
            return "CONFIRMED"
        if current >= baseline * 0.5:
            return "MODIFIED"
        return "REJECTED"
    if direction == "decrease":
        if current <= baseline * 0.9:
            return "CONFIRMED"
        if current <= baseline * 1.1:
            return "MODIFIED"
        return "REJECTED"
    return "MODIFIED"


def percent_change(baseline: float, current: float) -> float:
    if baseline == 0 or math.isnan(baseline) or math.isnan(current):
        return float("nan")
    return (current - baseline) / abs(baseline) * 100


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_v1 = load_dataset(V1_PATH)
    df_v2 = load_dataset(V2_PATH)

    results_rows: List[Dict[str, object]] = []
    validated_proteins_rows: List[Dict[str, object]] = []
    new_discoveries_rows: List[Dict[str, object]] = []

    # G1
    uni_v1 = compute_universality(df_v1)
    uni_v2 = compute_universality(df_v2)
    summary_v1 = summarize_universality(uni_v1)
    summary_v2 = summarize_universality(uni_v2)
    status_g1 = classify_status(summary_v1["universal_pct"], summary_v2["universal_pct"])
    results_rows.append(
        {
            "Insight_ID": "G1",
            "Tier": "GOLD",
            "Insight_Name": "Universal markers are rare",
            "Baseline_Metric": summary_v1["universal_pct"],
            "V2_Metric": summary_v2["universal_pct"],
            "Change_Percent": percent_change(
                summary_v1["universal_pct"], summary_v2["universal_pct"]
            ),
            "Status": status_g1,
            "Notes": ", ".join(summary_v2["top_markers"]),
        }
    )
    for marker in uni_v2.sort_values("Universality_Score", ascending=False).head(20).itertuples():
        validated_proteins_rows.append(
            {
                "Insight_ID": "G1",
                "Gene_Symbol": marker.Gene_Symbol,
                "Metric": "Universality_Score",
                "V2_Value": marker.Universality_Score,
                "V1_Value": float(
                    uni_v1.loc[
                        uni_v1["Gene_Symbol"] == marker.Gene_Symbol,
                        "Universality_Score",
                    ].max()
                )
                if not uni_v1[uni_v1["Gene_Symbol"] == marker.Gene_Symbol].empty
                else float("nan"),
            }
        )
    v1_universal_set = set(
        uni_v1["Gene_Symbol"][
            (uni_v1["N_Tissues"] >= 3) & (uni_v1["Direction_Consistency"] >= 0.7)
        ]
    )
    v2_universal_set = set(
        uni_v2["Gene_Symbol"][
            (uni_v2["N_Tissues"] >= 3) & (uni_v2["Direction_Consistency"] >= 0.7)
        ]
    )
    new_universals = v2_universal_set - v1_universal_set
    for gene in sorted(list(new_universals))[:20]:
        marker = uni_v2[uni_v2["Gene_Symbol"] == gene].iloc[0]
        new_discoveries_rows.append(
            {
                "Discovery_Type": "Universal Marker",
                "Gene_Symbol": gene,
                "Metric_Value": marker["Universality_Score"],
                "Context": "Emerges as universal in V2",
            }
        )

    # G2
    pcolce_v1 = extract_gene_metrics(df_v1, "PCOLCE")
    pcolce_v2 = extract_gene_metrics(df_v2, "PCOLCE")
    status_g2 = "CONFIRMED" if pcolce_v2["mean_delta"] < 0 else "REJECTED"
    results_rows.append(
        {
            "Insight_ID": "G2",
            "Tier": "GOLD",
            "Insight_Name": "PCOLCE quality paradigm",
            "Baseline_Metric": pcolce_v1["mean_delta"],
            "V2_Metric": pcolce_v2["mean_delta"],
            "Change_Percent": percent_change(pcolce_v1["mean_delta"], pcolce_v2["mean_delta"]),
            "Status": status_g2,
            "Notes": f"Consistency {pcolce_v2['direction_consistency']:.2f}, studies {pcolce_v2['n_studies']}",
        }
    )
    validated_proteins_rows.append(
        {
            "Insight_ID": "G2",
            "Gene_Symbol": "PCOLCE",
            "Metric": "Mean_Delta",
            "V2_Value": pcolce_v2["mean_delta"],
            "V1_Value": pcolce_v1["mean_delta"],
        }
    )

    # G3
    pca_v1 = compute_pca_signals(df_v1)
    pca_v2 = compute_pca_signals(df_v2)
    status_g3 = "CONFIRMED"
    if math.isnan(pca_v2["silhouette_age"]) or pca_v2["silhouette_age"] <= pca_v2["silhouette_study"]:
        status_g3 = "MODIFIED"
    if math.isnan(pca_v2["silhouette_age"]) or pca_v2["silhouette_age"] < 0:
        status_g3 = "REJECTED"
    results_rows.append(
        {
            "Insight_ID": "G3",
            "Tier": "GOLD",
            "Insight_Name": "Batch effects dominate biology",
            "Baseline_Metric": pca_v1["silhouette_age"],
            "V2_Metric": pca_v2["silhouette_age"],
            "Change_Percent": percent_change(pca_v1["silhouette_age"], pca_v2["silhouette_age"]),
            "Status": status_g3,
            "Notes": f"Age-Stdy Δ {pca_v2['age_minus_study']:.3f}",
        }
    )

    # G4
    weak_v1_df, weak_v1_metrics = compute_weak_signal_metrics(df_v1)
    weak_v2_df, weak_v2_metrics = compute_weak_signal_metrics(df_v2)
    status_g4 = classify_status(weak_v1_metrics["moderate_count"], weak_v2_metrics["moderate_count"])
    results_rows.append(
        {
            "Insight_ID": "G4",
            "Tier": "GOLD",
            "Insight_Name": "Weak signals compound",
            "Baseline_Metric": weak_v1_metrics["moderate_count"],
            "V2_Metric": weak_v2_metrics["moderate_count"],
            "Change_Percent": percent_change(
                weak_v1_metrics["moderate_count"], weak_v2_metrics["moderate_count"]
            ),
            "Status": status_g4,
            "Notes": "Top: "
            + ", ".join(
                weak_v2_df.sort_values("Abs_Mean_Zscore_Delta", ascending=False)
                .head(5)["Gene_Symbol"]
                .tolist()
            ),
        }
    )
    for row in weak_v2_df[weak_v2_df["Is_Moderate_Weak"]].head(30).itertuples():
        validated_proteins_rows.append(
            {
                "Insight_ID": "G4",
                "Gene_Symbol": row.Gene_Symbol,
                "Metric": "Abs_Mean_Delta",
                "V2_Value": row.Abs_Mean_Zscore_Delta,
                "V1_Value": float(
                    weak_v1_df.loc[
                        weak_v1_df["Gene_Symbol"] == row.Gene_Symbol,
                        "Abs_Mean_Zscore_Delta",
                    ].mean()
                ),
            }
        )

    # G5
    entropy_v1 = calculate_entropy_metrics(df_v1)
    entropy_v2 = calculate_entropy_metrics(df_v2)
    if entropy_v1.empty:
        threshold = 0.3
    else:
        sorted_vals = entropy_v1["Entropy_Transition"].dropna().sort_values(ascending=False)
        threshold = float(sorted_vals.iloc[51]) if len(sorted_vals) >= 52 else float(sorted_vals.median())
    switchers_v1 = identify_entropy_switchers(entropy_v1, threshold)
    switchers_v2 = identify_entropy_switchers(entropy_v2, threshold)
    status_g5 = classify_status(len(switchers_v1), len(switchers_v2))
    results_rows.append(
        {
            "Insight_ID": "G5",
            "Tier": "GOLD",
            "Insight_Name": "Entropy transitions",
            "Baseline_Metric": len(switchers_v1),
            "V2_Metric": len(switchers_v2),
            "Change_Percent": percent_change(len(switchers_v1), len(switchers_v2)),
            "Status": status_g5,
            "Notes": f"Threshold {threshold:.3f}",
        }
    )
    for row in switchers_v2.head(30).itertuples():
        validated_proteins_rows.append(
            {
                "Insight_ID": "G5",
                "Gene_Symbol": row.Protein,
                "Metric": "Entropy_Transition",
                "V2_Value": row.Entropy_Transition,
                "V1_Value": float(
                    switchers_v1.loc[
                        switchers_v1["Protein"] == row.Protein, "Entropy_Transition"
                    ].mean()
                )
                if not switchers_v1["Protein"].eq(row.Protein).empty
                else float("nan"),
            }
        )

    # G6
    antagonism_v1 = compute_antagonistic_events(df_v1)
    antagonism_v2 = compute_antagonistic_events(df_v2)
    status_g6 = classify_status(len(antagonism_v1), len(antagonism_v2))
    results_rows.append(
        {
            "Insight_ID": "G6",
            "Tier": "GOLD",
            "Insight_Name": "Compartment antagonistic remodeling",
            "Baseline_Metric": len(antagonism_v1),
            "V2_Metric": len(antagonism_v2),
            "Change_Percent": percent_change(len(antagonism_v1), len(antagonism_v2)),
            "Status": status_g6,
            "Notes": "Top: "
            + ", ".join(
                antagonism_v2.sort_values("Divergence", ascending=False)
                .head(3)["Gene_Symbol"]
                .tolist()
            ),
        }
    )
    for row in antagonism_v2.sort_values("Divergence", ascending=False).head(30).itertuples():
        validated_proteins_rows.append(
            {
                "Insight_ID": "G6",
                "Gene_Symbol": row.Gene_Symbol,
                "Metric": "Divergence",
                "V2_Value": row.Divergence,
                "V1_Value": float(
                    antagonism_v1.loc[
                        (antagonism_v1["Gene_Symbol"] == row.Gene_Symbol)
                        & (antagonism_v1["Compartment_A"] == row.Compartment_A)
                        & (antagonism_v1["Compartment_B"] == row.Compartment_B),
                        "Divergence",
                    ].mean()
                ),
            }
        )

    # G7
    species_v1 = compute_species_metrics(df_v1)
    species_v2 = compute_species_metrics(df_v2)
    status_g7 = classify_status(
        species_v1["shared_gene_count"],
        species_v2["shared_gene_count"],
    )
    results_rows.append(
        {
            "Insight_ID": "G7",
            "Tier": "GOLD",
            "Insight_Name": "Species divergence",
            "Baseline_Metric": species_v1["shared_gene_count"],
            "V2_Metric": species_v2["shared_gene_count"],
            "Change_Percent": percent_change(
                species_v1["shared_gene_count"],
                species_v2["shared_gene_count"],
            ),
            "Status": status_g7,
            "Notes": f"Human-mouse corr {species_v2['human_mouse_corr']:.2f}",
        }
    )

    # S1
    coag_v1 = compute_coagulation_metrics(df_v1)
    coag_v2 = compute_coagulation_metrics(df_v2)
    mean_v1 = float(coag_v1["Mean_Delta"].mean()) if not coag_v1.empty else float("nan")
    mean_v2 = float(coag_v2["Mean_Delta"].mean()) if not coag_v2.empty else float("nan")
    status_s1 = classify_status(mean_v1, mean_v2)
    results_rows.append(
        {
            "Insight_ID": "S1",
            "Tier": "SILVER",
            "Insight_Name": "Fibrinogen coagulation cascade",
            "Baseline_Metric": mean_v1,
            "V2_Metric": mean_v2,
            "Change_Percent": percent_change(mean_v1, mean_v2),
            "Status": status_s1,
            "Notes": "key genes: " + ", ".join(coag_v2["Gene_Symbol"].tolist()),
        }
    )
    for row in coag_v2.itertuples():
        validated_proteins_rows.append(
            {
                "Insight_ID": "S1",
                "Gene_Symbol": row.Gene_Symbol,
                "Metric": "Mean_Delta",
                "V2_Value": row.Mean_Delta,
                "V1_Value": float(
                    coag_v1.loc[
                        coag_v1["Gene_Symbol"] == row.Gene_Symbol, "Mean_Delta"
                    ].mean()
                ),
            }
        )

    # S2
    temporal_v1 = compute_temporal_patterns(df_v1)
    temporal_v2 = compute_temporal_patterns(df_v2)
    late_share_v1 = temporal_v1["Temporal_Pattern"].value_counts(normalize=True).get(
        "Late_Increase", 0
    )
    late_share_v2 = temporal_v2["Temporal_Pattern"].value_counts(normalize=True).get(
        "Late_Increase", 0
    )
    status_s2 = classify_status(late_share_v1, late_share_v2)
    late_v1 = temporal_v1[temporal_v1["Temporal_Pattern"] == "Late_Increase"]
    late_v2 = temporal_v2[temporal_v2["Temporal_Pattern"] == "Late_Increase"]
    results_rows.append(
        {
            "Insight_ID": "S2",
            "Tier": "SILVER",
            "Insight_Name": "Temporal intervention windows",
            "Baseline_Metric": late_share_v1,
            "V2_Metric": late_share_v2,
            "Change_Percent": percent_change(late_share_v1, late_share_v2),
            "Status": status_s2,
            "Notes": f"Late markers {late_share_v2:.2%}",
        }
    )
    for row in late_v2.sort_values("Mean_Zscore_Delta", ascending=False).head(10).itertuples():
        validated_proteins_rows.append(
            {
                "Insight_ID": "S2",
                "Gene_Symbol": row.Protein,
                "Metric": "Mean_Zscore_Delta",
                "V2_Value": row.Mean_Zscore_Delta,
                "V1_Value": float(
                    late_v1.loc[late_v1["Protein"] == row.Protein, "Mean_Zscore_Delta"].mean()
                ),
            }
        )

    # S3
    timp3_v1 = compute_timp3_metrics(df_v1)
    timp3_v2 = compute_timp3_metrics(df_v2)
    status_s3 = classify_status(timp3_v1.get("mean_delta", float("nan")), timp3_v2["mean_delta"])
    results_rows.append(
        {
            "Insight_ID": "S3",
            "Tier": "SILVER",
            "Insight_Name": "TIMP3 lock-in",
            "Baseline_Metric": timp3_v1.get("mean_delta", float("nan")),
            "V2_Metric": timp3_v2.get("mean_delta", float("nan")),
            "Change_Percent": percent_change(
                timp3_v1.get("mean_delta", float("nan")), timp3_v2.get("mean_delta", float("nan"))
            ),
            "Status": status_s3,
            "Notes": f"Def idx {timp3_v2['deficiency_index']:.2f}",
        }
    )
    validated_proteins_rows.append(
        {
            "Insight_ID": "S3",
            "Gene_Symbol": "TIMP3",
            "Metric": "Mean_Delta",
            "V2_Value": timp3_v2.get("mean_delta", float("nan")),
            "V1_Value": timp3_v1.get("mean_delta", float("nan")),
        }
    )

    # S4
    tsi_v1 = compute_tsi_markers(df_v1)
    tsi_v2 = compute_tsi_markers(df_v2)
    status_s4 = classify_status(len(tsi_v1), len(tsi_v2))
    results_rows.append(
        {
            "Insight_ID": "S4",
            "Tier": "SILVER",
            "Insight_Name": "Tissue-specific TSI",
            "Baseline_Metric": len(tsi_v1),
            "V2_Metric": len(tsi_v2),
            "Change_Percent": percent_change(len(tsi_v1), len(tsi_v2)),
            "Status": status_s4,
            "Notes": "Top tissues: "
            + ", ".join(tsi_v2.sort_values("TSI", ascending=False).head(3)["Tissue"].tolist()),
        }
    )
    for row in tsi_v2.head(30).itertuples():
        validated_proteins_rows.append(
            {
                "Insight_ID": "S4",
                "Gene_Symbol": row.Gene_Symbol,
                "Metric": "TSI",
                "V2_Value": row.TSI,
                "V1_Value": float(
                    tsi_v1.loc[
                        (tsi_v1["Gene_Symbol"] == row.Gene_Symbol)
                        & (tsi_v1["Tissue"] == row.Tissue),
                        "TSI",
                    ].mean()
                ),
            }
        )

    # S5
    panel_v1 = compute_biomarker_panel(df_v1)
    panel_v2 = compute_biomarker_panel(df_v2)
    top10_v1 = panel_v1.sort_values("Composite_Score", ascending=False).head(10)
    top10_v2 = panel_v2.sort_values("Composite_Score", ascending=False).head(10)
    mean_score_v1 = float(top10_v1["Composite_Score"].mean()) if not top10_v1.empty else float("nan")
    mean_score_v2 = float(top10_v2["Composite_Score"].mean()) if not top10_v2.empty else float("nan")
    status_s5 = classify_status(mean_score_v1, mean_score_v2)
    results_rows.append(
        {
            "Insight_ID": "S5",
            "Tier": "SILVER",
            "Insight_Name": "Biomarker panel",
            "Baseline_Metric": mean_score_v1,
            "V2_Metric": mean_score_v2,
            "Change_Percent": percent_change(mean_score_v1, mean_score_v2),
            "Status": status_s5,
            "Notes": "Top genes: " + ", ".join(top10_v2["Gene_Symbol"].tolist()),
        }
    )
    for row in top10_v2.itertuples():
        validated_proteins_rows.append(
            {
                "Insight_ID": "S5",
                "Gene_Symbol": row.Gene_Symbol,
                "Metric": "Composite_Score",
                "V2_Value": row.Composite_Score,
                "V1_Value": float(
                    top10_v1.loc[
                        top10_v1["Gene_Symbol"] == row.Gene_Symbol, "Composite_Score"
                    ].mean()
                ),
            }
        )

    results_df = pd.DataFrame(results_rows)
    validated_df = pd.DataFrame(validated_proteins_rows)
    discoveries_df = pd.DataFrame(new_discoveries_rows)

    results_path = os.path.join(OUTPUT_DIR, "validation_results_codex.csv")
    validated_path = os.path.join(OUTPUT_DIR, "v2_validated_proteins_codex.csv")
    discoveries_path = os.path.join(OUTPUT_DIR, "new_discoveries_codex.csv")

    results_df.to_csv(results_path, index=False)
    validated_df.to_csv(validated_path, index=False)
    discoveries_df.to_csv(discoveries_path, index=False)

    print(f"Saved results to {results_path}")
    print(f"Saved validated proteins to {validated_path}")
    print(f"Saved discoveries to {discoveries_path}")


if __name__ == "__main__":
    main()
