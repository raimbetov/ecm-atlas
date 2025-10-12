import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "05_Randles_paper_to_csv" / "claude_code" / "Randles_2021_wide_format.csv"
OUTPUT_DIR = ROOT / "06_Randles_z_score_by_tissue_compartment" / "codex_cli"

COMPARTMENT_COL = "Tissue_Compartment"
YOUNG_COL = "Abundance_Young"
OLD_COL = "Abundance_Old"
GENE_COL = "Gene_Symbol"
MARKERS = {"COL1A1", "COL1A2", "FN1"}


def compute_zscore(series: pd.Series) -> tuple[pd.Series, float, float]:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0:
        raise ValueError("Standard deviation is zero; z-score undefined.")
    z = (series - mean) / std
    return z, float(mean), float(std)


def process_compartment(df: pd.DataFrame, compartment: str) -> tuple[pd.DataFrame, dict]:
    comp_df = df[df[COMPARTMENT_COL] == compartment].copy()
    if comp_df.empty:
        raise ValueError(f"No rows found for compartment {compartment}.")

    stats: dict[str, dict[str, float]] = {}

    for col in (YOUNG_COL, OLD_COL):
        raw_skew = float(comp_df[col].skew())
        raw_mean = float(comp_df[col].mean())
        raw_std = float(comp_df[col].std(ddof=0))
        stats[col] = {
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "raw_skew": raw_skew,
        }

    needs_log = any(abs(metrics["raw_skew"]) > 1.0 for metrics in stats.values())

    if needs_log:
        comp_df[f"{YOUNG_COL}_log2"] = np.log2(comp_df[YOUNG_COL] + 1.0)
        comp_df[f"{OLD_COL}_log2"] = np.log2(comp_df[OLD_COL] + 1.0)
        young_basis = comp_df[f"{YOUNG_COL}_log2"]
        old_basis = comp_df[f"{OLD_COL}_log2"]
    else:
        young_basis = comp_df[YOUNG_COL]
        old_basis = comp_df[OLD_COL]

    z_young, mean_young, std_young = compute_zscore(young_basis)
    z_old, mean_old, std_old = compute_zscore(old_basis)

    comp_df["Zscore_Young"] = z_young
    comp_df["Zscore_Old"] = z_old
    comp_df["Zscore_Delta"] = comp_df["Zscore_Old"] - comp_df["Zscore_Young"]

    zstats = {
        "mean_young_basis": mean_young,
        "std_young_basis": std_young,
        "mean_old_basis": mean_old,
        "std_old_basis": std_old,
        "zscore_mean_young": float(comp_df["Zscore_Young"].mean()),
        "zscore_std_young": float(comp_df["Zscore_Young"].std(ddof=0)),
        "zscore_mean_old": float(comp_df["Zscore_Old"].mean()),
        "zscore_std_old": float(comp_df["Zscore_Old"].std(ddof=0)),
        "zscore_skew_young": float(comp_df["Zscore_Young"].skew()),
        "zscore_skew_old": float(comp_df["Zscore_Old"].skew()),
        "outlier_count_abs_gt_3": int((comp_df["Zscore_Young"].abs() > 3).sum() + (comp_df["Zscore_Old"].abs() > 3).sum()),
    }

    marker_presence = {marker: bool((comp_df[GENE_COL] == marker).any()) for marker in MARKERS}

    summary = {
        "compartment": compartment,
        "row_count": int(len(comp_df)),
        "needs_log2": needs_log,
        "stats": stats,
        "zscore": zstats,
        "marker_presence": marker_presence,
    }

    return comp_df, summary


def build_validation_report(summaries: list[dict]) -> str:
    lines = []
    lines.append("# Z-score Normalization Validation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    for summary in summaries:
        lines.append(f"## {summary['compartment']}")
        lines.append("")
        lines.append(f"- Rows: {summary['row_count']}")
        lines.append(f"- Log2 transform applied: {'yes' if summary['needs_log2'] else 'no'}")
        for col in (YOUNG_COL, OLD_COL):
            stats = summary["stats"][col]
            lines.append(
                f"- {col}: mean={stats['raw_mean']:.3f}, std={stats['raw_std']:.3f}, skew={stats['raw_skew']:.3f}"
            )
        z = summary["zscore"]
        lines.append(
            "- Zscore young mean {:.3f}, std {:.3f}, skew {:.3f}".format(
                z["zscore_mean_young"], z["zscore_std_young"], z["zscore_skew_young"]
            )
        )
        lines.append(
            "- Zscore old mean {:.3f}, std {:.3f}, skew {:.3f}".format(
                z["zscore_mean_old"], z["zscore_std_old"], z["zscore_skew_old"]
            )
        )
        total_values = summary['row_count'] * 2
        outlier_pct = summary['zscore']['outlier_count_abs_gt_3'] / total_values if total_values else 0.0
        lines.append(
            f"- |Z| > 3 count: {summary['zscore']['outlier_count_abs_gt_3']} ({outlier_pct:.2%} of values)"
        )
        marker_presence = summary["marker_presence"]
        present_markers = [m for m, present in marker_presence.items() if present]
        missing_markers = [m for m, present in marker_presence.items() if not present]
        lines.append(f"- Markers present: {', '.join(present_markers) if present_markers else 'none'}")
        if missing_markers:
            lines.append(f"- Markers missing: {', '.join(missing_markers)}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {COMPARTMENT_COL, YOUNG_COL, OLD_COL, GENE_COL}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    outputs = {
        "Glomerular": OUTPUT_DIR / "Randles_2021_Glomerular_zscore.csv",
        "Tubulointerstitial": OUTPUT_DIR / "Randles_2021_Tubulointerstitial_zscore.csv",
    }

    for compartment, output_path in outputs.items():
        comp_df, summary = process_compartment(df, compartment)
        summaries.append(summary)
        comp_df.to_csv(output_path, index=False)

    metadata = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "compartments": {summary["compartment"]: summary for summary in summaries},
        "success_criteria": {
            "zscore_mean_target": 0.0,
            "zscore_std_target": 1.0,
            "tolerance": 0.01,
        },
    }

    metadata_path = OUTPUT_DIR / "zscore_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report = build_validation_report(summaries)
    report_path = OUTPUT_DIR / "zscore_validation_report.md"
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
