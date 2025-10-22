"""Process external GEO datasets to extract calcium signaling genes."""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, List

import GEOparse
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "external_datasets"
OUTPUT_PATH = DATA_DIR / "GSE11475_expression_codex.csv"
TARGET_GENES = {
    "S100A8",
    "S100A9",
    "S100A10",
    "S100B",
    "CALM1",
    "CALM2",
    "CALM3",
    "CAMK1",
    "CAMK2A",
    "CAMK2B",
    "CAMK2D",
    "CAMK2G",
    "LOX",
    "LOXL2",
    "LOXL3",
    "LOXL4",
    "TGM1",
    "TGM2",
    "TGM3",
}


def load_gse11475() -> GEOparse.GEO.GSE:  # type: ignore
    soft_path = DATA_DIR / "GSE11475_family.soft.gz"
    if not soft_path.exists():
        raise FileNotFoundError("Soft file for GSE11475 not found")
    gse = GEOparse.get_GEO(filepath=str(soft_path), how="full")
    return gse


def build_probe_mapping(gse: GEOparse.GEO.GSE) -> Dict[str, str]:  # type: ignore
    mapping: Dict[str, str] = {}
    for gpl in gse.gpls.values():
        table = gpl.table
        if "Gene Symbol" in table.columns:
            for _, row in table.iterrows():
                gene = str(row.get("Gene Symbol", ""))
                if not gene:
                    continue
                for symbol in gene.split(" /// "):
                    mapping.setdefault(str(row["ID"]), symbol.strip().upper())
        elif "GENE" in table.columns:
            for _, row in table.iterrows():
                mapping.setdefault(str(row["ID"]), str(row["GENE"]).upper())
    return mapping


def extract_expression(gse: GEOparse.GEO.GSE, mapping: Dict[str, str]) -> pd.DataFrame:  # type: ignore
    expr: Dict[str, List[float]] = {gene: [] for gene in TARGET_GENES}
    sample_ids: List[str] = []
    for gsm_name, gsm in gse.gsms.items():
        sample_ids.append(gsm_name)
        table = gsm.table
        table["Symbol"] = table["ID_REF"].map(mapping)
        subset = table.dropna(subset=["Symbol"])
        sub = subset[subset["Symbol"].isin(TARGET_GENES)]
        grouped = sub.groupby("Symbol")["VALUE"].mean()
        for gene in TARGET_GENES:
            expr[gene].append(float(grouped.get(gene, float("nan"))))
    df = pd.DataFrame(expr, index=sample_ids)
    return df


def main() -> None:
    gse = load_gse11475()
    mapping = build_probe_mapping(gse)
    df = extract_expression(gse, mapping)
    df.sort_index(inplace=True)
    df.to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    main()
