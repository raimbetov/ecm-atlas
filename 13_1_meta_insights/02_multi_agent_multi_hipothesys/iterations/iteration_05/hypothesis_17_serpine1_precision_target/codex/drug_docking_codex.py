#!/usr/bin/env python3
"""Dock TM5441 and SK-216 into AlphaFold SERPINE1 (Q05682)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests
from vina import Vina

WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "data"
STRUCT_DIR = DATA_DIR / "structures"
DOCK_DIR = DATA_DIR / "docking"
VIS_DIR = WORKSPACE / "visualizations_codex"
UNIPROT_ID = "P05121"  # SERPINE1
RECEPTOR_PDB = STRUCT_DIR / "serpine1_alphafold.pdb"
RECEPTOR_PDBQT = STRUCT_DIR / "serpine1_alphafold.pdbqt"
RESULTS_CSV = DOCK_DIR / "docking_results_codex.csv"
SUMMARY_JSON = DATA_DIR / "docking_summary_codex.json"
SCORES_PLOT = VIS_DIR / "docking_scores_codex.png"

LIGANDS = {
    "TM5441": "44250349",
    "SK-216": "23624303",
}

OBABEL_BIN = WORKSPACE / ".venv" / "bin" / "obabel"


def ensure_dirs() -> None:
    for path in [DATA_DIR, STRUCT_DIR, DOCK_DIR, VIS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_alphafold() -> None:
    if RECEPTOR_PDB.exists():
        return
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{UNIPROT_ID}"
    for attempt in range(5):
        response = requests.get(api_url, timeout=60)
        if response.status_code == 200:
            break
        if attempt == 4:
            response.raise_for_status()
        time.sleep(2 * (attempt + 1))
    entries = response.json()
    pdb_url = None
    for entry in entries:
        if entry.get("uniprotAccession") == UNIPROT_ID and entry.get("pdbUrl"):
            pdb_url = entry["pdbUrl"]
            break
    if not pdb_url and entries:
        pdb_url = entries[0].get("pdbUrl")
    if not pdb_url:
        raise RuntimeError("AlphaFold API did not return a PDB URL for SERPINE1")
    for attempt in range(5):
        pdb_response = requests.get(pdb_url, timeout=60)
        if pdb_response.status_code == 200:
            break
        if attempt == 4:
            pdb_response.raise_for_status()
        time.sleep(2 * (attempt + 1))
    RECEPTOR_PDB.write_bytes(pdb_response.content)


def smiles_from_pubchem(cid: str) -> str:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    data = res.json()
    props = data["PropertyTable"]["Properties"][0]
    if "IsomericSMILES" in props:
        return props["IsomericSMILES"]
    if "SMILES" in props:
        return props["SMILES"]
    raise KeyError("PubChem response lacked SMILES data")


def prepare_ligand(name: str, cid: str) -> Dict[str, Path]:
    smiles = smiles_from_pubchem(cid)
    sdf_path = STRUCT_DIR / f"{name}.sdf"
    pdbqt_path = STRUCT_DIR / f"{name}.pdbqt"

    # Generate 3D conformer and export SDF
    subprocess.run(
        [
            str(OBABEL_BIN),
            f"-:{smiles}",
            "--gen3D",
            "-O",
            str(sdf_path),
        ],
        check=True,
    )

    # Convert to PDBQT with Gasteiger charges
    subprocess.run(
        [
            str(OBABEL_BIN),
            str(sdf_path),
            "-O",
            str(pdbqt_path),
            "--partialcharge",
            "gasteiger",
        ],
        check=True,
    )

    return {"sdf": sdf_path, "pdbqt": pdbqt_path}


def prepare_receptor() -> None:
    if RECEPTOR_PDBQT.exists():
        return
    subprocess.run(
        [
            str(OBABEL_BIN),
            str(RECEPTOR_PDB),
            "-O",
            str(RECEPTOR_PDBQT),
            "-p",
            "7.4",
            "--partialcharge",
            "gasteiger",
            "-xr",
        ],
        check=True,
    )


def compute_grid(box_padding: float = 6.0) -> Dict[str, np.ndarray]:
    xs, ys, zs = [], [], []
    with open(RECEPTOR_PDB) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
    coords = np.array([xs, ys, zs])
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1)
    center = (mins + maxs) / 2
    size = (maxs - mins) + box_padding
    size = np.clip(size, 20.0, 50.0)  # ensure manageable search box
    return {"center": center, "size": size}


def dock_ligand(ligand_name: str, ligand_pdbqt: Path, grid: Dict[str, np.ndarray]) -> Dict[str, object]:
    vina = Vina(sf_name="vina")
    vina.set_receptor(str(RECEPTOR_PDBQT))
    vina.set_ligand_from_file(str(ligand_pdbqt))
    vina.compute_vina_maps(center=grid["center"].tolist(), box_size=grid["size"].tolist())
    vina.dock(exhaustiveness=24, n_poses=10)
    scores = vina.energies(n_poses=10)[:, 0]
    out_pdbqt = DOCK_DIR / f"{ligand_name}_poses.pdbqt"
    vina.write_poses(str(out_pdbqt), n_poses=10, overwrite=True)
    return {
        "Ligand": ligand_name,
        "Best_Affinity_kcal_mol": float(scores.min()),
        "Mean_Affinity_kcal_mol": float(scores.mean()),
        "Poses_File": str(out_pdbqt),
    }


def plot_scores(df: pd.DataFrame) -> None:
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])  # Lazy import
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["Ligand"], df["Best_Affinity_kcal_mol"], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Best binding energy (kcal/mol)")
    ax.set_title("SERPINE1 docking scores")
    ax.axhline(-7.0, color="red", linestyle="--", linewidth=0.8, label="Good affinity threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SCORES_PLOT, dpi=300)
    plt.close()


def main() -> None:
    ensure_dirs()
    download_alphafold()
    prepare_receptor()
    grid = compute_grid()

    results = []
    for name, cid in LIGANDS.items():
        lig_paths = prepare_ligand(name, cid)
        result = dock_ligand(name, lig_paths["pdbqt"], grid)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    plot_scores(df)
    summary = {
        "grid_center": grid["center"].tolist(),
        "grid_size": grid["size"].tolist(),
        "results": df.to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
