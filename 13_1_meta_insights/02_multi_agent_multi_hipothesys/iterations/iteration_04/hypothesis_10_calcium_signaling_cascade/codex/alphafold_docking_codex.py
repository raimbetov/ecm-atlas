"""Coarse docking of S100-calmodulin complexes using AlphaFold monomers."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from Bio import PDB
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

WORKSPACE = Path(__file__).resolve().parent
STRUCTURE_DIR = WORKSPACE / "alphafold_structures"
VIS_DIR = WORKSPACE / "visualizations_codex"
RESULT_PATH = WORKSPACE / "structural_binding_codex.csv"

CUSTOM_PDB = {
    "P62158": "https://files.rcsb.org/download/1CLL.pdb",
    "P0DP23": "https://files.rcsb.org/download/1CLL.pdb",
}

PAIRINGS = [
    ("S100A10", "P60903", "CALM1", "P62158"),
    ("S100B", "P04271", "CALM1", "P62158"),
    ("S100A9", "P06702", "CALM2", "P0DP23"),
]

N_SAMPLES = 500
CONTACT_THRESHOLD = 4.5
CLASH_THRESHOLD = 2.0


@dataclass
class DockingResult:
    complex_id: str
    contacts: int
    clashes: int
    score: float
    translation: Tuple[float, float, float]
    interface_s100: List[str]
    interface_calm: List[str]
    pdb_path: Path

    def to_dict(self) -> Dict[str, object]:
        return {
            "complex": self.complex_id,
            "contacts": self.contacts,
            "clashes": self.clashes,
            "score": self.score,
            "translation": list(self.translation),
            "interface_s100": ";".join(self.interface_s100),
            "interface_calm": ";".join(self.interface_calm),
            "pdb_path": str(self.pdb_path),
        }


def ensure_directories() -> None:
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)


def download_structure(uniprot_id: str) -> Path:
    output_path = STRUCTURE_DIR / f"AF-{uniprot_id}-model_v4.pdb"
    if output_path.exists():
        return output_path
    if uniprot_id in CUSTOM_PDB:
        pdb_url = CUSTOM_PDB[uniprot_id]
        response = requests.get(pdb_url, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return output_path

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    for attempt in range(3):
        response = requests.get(api_url, timeout=60)
        if response.status_code == 200:
            records = response.json()
            break
        if attempt == 2:
            response.raise_for_status()
    else:  # pragma: no cover
        raise RuntimeError(f"No AlphaFold metadata for {uniprot_id}")
    if not records:
        raise RuntimeError(f"No AlphaFold metadata for {uniprot_id}")
    pdb_url = records[0].get("pdbUrl")
    if not pdb_url:
        raise RuntimeError(f"Missing pdbUrl for {uniprot_id}")
    for attempt in range(3):
        pdb_response = requests.get(pdb_url, timeout=120)
        if pdb_response.status_code == 200:
            output_path.write_bytes(pdb_response.content)
            return output_path
        if attempt == 2:
            pdb_response.raise_for_status()
    raise RuntimeError(f"Failed to download PDB for {uniprot_id}")


def structure_to_coordinates(path: Path) -> Tuple[np.ndarray, List[PDB.Atom.Atom], List[str], PDB.Structure.Structure]:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, path)
    atoms: List[PDB.Atom.Atom] = [atom for atom in structure.get_atoms() if atom.element != "H"]
    coords = np.array([atom.coord for atom in atoms], dtype=float)
    residues: List[str] = []
    for atom in atoms:
        residue = atom.get_parent()
        res_id = residue.get_id()[1]
        chain_id = residue.get_parent().id
        residues.append(f"{chain_id}-{residue.get_resname()}-{res_id}")
    return coords, atoms, residues, structure


def random_rotation_matrix() -> np.ndarray:
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, 2 * math.pi)
    z = random.uniform(0, 2 * math.pi)
    Rz = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(phi), 0, math.sin(phi)], [0, 1, 0], [-math.sin(phi), 0, math.cos(phi)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(z), -math.sin(z)], [0, math.sin(z), math.cos(z)]])
    return Rz @ Ry @ Rx


def apply_transform(atoms: List[PDB.Atom.Atom], coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> None:
    transformed = coords @ rotation.T + translation
    for atom, new_coord in zip(atoms, transformed):
        atom.set_coord(new_coord)


def compute_interface(residues: List[str], distances: np.ndarray, threshold: float) -> List[str]:
    idx = np.where(distances.min(axis=1) <= threshold)[0]
    return sorted({residues[i] for i in idx})


def dock_pair(s_name: str, s_path: Path, c_name: str, c_path: Path) -> DockingResult:
    s_coords, s_atoms, s_residues, s_structure = structure_to_coordinates(s_path)
    c_coords, c_atoms, c_residues, c_structure = structure_to_coordinates(c_path)

    s_coords_centered = s_coords - s_coords.mean(axis=0)
    c_coords_centered = c_coords - c_coords.mean(axis=0)

    best_score = -float("inf")
    best_state = None
    random.seed(42)

    for _ in range(N_SAMPLES):
        rotation = random_rotation_matrix()
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-8
        distance = random.uniform(10.0, 30.0)
        translation = direction * distance

        transformed = c_coords_centered @ rotation.T + translation
        d_matrix = cdist(s_coords_centered, transformed)
        contacts = int(np.sum(d_matrix <= CONTACT_THRESHOLD))
        clashes = int(np.sum(d_matrix <= CLASH_THRESHOLD))
        score = contacts - 3 * clashes
        if score > best_score:
            best_score = score
            best_state = (rotation, translation, d_matrix)

    assert best_state is not None
    rotation, translation, d_matrix = best_state
    s_structure_copy = copy.deepcopy(s_structure)
    c_structure_copy = copy.deepcopy(c_structure)
    apply_transform(list(s_structure_copy.get_atoms()), s_coords_centered, np.eye(3), np.zeros(3))
    apply_transform(list(c_structure_copy.get_atoms()), c_coords_centered, rotation, translation)

    combined_structure = PDB.Structure.Structure(f"{s_name}_{c_name}")
    model = PDB.Model.Model(0)
    combined_structure.add(model)
    chain_a = PDB.Chain.Chain("A")
    chain_b = PDB.Chain.Chain("B")
    model.add(chain_a)
    model.add(chain_b)
    for residue in s_structure_copy.get_residues():
        chain_a.add(residue.copy())
    for residue in c_structure_copy.get_residues():
        chain_b.add(residue.copy())

    io = PDB.PDBIO()
    io.set_structure(combined_structure)
    out_path = STRUCTURE_DIR / f"{s_name}_{c_name}_complex_codex.pdb"
    io.save(str(out_path))

    interface_s100 = compute_interface(s_residues, d_matrix, threshold=5.0)
    interface_calm = compute_interface(c_residues, d_matrix.T, threshold=5.0)

    contacts = int(np.sum(d_matrix <= CONTACT_THRESHOLD))
    clashes = int(np.sum(d_matrix <= CLASH_THRESHOLD))
    return DockingResult(
        complex_id=f"{s_name}-{c_name}",
        contacts=contacts,
        clashes=clashes,
        score=best_score,
        translation=tuple(float(x) for x in translation),
        interface_s100=interface_s100,
        interface_calm=interface_calm,
        pdb_path=out_path,
    )


def main() -> None:
    ensure_directories()
    records: List[DockingResult] = []
    for s_name, s_uniprot, c_name, c_uniprot in PAIRINGS:
        s_path = download_structure(s_uniprot)
        c_path = download_structure(c_uniprot)
        result = dock_pair(s_name, s_path, c_name, c_path)
        records.append(result)

    df = pd.DataFrame([r.to_dict() for r in records])
    df.to_csv(RESULT_PATH, index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(df["complex"], df["contacts"], color="#4c72b0", label="contacts")
    plt.bar(df["complex"], df["clashes"], color="#dd8452", alpha=0.7, label="clashes")
    plt.ylabel("Count")
    plt.title("Coarse docking contact vs clash counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIS_DIR / "alphafold_contacts_codex.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
