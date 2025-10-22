#!/usr/bin/env python3
"""Fetch ADMETlab 2.0 predictions for TM5441 and SK-216."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "data"
ADMET_CSV = DATA_DIR / "drug_admet_codex.csv"
ADMET_JSON = DATA_DIR / "drug_admet_summary_codex.json"

LIGAND_CIDS = {
    "TM5441": "44250349",
    "SK-216": "23624303",
}

ADMETLAB_URL = "https://admetmesh.scbdd.com/service/evaluation/index"
SUBMIT_URL = "https://admetmesh.scbdd.com/service/evaluation/cal"
SCORE_MAP = {
    "+++": "High",
    "++": "Moderate",
    "+": "Slight",
    "---": "Very Low",
    "--": "Low",
    "-": "Negative",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CodexBot/1.0)",
    "Referer": ADMETLAB_URL,
}


def smiles_from_pubchem(cid: str) -> str:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    props = resp.json()["PropertyTable"]["Properties"][0]
    if "IsomericSMILES" in props:
        return props["IsomericSMILES"]
    if "SMILES" in props:
        return props["SMILES"]
    raise KeyError("SMILES not found in PubChem response")


def fetch_admet(smiles: str) -> List[pd.DataFrame]:
    session = requests.Session()
    session.headers.update(HEADERS)
    resp = session.get(ADMETLAB_URL, verify=False, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    token_tag = soup.find("input", {"name": "csrfmiddlewaretoken"})
    if not token_tag:
        raise RuntimeError("Failed to retrieve CSRF token from ADMETlab")
    token = token_tag["value"]

    post_resp = session.post(
        SUBMIT_URL,
        data={
            "csrfmiddlewaretoken": token,
            "smiles": smiles,
            "method": "1",
        },
        headers=HEADERS,
        verify=False,
        timeout=60,
    )
    post_resp.raise_for_status()
    soup = BeautifulSoup(post_resp.text, "html.parser")
    tables = soup.find_all("table")
    frames = []
    for table in tables:
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(cells)
        if rows:
            frames.append(pd.DataFrame(rows, columns=["Descriptor", "Value", "Extra", "Extra2"]))
    if not frames:
        raise RuntimeError("ADMETlab returned no data tables")
    return frames


def normalize_score(value: str) -> str:
    value = value.strip()
    return SCORE_MAP.get(value, value)


def extract_metrics(frames: List[pd.DataFrame]) -> Dict[str, str]:
    metrics = {}
    for frame in frames:
        for _, row in frame.iterrows():
            descriptor_raw = row["Descriptor"]
            value_raw = row["Value"]
            descriptor = descriptor_raw.strip() if isinstance(descriptor_raw, str) else str(descriptor_raw)
            if isinstance(value_raw, str):
                value = value_raw.strip()
            elif pd.isna(value_raw):
                value = ""
            else:
                value = str(value_raw)
            if descriptor in {
                "Molecular Weight (MW)",
                "TPSA",
                "logP",
                "logS",
                "nRot",
                "HIA",
                "F20%",
                "F30%",
                "PPB",
                "VD",
                "Fu",
                "CYP2D6 inhibitor",
                "CYP3A4 inhibitor",
                "CYP3A4 substrate",
                "CL",
                "T1/2",
                "hERG Blockers",
                "H-HT",
                "DILI",
                "AMES Toxicity",
                "Rat Oral Acute Toxicity",
            }:
                metrics[descriptor] = normalize_score(value)
    return metrics


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    summaries = {}
    for name, cid in LIGAND_CIDS.items():
        smiles = smiles_from_pubchem(cid)
        frames = fetch_admet(smiles)
        metrics = extract_metrics(frames)
        metrics.update({
            "Ligand": name,
            "CID": cid,
            "SMILES": smiles,
        })
        records.append(metrics)
        summaries[name] = metrics
        time.sleep(2)  # be polite

    df = pd.DataFrame(records)
    df.to_csv(ADMET_CSV, index=False)
    ADMET_JSON.write_text(json.dumps(summaries, indent=2))
    print(df)


if __name__ == "__main__":
    requests.packages.urllib3.disable_warnings()
    main()
