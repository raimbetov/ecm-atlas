#!/usr/bin/env python3
"""Collect ClinicalTrials.gov studies mentioning SERPINE1/PAI-1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "data"
TRIALS_CSV = DATA_DIR / "clinical_trials_codex.csv"
TRIALS_JSON = DATA_DIR / "clinical_trials_summary_codex.json"
API_URL = "https://clinicaltrials.gov/api/v2/studies"
QUERY = "SERPINE1 OR PAI-1 OR plasminogen activator inhibitor-1 OR tiplaxtinin"
PAGE_SIZE = 50


def fetch_page(token: Optional[str] = None) -> Dict:
    params = {
        "query.term": QUERY,
        "pageSize": PAGE_SIZE,
    }
    if token:
        params["pageToken"] = token
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect_trials(max_pages: int = 5) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    token: Optional[str] = None
    for _ in range(max_pages):
        data = fetch_page(token)
        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            design_module = protocol.get("designModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            arms_module = protocol.get("armsInterventionsModule", {})
            derived = study.get("derivedSection", {})

            nct_id = identification.get("nctId")
            if not nct_id:
                continue

            interventions = []
            for intervention in arms_module.get("interventions", []) or []:
                name = intervention.get("name")
                itype = intervention.get("type")
                if name:
                    interventions.append(f"{itype}: {name}" if itype else name)

            phases = design_module.get("phases") or []
            phases_str = ", ".join(phases) if isinstance(phases, list) else str(phases)

            conditions = conditions_module.get("conditions") or []
            if isinstance(conditions, list):
                condition_str = "; ".join(conditions)
            else:
                condition_str = str(conditions)

            enrollment = derived.get("derivedEnrollmentCount")

            record = {
                "NCTId": nct_id,
                "BriefTitle": identification.get("briefTitle", ""),
                "OverallStatus": status_module.get("overallStatus", ""),
                "Phase": phases_str,
                "Conditions": condition_str,
                "Interventions": "; ".join(interventions),
                "StudyType": design_module.get("studyType", ""),
                "StartDate": status_module.get("startDateStruct", {}).get("date", ""),
                "PrimaryCompletionDate": status_module.get("primaryCompletionDateStruct", {}).get("date", ""),
                "Enrollment":"" if enrollment is None else str(enrollment),
            }
            records.append(record)
        token = data.get("nextPageToken")
        if not token:
            break
    return records


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_trials()
    df = pd.DataFrame(records).drop_duplicates("NCTId")
    df.to_csv(TRIALS_CSV, index=False)

    summary = {
        "query": QUERY,
        "total_trials": int(df.shape[0]),
        "phase_breakdown": df["Phase"].value_counts().to_dict(),
        "status_breakdown": df["OverallStatus"].value_counts().to_dict(),
        "recent_trials": df.head(10).to_dict(orient="records"),
    }
    TRIALS_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
