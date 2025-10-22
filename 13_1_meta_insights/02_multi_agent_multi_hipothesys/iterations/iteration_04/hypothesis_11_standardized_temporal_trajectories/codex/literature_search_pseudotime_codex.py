"""Automated literature search for pseudo-time trajectory inference best practices."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

import requests

QUERIES = [
    "pseudo-time construction aging proteomics",
    "trajectory inference methods comparison",
    "temporal modeling best practices omics",
    "diffusion maps aging",
    "Slingshot trajectory inference validation",
    "longitudinal aging proteomics",
]

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,venue,url,authors,citationCount"
LIMIT = 5
OUTPUT_CSV = Path("literature_recommendations_codex.csv")


def run_query(query: str) -> List[dict]:
    params = {
        "query": query,
        "limit": LIMIT,
        "fields": FIELDS,
    }
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    records = response.json().get("data", [])
    cleaned = []
    for record in records:
        authors = ", ".join(a.get("name", "") for a in record.get("authors", [])[:4])
        cleaned.append(
            {
                "query": query,
                "title": record.get("title", ""),
                "year": record.get("year", ""),
                "venue": record.get("venue", ""),
                "citationCount": record.get("citationCount", 0),
                "url": record.get("url", ""),
                "lead_authors": authors,
            }
        )
    return cleaned


def main() -> None:
    all_rows: List[dict] = []
    for query in QUERIES:
        try:
            all_rows.extend(run_query(query))
        except requests.HTTPError as exc:
            print(f"HTTP error for query '{query}': {exc}")
        except requests.RequestException as exc:
            print(f"Request exception for query '{query}': {exc}")
    if all_rows:
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved search results to {OUTPUT_CSV}")
    else:
        print("No records retrieved.")


if __name__ == "__main__":
    main()
