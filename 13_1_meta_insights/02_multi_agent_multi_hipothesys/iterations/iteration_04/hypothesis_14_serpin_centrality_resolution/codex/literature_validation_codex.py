#!/usr/bin/env python3
"""Search literature for network centrality best practices and export curated notes."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List

import requests

AGENT = "codex"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "literature_network_centrality.md"

QUERIES = [
    "network centrality metrics comparison proteomics",
    "betweenness vs eigenvector centrality biological networks",
    "PageRank protein interaction networks",
    "centrality lethality rule validation",
    "serpin protease inhibitor networks aging",
    "graph theory systems biology best practices",
    "knockout experiments network topology",
]

FIELDS = "title,year,authors,venue,url,abstract"
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
MAX_RESULTS = 5


def fetch_papers(query: str) -> List[Dict[str, str]]:
    params = {"query": query, "fields": FIELDS, "limit": MAX_RESULTS}
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])


def format_entry(paper: Dict[str, str]) -> str:
    title = paper.get("title", "Unknown title")
    year = paper.get("year")
    url = paper.get("url")
    venue = paper.get("venue")
    authors = paper.get("authors", [])
    author_names = ", ".join(author.get("name", "?") for author in authors[:5])
    if len(authors) > 5:
        author_names += ", et al."
    abstract = paper.get("abstract") or "Abstract unavailable."
    abstract = " ".join(abstract.strip().split())
    abstract_snippet = textwrap.shorten(abstract, width=420, placeholder="…")
    lines = [f"- **{title}** ({year}, {venue})", f"  - Authors: {author_names}", f"  - Link: {url}", f"  - Key Insight: {abstract_snippet}"]
    return "\n".join(lines)


def build_report() -> None:
    sections = ["# Network Centrality Literature Synthesis\n"]
    for query in QUERIES:
        sections.append(f"## Query: {query}\n")
        try:
            papers = fetch_papers(query)
        except requests.HTTPError as exc:
            sections.append(f"- Error fetching results: {exc}\n")
            continue
        if not papers:
            sections.append("- No results returned.\n")
            continue
        for paper in papers:
            sections.append(format_entry(paper) + "\n")
    OUTPUT_PATH.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    build_report()
