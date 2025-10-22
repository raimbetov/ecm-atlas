"""Automated literature and dataset mining for calcium signaling cascade."""
from __future__ import annotations

import json
import math
import os
import re
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from Bio import Entrez

WORKSPACE = Path(__file__).resolve().parent
LITERATURE_CSV = WORKSPACE / "literature_findings_codex.csv"
LITERATURE_MD = WORKSPACE / "literature_review.md"
DATASET_CSV = WORKSPACE / "new_datasets_codex.csv"
EXTERNAL_DATA_DIR = WORKSPACE / "external_datasets"

EMAIL = os.environ.get("PUBMED_EMAIL", "codex.agent@openai.com")
Entrez.email = EMAIL
Entrez.tool = "codex_calcium_pipeline"

PUBMED_QUERIES = [
    "S100 calcium signaling aging",
    "calmodulin CAMK extracellular matrix",
    "LOX transglutaminase calcium regulation",
    "S100 calmodulin binding",
    "S100A1 calmodulin binding",
    "CAMK2 fibrosis collagen",
]

GEO_QUERIES = [
    "calcium signaling aging",
    "calmodulin aging",
    "CAMK2 fibrosis",
    "CALM1 aging",
    "CAMK2A aging",
]

PRIDE_QUERIES = [
    "calmodulin",
    "CAMK",
]

BIO_RXIV_QUERIES = [
    "S100 calmodulin",
    "calcium signaling extracellular matrix",
]

MAX_PAPERS_PER_QUERY = 5
MAX_PREPRINTS_PER_QUERY = 5
MAX_DATASETS_PER_QUERY = 5


@dataclass
class Publication:
    source: str
    query: str
    title: str
    authors: str
    journal: str
    year: int
    doi: Optional[str]
    identifier: str
    url: str
    abstract: str
    highlights: List[str]

    def to_dict(self) -> Dict[str, object]:
        record = asdict(self)
        record["highlights"] = " ; ".join(self.highlights)
        return record


def chunked(it: Iterable[str], size: int) -> Iterable[List[str]]:
    bucket: List[str] = []
    for item in it:
        bucket.append(item)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def sanitize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def summarize_abstract(abstract: str, terms: List[str]) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    highlights: List[str] = []
    for sent in sentences:
        lowered = sent.lower()
        if any(term in lowered for term in terms):
            highlights.append(sanitize_text(sent))
        if len(highlights) >= 3:
            break
    if not highlights:
        highlights = sentences[:2]
    return [sanitize_text(h) for h in highlights if h]


def query_pubmed(term: str) -> List[Publication]:
    search = Entrez.esearch(
        db="pubmed",
        term=term,
        sort="cited",
        retmax=MAX_PAPERS_PER_QUERY,
        datetype="pdat",
        mindate="2020",
        maxdate=str(datetime.now().year),
    )
    record = Entrez.read(search)
    ids = record.get("IdList", [])
    if not ids:
        return []

    publications: List[Publication] = []
    for chunk in chunked(ids, 5):
        summary_handle = Entrez.esummary(db="pubmed", id=",".join(chunk))
        summaries = Entrez.read(summary_handle)
        fetch_handle = Entrez.efetch(
            db="pubmed", id=",".join(chunk), rettype="abstract", retmode="xml"
        )
        fetch_records = Entrez.read(fetch_handle)
        abstracts_map: Dict[str, str] = {}
        for article in fetch_records.get("PubmedArticle", []):
            pmid = article["MedlineCitation"]["PMID"]
            abstract_texts = article["MedlineCitation"].get("Article", {}).get("Abstract", {}).get("AbstractText", [])
            abstract_str = " ".join(str(t) for t in abstract_texts)
            abstracts_map[str(pmid)] = sanitize_text(abstract_str)

        for doc in summaries:
            pmid = doc.get("Id")
            title = sanitize_text(doc.get("Title", ""))
            journal = sanitize_text(doc.get("FullJournalName", doc.get("Source", "")))
            authors = ", ".join(a["Name"] for a in doc.get("Authors", [])[:5])
            pubdate = doc.get("PubDate", "")
            year_match = re.search(r"(\d{4})", pubdate)
            year = int(year_match.group(1)) if year_match else datetime.now().year
            doi = doc.get("elocationid")
            doi = doi if doi and doi.lower().startswith("10.") else doc.get("DOI")
            abstract = abstracts_map.get(pmid, "")
            highlights = summarize_abstract(abstract, ["s100", "calmod", "camk", "lox", "tgm"])

            publications.append(
                Publication(
                    source="PubMed",
                    query=term,
                    title=title,
                    authors=authors,
                    journal=journal,
                    year=year,
                    doi=doi,
                    identifier=pmid,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    abstract=abstract,
                    highlights=highlights,
                )
            )
    return publications


def query_biorxiv(term: str) -> List[Publication]:
    url = f"https://api.biorxiv.org/details/biorxiv/{requests.utils.quote(term)}/0/{MAX_PREPRINTS_PER_QUERY - 1}"
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return []
    data = response.json().get("collection", [])
    publications: List[Publication] = []
    for item in data:
        title = sanitize_text(item.get("title", ""))
        authors = sanitize_text(item.get("authors", ""))
        year = int(item.get("date", "2020")[:4])
        doi = sanitize_text(item.get("doi", "")) or None
        abstract = sanitize_text(item.get("abstract", ""))
        highlights = summarize_abstract(abstract, ["s100", "calmod", "camk", "extracellular"])
        publications.append(
            Publication(
                source="bioRxiv",
                query=term,
                title=title,
                authors=authors,
                journal="bioRxiv",
                year=year,
                doi=doi,
                identifier=doi or title[:50],
                url=f"https://www.biorxiv.org/content/{doi}v1" if doi else "",
                abstract=abstract,
                highlights=highlights,
            )
        )
    return publications


@dataclass
class DatasetMetadata:
    source: str
    query: str
    accession: str
    title: str
    organism: str
    data_type: str
    description: str
    contains_targets: str
    download_url: str
    n_samples: Optional[int]
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


REQUIRED_GENES = {
    "CALM1", "CALM2", "CALM3", "CAMK1", "CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G"
}


def geo_contains_targets(summary: Dict[str, str]) -> bool:
    text = " ".join(
        str(summary.get(field, "")) for field in ["title", "summary", "suppFileLink", "platform"]
    ).upper()
    return any(gene in text for gene in REQUIRED_GENES)


def query_geo(term: str) -> List[DatasetMetadata]:
    search = Entrez.esearch(db="gds", term=term, retmax=MAX_DATASETS_PER_QUERY, sort="relevance")
    record = Entrez.read(search)
    ids = record.get("IdList", [])
    datasets: List[DatasetMetadata] = []
    if not ids:
        return datasets
    summary_handle = Entrez.esummary(db="gds", id=",".join(ids))
    summaries = Entrez.read(summary_handle)
    for doc in summaries:
        accession = doc.get("Accession", "")
        title = sanitize_text(doc.get("title", ""))
        organism = sanitize_text(doc.get("taxon", ""))
        data_type = sanitize_text(doc.get("gdsType", ""))
        description = sanitize_text(doc.get("summary", ""))
        contains = geo_contains_targets(doc)
        link = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
        n_samples = doc.get("n_samples")
        notes = "Contains CALM/CAMK" if contains else "Needs manual gene check"
        datasets.append(
            DatasetMetadata(
                source="GEO",
                query=term,
                accession=accession,
                title=title,
                organism=organism,
                data_type=data_type,
                description=description,
                contains_targets="yes" if contains else "unknown",
                download_url=link,
                n_samples=int(n_samples) if n_samples else None,
                notes=notes,
            )
        )
    return datasets


def query_pride(term: str) -> List[DatasetMetadata]:
    params = {
        "pageSize": MAX_DATASETS_PER_QUERY,
        "page": 0,
        "q": term,
    }
    url = "https://www.ebi.ac.uk/pride/ws/archive/project/list"
    response = requests.get(url, params=params, timeout=30)
    datasets: List[DatasetMetadata] = []
    if response.status_code != 200:
        return datasets
    data = response.json().get("list", [])
    for item in data:
        accession = item.get("accession")
        title = sanitize_text(item.get("title", ""))
        organism = sanitize_text(item.get("species", [{}])[0].get("name", ""))
        description = sanitize_text(item.get("projectDescription", ""))
        data_type = sanitize_text(
            ", ".join(instrument.get("name", "") for instrument in item.get("instruments", []))
        ) or "Proteomics"
        notes = "Contains CALM/CAMK" if any(gene in description.upper() for gene in REQUIRED_GENES) else "Screen manually"
        download_url = f"https://www.ebi.ac.uk/pride/archive/projects/{accession}"
        datasets.append(
            DatasetMetadata(
                source="PRIDE",
                query=term,
                accession=accession,
                title=title,
                organism=organism,
                data_type=data_type,
                description=description,
                contains_targets="yes" if "CALM" in description.upper() or "CAMK" in description.upper() else "unknown",
                download_url=download_url,
                n_samples=None,
                notes=notes,
            )
        )
    return datasets


def download_geo_soft(accession: str, dest_dir: Path) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:len(accession)-3]}nnn/{accession}/soft/{accession}_family.soft.gz"
    response = requests.get(soft_url, stream=True, timeout=60)
    if response.status_code != 200:
        return None
    output_path = dest_dir / f"{accession}_family.soft.gz"
    with open(output_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                handle.write(chunk)
    return output_path


def build_literature_markdown(records: List[Publication]) -> str:
    lines = ["# S100→CALM/CAMK Literature Review (codex)", ""]
    grouped: Dict[str, List[Publication]] = {}
    for pub in records:
        grouped.setdefault(pub.query, []).append(pub)
    for query, pubs in grouped.items():
        lines.append(f"## Query: {query}")
        for pub in pubs:
            citation = f"{pub.authors} ({pub.year}). *{pub.title}*. {pub.journal}."
            detail = "  ".join(pub.highlights[:2])
            doi_part = f" DOI: {pub.doi}" if pub.doi else ""
            lines.append(f"- {citation}{doi_part}\n  - Source: {pub.url}\n  - Key: {detail}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    publications: List[Publication] = []
    for term in PUBMED_QUERIES:
        try:
            publications.extend(query_pubmed(term))
        except Exception as exc:  # pragma: no cover - network failures
            print(f"[WARN] PubMed query failed for '{term}': {exc}")

    for term in BIO_RXIV_QUERIES:
        try:
            publications.extend(query_biorxiv(term))
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] bioRxiv query failed for '{term}': {exc}")

    if publications:
        df = pd.DataFrame([pub.to_dict() for pub in publications])
        df.sort_values(by=["source", "query", "year"], ascending=[True, True, False], inplace=True)
        df.to_csv(LITERATURE_CSV, index=False)
        LITERATURE_MD.write_text(build_literature_markdown(publications))
    else:
        print("[WARN] No publications retrieved")

    datasets: List[DatasetMetadata] = []
    for term in GEO_QUERIES:
        try:
            datasets.extend(query_geo(term))
        except Exception as exc:
            print(f"[WARN] GEO query failed for '{term}': {exc}")
    for term in PRIDE_QUERIES:
        try:
            datasets.extend(query_pride(term))
        except Exception as exc:
            print(f"[WARN] PRIDE query failed for '{term}': {exc}")

    if datasets:
        df_datasets = pd.DataFrame([d.to_dict() for d in datasets])
        df_datasets.sort_values(by=["source", "contains_targets", "query"], ascending=[True, False, True], inplace=True)
        df_datasets.to_csv(DATASET_CSV, index=False)
    else:
        print("[WARN] No datasets retrieved")

    # Attempt to download at least one GEO dataset with CALM/CAMK reference
    downloaded = False
    for meta in datasets:
        if meta.source == "GEO" and meta.contains_targets == "yes":
            path = download_geo_soft(meta.accession, EXTERNAL_DATA_DIR)
            if path:
                print(f"Downloaded {meta.accession} to {path}")
                downloaded = True
                break
    if not downloaded:
        print("[WARN] Could not download GEO dataset meeting criteria")


if __name__ == "__main__":
    main()
