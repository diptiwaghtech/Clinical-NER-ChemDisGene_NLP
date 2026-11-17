"""Download, parse and prepare ChemDisGene PubTator annotations."""
import gzip
import json
import random
import re
from pathlib import Path

import requests

from .config import (
    RAW_DIR, RAW_FILE, PROCESSED_DIR, PUBTATOR_URL, SEED,
    TRAIN_SIZE, VALID_SIZE, TOKEN_PATTERN, ENTITY_TYPES
)

TOKEN_RE = re.compile(TOKEN_PATTERN)
TYPE_MAP = {
    "chemical": "CHEMICAL",
    "chemical_substance": "CHEMICAL",
    "disease": "DISEASE",
    "disease_or_phenotypic_feature": "DISEASE",
    "gene": "GENE",
    "gene_or_gene_product": "GENE",
    "gene_product": "GENE",
}


def download_dataset(force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        return RAW_FILE
    response = requests.get(PUBTATOR_URL, timeout=120)
    response.raise_for_status()
    RAW_FILE.write_bytes(response.content)
    # Verify gzip integrity early.
    with gzip.open(RAW_FILE, "rt", encoding="utf-8") as handle:
        handle.read(100)
    return RAW_FILE


def parse_pubtator(path):
    """Parse PubTator documents and annotations."""
    opener = gzip.open if str(path).endswith(".gz") else open
    documents = []
    current = {}

    def finish():
        if not current:
            return
        title = current.get("title", "")
        abstract = current.get("abstract", "")
        text = title + " " + abstract if title and abstract else title or abstract
        entities = []
        for ann in current.get("annotations", []):
            start, end, mention, raw_type = ann
            label = TYPE_MAP.get(raw_type.lower().replace(" ", "_"))
            if label not in ENTITY_TYPES:
                continue
            # Keep only spans that point to the expected mention in joined text.
            if 0 <= start < end <= len(text) and text[start:end] == mention:
                entities.append({
                    "start": start, "end": end,
                    "text": mention, "label": label
                })
            
        documents.append({
            "doc_id": current.get("pmid", str(len(documents))),
            "text": text,
            "entities": entities,
        })

    with opener(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            
            if not line.strip():
                finish()
                current = {}
                continue
            if "|" in line:
                parts = line.split("|", 2)

                if len(parts) == 3 and parts[1] in ("t", "a"):
                    current.setdefault("pmid", parts[0])

                    if parts[1] == "t":
                        current["title"] = parts[2]
                    else:
                        current["abstract"] = parts[2]

                    continue
            
            parts = line.split("\t")
            
            if len(parts) >= 5:
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                    mention = parts[3]
                    entity_type = parts[4]

                    current.setdefault("pmid", parts[0])
                    current.setdefault("annotations", []).append(
                        (start, end, mention, entity_type)
                    )
                except (ValueError, IndexError):
                    continue
        finish()
    return documents


def tokenize_with_offsets(text):
    return [
        {"text": m.group(), "start": m.start(), "end": m.end()}
        for m in TOKEN_RE.finditer(text)
    ]


def document_to_sequences(document):
    tokens = tokenize_with_offsets(document["text"])
    tags = ["O"] * len(tokens)
    # Prefer longer spans if an unusual overlap occurs.
    for entity in sorted(document["entities"],
                         key=lambda e: (e["start"], -(e["end"] - e["start"]))):
        indexes = [
            i for i, token in enumerate(tokens)
            if token["start"] < entity["end"] and token["end"] > entity["start"]
        ]
        if not indexes:
            continue
        # Exact boundaries are required; otherwise skip to avoid corrupt BIO tags.
        if (tokens[indexes[0]]["start"] != entity["start"]
                or tokens[indexes[-1]]["end"] != entity["end"]):
            continue
        if any(tags[i] != "O" for i in indexes):
            continue
        tags[indexes[0]] = "B-" + entity["label"]
        for i in indexes[1:]:
            tags[i] = "I-" + entity["label"]
    return tokens, tags


def split_documents(documents):
    """Deterministic document-level 70/15/15 split."""
    docs = list(documents)
    random.Random(SEED).shuffle(docs)
    n = len(docs)
    n_train = int(n * TRAIN_SIZE)
    n_valid = int(n * VALID_SIZE)
    return {
        "train": docs[:n_train],
        "validation": docs[n_train:n_train + n_valid],
        "test": docs[n_train + n_valid:],
    }


def save_splits(splits):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split_name, docs in splits.items():
        out = PROCESSED_DIR / f"{split_name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        manifest[split_name] = [d["doc_id"] for d in docs]
    (PROCESSED_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def load_split(name):
    path = PROCESSED_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError("Splits not found. Run: python -m src.train")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_data(force_download=False):
    path = download_dataset(force=force_download)
    docs = parse_pubtator(path)
    if not docs:
        raise RuntimeError("No documents were parsed. Check the upstream PubTator file format.")
    # Exclude empty text documents and retain only records with tokenizable text.
    docs = [d for d in docs if d["text"].strip()]
    splits = split_documents(docs)
    save_splits(splits)
    print(f"Parsed {len(docs)} documents.")
    print("Split sizes:", {k: len(v) for k, v in splits.items()})
    print("Aligned entity mentions:", sum(len(d["entities"]) for d in docs))
    return splits
