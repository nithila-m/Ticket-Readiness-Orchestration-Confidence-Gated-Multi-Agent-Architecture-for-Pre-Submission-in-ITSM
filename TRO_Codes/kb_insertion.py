"""
insert_to_chromadb.py
======================

Loads KB articles and synthetic tickets from JSON files on disk and inserts
them into two separate ChromaDB collections:

    - "kb_articles"  -> used by Agent 3 (KB Retrieval & Deflection Gate)
    - "tickets"       -> used by Agent 4 (Correlation Engine / duplicate detection)

--------------------------------------------------------------------------
1. DEPENDENCIES — install before running
--------------------------------------------------------------------------
    pip install chromadb sentence-transformers

    (Optional, only if you plan to swap in a hosted embedding API instead
    of the local sentence-transformers model:)
    pip install openai            # if using OpenAI embeddings
    pip install chromadb-client   # if connecting to a remote Chroma server

    Python 3.10+ recommended (matches the rest of the TRO stack).

--------------------------------------------------------------------------
2. WHERE TO PUT YOUR JSON PAYLOADS (placeholders, not embedded in code)
--------------------------------------------------------------------------
Create this folder structure next to this script:

    tro_chroma_ingest/
    ├── insert_to_chromadb.py          <- this file
    ├── chroma_db/                     <- auto-created, persisted vector store
    └── data/
        ├── kb/                        <- put ALL kb_*.json files here
        │   ├── kb_ad_account.json
        │   ├── kb_printer.json
        │   ├── kb_wifi.json
        │   ├── kb_teams.json
        │   └── kb_email.json
        └── tickets/                   <- put ALL ticket_*.json files here
            ├── tickets_ad_account.json
            ├── tickets_printer.json
            ├── tickets_wifi.json
            ├── tickets_teams.json
            └── tickets_email.json

Each file must contain a JSON ARRAY of objects (exactly the JSON payloads
generated earlier in this conversation — copy/paste each array into its own
.json file). The script auto-discovers every *.json file in data/kb/ and
data/tickets/, so you can drop in as many category files as you want without
touching the code.

--------------------------------------------------------------------------
3. RUN
--------------------------------------------------------------------------
    python insert_to_chromadb.py

    Re-running is safe: documents are inserted with `upsert()` keyed on their
    own IDs (kb_id / ticket_id), so re-running after editing a JSON file
    updates existing records instead of duplicating them.
"""

import json
import glob
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# --------------------------------------------------------------------------
# CONFIG — adjust paths/model here if your folder layout differs
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DATA_DIR = BASE_DIR / "chroma_db" / "data" / "kb"
TICKETS_DATA_DIR = BASE_DIR / "chroma_db" / "data" / "tickets"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

KB_COLLECTION_NAME = "kb_articles"
TICKETS_COLLECTION_NAME = "tickets"

# Local, free, no-API-key embedding model (same one referenced in the TRO
# proposal for KB/ticket embeddings). Swap this embedding function out for
# OpenAIEmbeddingFunction / AzureOpenAI / etc. if you move to a hosted model.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------------------------------
# STEP 1 — load JSON payloads from disk (placeholders — no data hardcoded here)
# --------------------------------------------------------------------------
def load_json_array_files(directory: Path) -> list[dict]:
    """
    Reads every *.json file in `directory`, expects each file to contain a
    JSON array of objects, and returns one flat list of all records combined.
    """
    if not directory.exists():
        print(f"[WARN] Directory not found, skipping: {directory}")
        return []

    records: list[dict] = []
    json_files = sorted(glob.glob(str(directory / "*.json")))

    if not json_files:
        print(f"[WARN] No .json files found in {directory}")
        return []

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse {file_path}: {e}")
                continue

        if not isinstance(data, list):
            print(f"[WARN] {file_path} does not contain a JSON array, skipping")
            continue

        print(f"[OK] Loaded {len(data)} records from {os.path.basename(file_path)}")
        records.extend(data)

    return records


# --------------------------------------------------------------------------
# STEP 2 — build the ChromaDB client + collections
# --------------------------------------------------------------------------
def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )


def get_or_create_collections(client, embed_fn):
    kb_collection = client.get_or_create_collection(
        name=KB_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={
            "description": "SOP/FAQ knowledge base articles for Agent 3 deflection",
            "hnsw:space": "cosine",   # <-- added: makes distances cosine, not default L2
        },
    )
    tickets_collection = client.get_or_create_collection(
        name=TICKETS_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={
            "description": "Synthetic/real tickets for Agent 4 correlation & duplicate detection",
            "hnsw:space": "cosine",
        },
    )
    return kb_collection, tickets_collection


# --------------------------------------------------------------------------
# STEP 3 — transform records into Chroma's (ids, documents, metadatas) shape
# --------------------------------------------------------------------------
def prepare_kb_records(kb_records: list[dict]):
    """
    Embeds on `title + symptoms` (what a user's issue text will semantically
    match against) and keeps the full resolution + everything else as metadata.
    """
    ids, documents, metadatas = [], [], []

    for record in kb_records:
        kb_id = record.get("kb_id")
        if not kb_id:
            print(f"[WARN] Skipping KB record with no kb_id: {record}")
            continue

        document_text = f"{record.get('title', '')}. {record.get('symptoms', '')}"

        ids.append(kb_id)
        documents.append(document_text)
        metadatas.append(
            {
                "category": record.get("category", ""),
                "title": record.get("title", ""),
                "symptoms": record.get("symptoms", ""),
                "resolution": record.get("resolution", ""),
            }
        )

    return ids, documents, metadatas


def prepare_ticket_records(ticket_records: list[dict]):
    """
    Embeds on `subject + message` (+ ocr_text if present) since that's the
    free-text a duplicate/incident-correlation check needs to compare against.
    All structured fields (category, building, labels, etc.) go to metadata.
    """
    ids, documents, metadatas = [], [], []

    for record in ticket_records:
        ticket_id = record.get("ticket_id")
        if not ticket_id:
            print(f"[WARN] Skipping ticket record with no ticket_id: {record}")
            continue

        text_parts = [record.get("subject", ""), record.get("message", "")]
        if record.get("ocr_text"):
            text_parts.append(record["ocr_text"])
        document_text = ". ".join(part for part in text_parts if part)

        ids.append(ticket_id)
        documents.append(document_text)

        # Chroma metadata values must be str/int/float/bool (no None, no lists)
        # so we sanitize before inserting.
        metadata = {
            "category_selected": record.get("category_selected", ""),
            "true_category": record.get("true_category", ""),
            "building": record.get("building", ""),
            "room": record.get("room") or "",
            "priority": record.get("priority", ""),
            "completeness_level": record.get("completeness_level", ""),
            "is_deflectable": bool(record.get("is_deflectable", False)),
            "duplicate_cluster_id": record.get("duplicate_cluster_id") or "",
            "missing_fields": ",".join(record.get("missing_fields", [])),
        }
        metadatas.append(metadata)

    return ids, documents, metadatas


# --------------------------------------------------------------------------
# STEP 4 — insert (upsert) into Chroma
# --------------------------------------------------------------------------
def upsert_into_collection(collection, ids, documents, metadatas, label: str):
    if not ids:
        print(f"[WARN] Nothing to insert for {label} (empty list)")
        return

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[OK] Upserted {len(ids)} records into '{collection.name}' ({label})")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    print("=== TRO ChromaDB Ingestion ===")

    # --- 1. Load raw JSON payloads from disk (placeholders, see paths above)
    kb_records = load_json_array_files(KB_DATA_DIR)
    ticket_records = load_json_array_files(TICKETS_DATA_DIR)

    # --- 2. Set up Chroma client, embedding function, collections
    client = get_chroma_client()
    embed_fn = get_embedding_function()
    kb_collection, tickets_collection = get_or_create_collections(client, embed_fn)

    # --- 3. Transform into Chroma's expected shape
    kb_ids, kb_docs, kb_meta = prepare_kb_records(kb_records)
    ticket_ids, ticket_docs, ticket_meta = prepare_ticket_records(ticket_records)

    # --- 4. Insert
    upsert_into_collection(kb_collection, kb_ids, kb_docs, kb_meta, "KB articles")
    upsert_into_collection(tickets_collection, ticket_ids, ticket_docs, ticket_meta, "tickets")

    print("\n=== Done ===")
    print(f"kb_articles collection count: {kb_collection.count()}")
    print(f"tickets collection count:     {tickets_collection.count()}")


if __name__ == "__main__":
    main()
