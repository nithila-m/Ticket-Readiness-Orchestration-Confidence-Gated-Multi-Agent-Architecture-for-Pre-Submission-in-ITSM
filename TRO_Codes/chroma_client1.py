"""
shared/chroma_client.py
========================

Single source of truth for how every script/agent connects to ChromaDB.
Both insert_to_chromadb.py (batch ingestion) and retrieval_agent.py
(Agent 2's live deflect() calls) import from here, so the persist path,
embedding model, and distance metric can never silently drift apart
between ingestion time and query time.

--------------------------------------------------------------------------
FOLDER LAYOUT THIS FILE ASSUMES
--------------------------------------------------------------------------
    TRO_Codes/
    ├── kb_insertion.py
    ├── chroma_client1.py        <- this file
    ├── agents/
    │   └── kb_retrieval_agent.py
    └── chroma_db/                  <- Chroma's own persisted index/db files
        └── data/
            ├── kb/                 <- kb_*.json files
            │   ├── kb_ad_account.json
            │   └── ...
            └── tickets/            <- tickets_*.json files
                ├── tickets_ad_account.json
                └── ...

Note: chroma_db/ holds BOTH Chroma's internal persistence files (sqlite,
index segments — written automatically by PersistentClient) AND the
data/kb, data/tickets source JSON folders. This is fine — Chroma only
touches its own internal files and never looks inside data/, so there's
no conflict, but keep this in mind if you ever want to .gitignore only
the persisted index and not the source JSON.

--------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------
    pip install chromadb sentence-transformers
"""
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
# --------------------------------------------------------------------------
# PATHS — adjust here if your layout ever changes; nowhere else needs to.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

KB_DATA_DIR = CHROMA_PERSIST_DIR / "data" / "kb"
TICKETS_DATA_DIR = CHROMA_PERSIST_DIR / "data" / "tickets"


# --------------------------------------------------------------------------
# EMBEDDING MODEL — must match whatever was used at ingestion time.
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------------------------------
# COLLECTION NAMES
# --------------------------------------------------------------------------
KB_COLLECTION_NAME = "kb_articles"
TICKETS_COLLECTION_NAME = "tickets"


def get_client() -> chromadb.PersistentClient:
    """Single persistent client, pointed at chroma_db/."""
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def get_embedding_function():
    """Local, free, no-API-key embedder — same one insert_to_chromadb.py uses."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )


def get_collection(name: str, description: str = ""):
    """
    Opens (or creates, if it doesn't exist yet) a collection by name, always
    with cosine distance space explicitly set.

    IMPORTANT: hnsw:space="cosine" must be set here AND must match whatever
    was set when the collection was first created during ingestion — Chroma
    fixes the distance metric at creation time and does not change it
    retroactively. If you ever change this, delete chroma_db/ and re-run
    insert_to_chromadb.py once to rebuild the index.
    """
    client = get_client()
    embed_fn = get_embedding_function()

    return client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={
            "description": description,
            "hnsw:space": "cosine",
        },
    )


def get_kb_collection():
    return get_collection(
        KB_COLLECTION_NAME,
        description="SOP/FAQ knowledge base articles for Agent 2 deflection",
    )


def get_tickets_collection():
    return get_collection(
        TICKETS_COLLECTION_NAME,
        description="Synthetic/real tickets for Pipeline 2 correlation & duplicate detection",
    )