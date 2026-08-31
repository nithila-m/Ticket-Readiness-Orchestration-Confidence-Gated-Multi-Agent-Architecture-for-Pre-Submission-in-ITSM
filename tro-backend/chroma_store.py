"""
Persistent ChromaDB client for TRO.

Two collections are created if they don't already exist:
  - kb_articles : will hold the knowledge base (Person B)
  - tickets     : will hold past tickets for duplicate detection (Person B)

Both collections start empty. This file only wires up storage;
it does not add documents, compute embeddings, or run queries.
"""

import chromadb

# Persistent client stores data on disk under ./chroma_data
# so collections survive server restarts.
CHROMA_PATH = "./chroma_data"

client = chromadb.PersistentClient(path=CHROMA_PATH)

# get_or_create is idempotent: safe to import this module many times.
kb_articles = client.get_or_create_collection(name="kb_articles")
tickets = client.get_or_create_collection(name="tickets")


def list_collections() -> list[str]:
    """Return the names of all collections in the persisted store."""
    return [c.name for c in client.list_collections()]


def collection_counts() -> dict[str, int]:
    """Return document count per collection. Useful for sanity checks."""
    return {
        "kb_articles": kb_articles.count(),
        "tickets": tickets.count(),
    }


if __name__ == "__main__":
    # Running `python chroma_store.py` directly does a quick sanity print.
    print(f"Chroma persistence path: {CHROMA_PATH}")
    print(f"Collections present: {list_collections()}")
    print(f"Document counts: {collection_counts()}")

'''
A couple of design notes:
* get_or_create_collection means importing this module never crashes on a re-run — 
  the collections just come back as-is if they already exist.
* The two helper functions (list_collections, collection_counts) exist purely for verification. 
  Person B can ignore them and use kb_articles / tickets directly.
* The if __name__ == "__main__" block lets you run python chroma_store.py as a 
  self-test — no separate script needed for a basic check.
'''