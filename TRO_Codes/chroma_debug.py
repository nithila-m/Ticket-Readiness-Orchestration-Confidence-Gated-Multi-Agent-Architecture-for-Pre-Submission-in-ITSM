"""
debug_chroma.py
================

Diagnostic script — run this BEFORE debugging retrieval logic, whenever
retrieval looks wrong. It answers, in order:

    1. Is the persisted DB even at the path you think it is?
    2. Does the "kb_articles" collection exist?
    3. Is it empty, or does it have documents?
    4. What distance space was it created with (cosine vs L2)?
    5. What do a few stored records actually look like (documents +
       metadata + embedding dimensionality)?
    6. Does a live test query return anything sensible?

Run from TRO_Codes/:
    python debug_chroma.py

Dependencies: same as the rest of the project (chromadb, sentence-transformers)
"""

from chroma_client1 import (
    get_client,
    CHROMA_PERSIST_DIR,
    KB_DATA_DIR,
    TICKETS_DATA_DIR,
    KB_COLLECTION_NAME,
    TICKETS_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


def line(char="-", n=70):
    print(char * n)


def inspect_collection(client, name: str):
    line("=")
    print(f"COLLECTION: {name}")
    line("=")

    try:
        collection = client.get_collection(name=name)
    except Exception as e:
        print(f"[FAIL] Could not open collection '{name}': {e}")
        print("       -> This collection was never created. Run insert_to_chromadb.py.")
        return

    count = collection.count()
    print(f"Collection exists: YES")
    print(f"Document count:    {count}")

    # Distance space is stored in the collection's own metadata
    meta = collection.metadata or {}
    space = meta.get("hnsw:space", "l2 (DEFAULT — was not explicitly set!)")
    print(f"Distance space:    {space}")
    if space != "cosine":
        print("[WARN] Distance space is not 'cosine'. `confidence = 1 - distance`")
        print("       will NOT be a valid similarity score. Delete chroma_db/ and")
        print("       re-run ingestion with hnsw:space='cosine' set at creation time.")

    if count == 0:
        print("\n[FAIL] Collection is EMPTY.")
        print("       -> Ingestion ran but inserted nothing, or hasn't been run yet.")
        print(f"       -> Check that {name.upper()}_DATA_DIR actually contains .json files:")
        data_dir = KB_DATA_DIR if name == KB_COLLECTION_NAME else TICKETS_DATA_DIR
        print(f"          {data_dir.resolve()}")
        print(f"          exists: {data_dir.exists()}")
        if data_dir.exists():
            files = list(data_dir.glob("*.json"))
            print(f"          .json files found: {[f.name for f in files]}")
        return

    # Peek at a small sample of what's actually stored
    sample = collection.peek(limit=3)
    print(f"\nSample of {min(3, count)} stored record(s):")
    line()
    for i in range(len(sample["ids"])):
        print(f"  id:        {sample['ids'][i]}")
        doc = sample["documents"][i] if sample.get("documents") else None
        print(f"  document:  {doc}")
        meta_i = sample["metadatas"][i] if sample.get("metadatas") else None
        print(f"  metadata:  {meta_i}")
        emb = sample["embeddings"][i] if sample.get("embeddings") is not None else None
        if emb is not None:
            print(f"  embedding: dim={len(emb)}, first 5 values={list(emb[:5])}")
        line()


def run_test_query(client, kb_collection_name: str, query_text: str, top_k: int = 5):
    line("=")
    print("TEST QUERY")
    line("=")
    print(f"Query: {query_text}")

    try:
        collection = client.get_collection(name=kb_collection_name)
    except Exception as e:
        print(f"[FAIL] Could not open '{kb_collection_name}': {e}")
        return

    if collection.count() == 0:
        print("[SKIP] Collection is empty, nothing to query against.")
        return

    results = collection.query(query_texts=[query_text], n_results=top_k)

    print(f"\nTop {top_k} results:")
    for rank, (doc_id, distance, meta) in enumerate(
        zip(results["ids"][0], results["distances"][0], results["metadatas"][0]), start=1
    ):
        similarity = 1 - distance
        title = meta.get("title", "(no title)")
        print(f"  {rank}. {doc_id} - {title} - distance={distance:.4f}  similarity={similarity:.4f}")


def main():
    line("#")
    print("TRO CHROMADB DIAGNOSTIC")
    line("#")

    print(f"\nPersist directory (both scripts should share this path):")
    print(f"  {CHROMA_PERSIST_DIR.resolve()}")
    print(f"  exists: {CHROMA_PERSIST_DIR.exists()}")

    print(f"\nEmbedding model configured: {EMBEDDING_MODEL_NAME}")
    print("  (must be IDENTICAL in insert_to_chromadb.py and retrieval_agent.py —")
    print("   both should import this value from shared/chroma_client.py, never")
    print("   hardcode it separately, or embeddings won't be comparable.)\n")

    client = get_client()

    print("All collections currently in this persisted DB:")
    existing = client.list_collections()
    if not existing:
        print("  (none — the DB at this path has never had a collection created in it)")
    else:
        for c in existing:
            print(f"  - {c.name}")
    print()

    inspect_collection(client, KB_COLLECTION_NAME)
    print()
    inspect_collection(client, TICKETS_COLLECTION_NAME)
    print()

    run_test_query(
        client,
        KB_COLLECTION_NAME,
        "VPN keeps disconnecting right after my laptop wakes up from sleep",
    )


if __name__ == "__main__":
    main()