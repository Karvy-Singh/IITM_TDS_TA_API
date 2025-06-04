#!/usr/bin/env python3
"""
test_faiss_search.py

1) Load scraped posts from 'tds_discourse_posts.json'
2) Chunk each post into ~300–500‐word snippets
3) Embed with Sentence‐Transformers
4) Build a FAISS index
5) Enter a CLI loop: ask any text question → see top‐5 matching snippets (with post URLs)
"""

import json
import os
import sys
import pickle
from datetime import datetime
from tqdm import tqdm

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

# ─────── CONFIGURATION ───────

# 1) Path to your scraped JSON file that you generated earlier:
SCRAPED_JSON_PATH = "tds_discourse_posts.json"

# 2) Where to save the FAISS index + metadata (so you don't rebuild every time)
INDEX_PATH    = "tds_discourse_index.faiss"
META_PATH     = "tds_discourse_metadata.pkl"

# 3) Which sentence‐transformer model to use for embeddings:
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# 4) How many search hits to return:
TOP_K = 5

# ─────── HELPERS ───────

def chunk_text(text, max_len_chars=500):
    """
    Split a long text into smaller chunks of up to roughly max_len_chars,
    by sentence. Uses NLTK's sent_tokenize.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_len_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks


def build_index(posts_json_path, model_name, index_path, meta_path):
    """
    1) Load the scraped posts JSON.
    2) Chunk each post into snippets.
    3) Embed all snippets with sentence-transformers.
    4) Build & save a FAISS index + metadata.
    """
    # 1) Load scraped posts
    if not os.path.exists(posts_json_path):
        print(f"ERROR → could not find '{posts_json_path}'. Exiting.")
        sys.exit(1)

    print(f"Loading scraped posts from '{posts_json_path}' …")
    with open(posts_json_path, "r", encoding="utf‐8") as f:
        posts = json.load(f)

    # 2) Chunk each post into smaller snippets
    print(f"Chunking {len(posts)} posts into snippets…")
    all_snippets = []
    metadata     = []

    for post in tqdm(posts):
        text   = post.get("content", "")
        chunks = chunk_text(text, max_len_chars=500)  # ~500 characters per chunk

        for snippet in chunks:
            all_snippets.append(snippet)
            metadata.append({
                "topic_id": post.get("topic_id", ""),
                "topic_title": post.get("topic_title", ""),
                "post_number": post.get("post_number", ""),
                "created_at": post.get("created_at", ""),
                "post_url": post.get("post_url", ""),
                "snippet": snippet
            })

    # 3) Embed all snippets
    print(f"Loading SentenceTransformer model '{model_name}' …")
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(all_snippets)} snippets (this may take a minute)…")
    embeddings = model.encode(all_snippets, convert_to_numpy=True)

    # 4) Build FAISS index (L2 / cosine via normalized vectors)
    d = embeddings.shape[1]
    print(f"Building FAISS Index (dimension = {d}) …")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype("float32"))

    # 5) Save index + metadata
    print(f"Saving FAISS index to '{index_path}' …")
    faiss.write_index(index, index_path)

    print(f"Saving metadata to '{meta_path}' …")
    with open(meta_path, "wb") as mf:
        pickle.dump(metadata, mf)

    print("Index build complete.")
    return index, metadata


def load_index_and_meta(index_path, meta_path):
    """
    Load a prebuilt FAISS index and its corresponding metadata list.
    """
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None

    print(f"Loading FAISS index from '{index_path}' …")
    index = faiss.read_index(index_path)
    print(f"Loading metadata from '{meta_path}' …")
    with open(meta_path, "rb") as mf:
        metadata = pickle.load(mf)
    return index, metadata


def search_faiss(index, metadata, model, query, top_k=5):
    """
    1) Embed the query
    2) Run FAISS search
    3) Return a list of (score, metadata) for the top_k hits
    """
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    # D: squared L2 distances, I: indices in metadata
    results = []
    for dist, idx in zip(D[0], I[0]):
        meta = metadata[idx]
        results.append((dist, meta))
    return results


if __name__ == "__main__":
    # ─────── 1) Try to load an existing index & metadata ───────
    index, metadata = load_index_and_meta(INDEX_PATH, META_PATH)

    # If not found on disk, build it from the scraped JSON
    if index is None or metadata is None:
        index, metadata = build_index(
            posts_json_path=SCRAPED_JSON_PATH,
            model_name=EMBED_MODEL_NAME,
            index_path=INDEX_PATH,
            meta_path=META_PATH
        )

    # Load the same SentenceTransformer model for querying
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("\n====== FAISS SEARCH READY ======\n")

    # ─────── 2) Enter interactive loop for user questions ───────
    print("Type a question (or 'exit' to quit):")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        # 3) Perform search
        hits = search_faiss(index, metadata, model, query, top_k=TOP_K)
        print(f"\nTop {TOP_K} matches:\n")
        for rank, (dist, meta) in enumerate(hits, start=1):
            print(f"--- Rank #{rank}  (score: {dist:.4f}) ---")
            print(f"Topic ID   : {meta['topic_id']}")
            print(f"Topic Title: {meta['topic_title']}")
            print(f"Post #     : {meta['post_number']}")
            print(f"Created At : {meta['created_at']}")
            print(f"URL        : {meta['post_url']}")
            snippet = meta["snippet"]
            print(f"Snippet    : {snippet[:200].strip()}{'…' if len(snippet)>200 else ''}")
            print()
        print("───────────────────────────────────────")

