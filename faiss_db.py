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

from fastapi import FastAPI
import base64
from io import BytesIO

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from openai import OpenAI
client = OpenAI(api_key="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDExMjdAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.xJoqjnl3dcuRcZfdQ46I7sEH7N--qvq0pprsqunP5P8",
                base_url="https://aipipe.org/openai/v1/chat/completions")

# ─────── CONFIGURATION ───────

# 1) Path to your scraped JSON file that you generated earlier:
SCRAPED_JSON_PATHS = ["./discourse_filtered_posts.jsonl","./course_content.jsonl"]

# 2) Where to save the FAISS index + metadata (so you don't rebuild every time)
INDEX_PATH    = "tds_discourse_index.faiss"
META_PATH     = "tds_discourse_metadata.pkl"

# 3) Which sentence‐transformer model to use for embeddings:
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# 4) How many search hits to return:
TOP_K = 5

# ─────── HELPERS ───────

from PIL import Image, ImageFilter, ImageOps
import io, base64

def preprocess_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    # Optional: deskew / enhance contrast / remove noise
    img = img.filter(ImageFilter.MedianFilter())
    img = ImageOps.autocontrast(img, cutoff=2)
    return img

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# load once at startup
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to("cuda")

def vqa_answer(image: Image.Image, question: str) -> str:
    inputs = processor(images=image, text=question, return_tensors="pt").to("cuda")
    out    = model.generate(**inputs, max_new_tokens=64)
    return processor.decode(out[0], skip_special_tokens=True)


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

def build_index(jsonl_paths, model_name, index_path, meta_path, max_len_chars=500):
    """
    1) Load all JSONL files in jsonl_paths.
    2) For each record, detect its schema and normalize into a common metadata dict.
    3) Chunk each post.content into snippets.
    4) Embed all snippets.
    5) Build & save a single FAISS index + single metadata list.
    """
    # 1) Load & normalize posts
    print(f"Loading JSONL from: {jsonl_paths!r}")
    normalized_posts = []
    for p in jsonl_paths:
        if not os.path.exists(p):
            print(f"ERROR → could not find '{p}'. Exiting.")
            sys.exit(1)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                post = json.loads(line)
                # schema A?
                if "topic_id" in post:
                    normalized_posts.append({
                        "content":     post.get("content",      ""),
                        "post_url":    post.get("post_url",     ""),
                        "topic_id":    post.get("topic_id",     ""),
                        "topic_title": post.get("topic_title",  ""),
                        "post_number": post.get("post_number",  ""),
                        "created_at":  post.get("created_at",   ""),
                        "_source":     os.path.basename(p),
                    })
                # schema B?
                elif "content" in post and "url" in post:
                    normalized_posts.append({
                        "content":     post["content"],
                        "post_url":    post.get("url",           ""),
                        "topic_id":    "",
                        "topic_title": "",
                        "post_number": "",
                        "created_at":  "",
                        "_source":     os.path.basename(p),
                    })
                else:
                    # if you have other schemas, handle them here (or skip)
                    print(f"WARNING → skipping unrecognized record in {p}: {post.keys()}")
    print(f"Loaded and normalized {len(normalized_posts)} posts.")

    # 2) Chunk into snippets + collect metadata
    print(f"Chunking posts into snippets (max {max_len_chars} chars)…")
    all_snippets = []
    metadata     = []
    for post in tqdm(normalized_posts):
        text = post["content"]
        chunks = chunk_text(text, max_len_chars=max_len_chars)
        for snippet in chunks:
            all_snippets.append(snippet)
            # for each snippet, carry over the normalized metadata
            md = {
                "snippet":     snippet,
                "post_url":    post["post_url"],
                "topic_id":    post["topic_id"],
                "topic_title": post["topic_title"],
                "post_number": post["post_number"],
                "created_at":  post["created_at"],
                "source_file": post["_source"],
            }
            metadata.append(md)

    # 3) Embed
    print(f"Loading SentenceTransformer('{model_name}') …")
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(all_snippets)} snippets …")
    embeddings = model.encode(all_snippets, convert_to_numpy=True)

    # 4) Build FAISS index
    d = embeddings.shape[1]
    print(f"Building FAISS Index (dim={d}) …")
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

def generate_answer(index,metadata,embed_model,question: str, k: int = 5):
    hits = search_faiss(index, metadata, embed_model, question, top_k=k)

    # Build a compact context block for the LLM
    context = "\n\n".join(
        f"[{i+1}] {meta['snippet']}" for i, (_, meta) in enumerate(hits)
    )

    
    prompt = f"""
    You are a teaching assistant. Answer the student question **concisely**,
    citing sources in the form [1], [2] … that correspond to the snippets below.

    Question: {question}

    Snippets:
    {context}

    Answer:
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",     # or your Ollama model for local runs
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    ).choices[0].message.content

    # Build the links array → only the URLs that were actually cited
    links = []
    for i, (_, meta) in enumerate(hits):
        tag = f"[{i+1}]"
        if tag in completion:
            links.append({"url": meta["post_url"], "text": meta["topic_title"]})

    return {"answer": completion, "links": links}

app = FastAPI()

@app.post("/api")
def qa_endpoint(payload: dict):
    index, metadata = load_index_and_meta(INDEX_PATH, META_PATH)

    # If not foun/d on disk, build it from the scraped JSON
    if index is None or metadata is None:
        index, metadata = build_index(
            jsonl_paths=SCRAPED_JSON_PATHS,
            model_name=EMBED_MODEL_NAME,
            index_path=INDEX_PATH,
            meta_path=META_PATH
        )

    # Load the same SentenceTransformer model for querying
    model = SentenceTransformer(EMBED_MODEL_NAME)

    question = payload["question"]
    if img_b64 := payload.get("image"):
        img = preprocess_image(img_b64)

        ocr_text = pytesseract.image_to_string(img).strip()
        vqa_text = vqa_answer(img, question)

        image_context = f""" OCR extracted:
                        {ocr_text or '<none>'}
                        Vision-model summary:
                        {vqa_text}
                        """
        question += f"\n\n(Attached image's context)\n\n{image_context}"

    response = generate_answer(index,metadata,model,question)
    return response


# if __name__ == "__main__":
#     # ─────── 1) Try to load an existing index & metadata ───────
#     index, metadata = load_index_and_meta(INDEX_PATH, META_PATH)
# 
#     # If not foun/d on disk, build it from the scraped JSON
#     if index is None or metadata is None:
#         index, metadata = build_index(
#             posts_json_path=SCRAPED_JSON_PATH,
#             model_name=EMBED_MODEL_NAME,
#             index_path=INDEX_PATH,
#             meta_path=META_PATH
#         )
# 
#     # Load the same SentenceTransformer model for querying
#     model = SentenceTransformer(EMBED_MODEL_NAME)
# 
#     print("\n====== FAISS SEARCH READY ======\n")
# 
#     # ─────── 2) Enter interactive loop for user questions ───────
#     print("Type a question (or 'exit' to quit):")
#     while True:
#         query = input("\n> ").strip()
#         if query.lower() in ("exit", "quit"):
#             print("Exiting.")
#             break
# 
#         ans=generate_answer(index,metadata,model,query)
#         print("answer:", ans["answer"])
#         print()
#         print("links", ans["links"])

#         # 3) Perform search
#         hits = search_faiss(index, metadata, model, query, top_k=TOP_K)
#         print(f"\nTop {TOP_K} matches:\n")
#         for rank, (dist, meta) in enumerate(hits, start=1):
#             print(f"--- Rank #{rank}  (score: {dist:.4f}) ---")
#             print(f"Topic ID   : {meta['topic_id']}")
#             print(f"Topic Title: {meta['topic_title']}")
#             print(f"Post #     : {meta['post_number']}")
#             print(f"Created At : {meta['created_at']}")
#             print(f"URL        : {meta['post_url']}")
#             snippet = meta["snippet"]
#             print(f"Snippet    : {snippet[:200].strip()}{'…' if len(snippet)>200 else ''}")
#             print()
#         print("───────────────────────────────────────")

