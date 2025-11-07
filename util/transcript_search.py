#!/usr/bin/env python3
import os, json, glob, math, argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple

import srt
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

# --------------------------
# Config (change as you like)
# --------------------------
EMBED_MODEL = "BAAI/bge-m3"                 # multilingual, strong dense
RERANK_MODEL = "BAAI/bge-reranker-large"    # strong; if GPU RAM is tight, try -base or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
CHUNK_MAX_CHARS = 600                       # soft cap for chunk text length
CHUNK_OVERLAP = 80                          # characters overlap when merging consecutive subs
TOPK_RETRIEVE = 50
TOPN_RETURN = 10
NORMALIZE_EMB = True                        # enables cosine with FAISS inner product

@dataclass
class Chunk:
    doc_id: str        # filename without extension
    source_path: str   # full path to the srt
    start: float       # seconds
    end: float         # seconds
    text: str          # chunk text

def read_srt_file(path: str) -> List[Chunk]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    subs = list(srt.parse(content))

    # Merge subs into larger chunks with soft length cap + a small overlap
    chunks: List[Chunk] = []
    cur_text, cur_start, cur_end = [], None, None
    fn = os.path.basename(path)
    doc_id = os.path.splitext(fn)[0]

    def flush():
        nonlocal cur_text, cur_start, cur_end
        if cur_text:
            t = " ".join(cur_text).strip()
            if t:
                chunks.append(Chunk(
                    doc_id=doc_id,
                    source_path=path,
                    start=cur_start,
                    end=cur_end,
                    text=t
                ))
        cur_text, cur_start, cur_end = [], None, None

    prev_tail = ""
    for s in subs:
        start_s = s.start.total_seconds()
        end_s = s.end.total_seconds()
        txt = s.content.replace("\n", " ").strip()

        if cur_start is None:
            cur_start = start_s
        cur_end = end_s

        # Append with small overlap help (softly join contiguous subs)
        piece = txt
        # If adding would exceed limit, flush then start anew (carry tiny overlap)
        if sum(len(p) for p in cur_text) + len(piece) + (1 if cur_text else 0) > CHUNK_MAX_CHARS:
            flush()
            # start new with a little context overlap from previous chunk tail
            if prev_tail:
                cur_start = max(start_s - 0.5, 0.0)
                cur_text = [prev_tail]
            else:
                cur_text = []
            cur_end = end_s

        if cur_text:
            cur_text.append(" " + piece)
        else:
            cur_text = [piece]

        # update prev_tail for next chunk
        short_tail = piece[-CHUNK_OVERLAP:].strip() if len(piece) > CHUNK_OVERLAP else piece
        prev_tail = short_tail

    flush()
    return chunks

def ingest_folder(srt_dir: str) -> List[Chunk]:
    paths = sorted(glob.glob(os.path.join(srt_dir, "**", "*.srt"), recursive=True))
    all_chunks: List[Chunk] = []
    for p in paths:
        try:
            cs = read_srt_file(p)
            all_chunks.extend(cs)
        except Exception as e:
            print(f"[warn] failed to parse {p}: {e}")
    return all_chunks

def build_dense_index(chunks: List[Chunk], embed_model: str, normalize: bool = True):
    model = SentenceTransformer(embed_model)
    texts = [c.text for c in chunks]
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=normalize)
    dim = embs.shape[1]
    # Cosine ≈ inner-product if normalized
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    return index, embs, model

def build_bm25(chunks: List[Chunk]):
    tokenized = [ch.text.split() for ch in chunks]
    return BM25Okapi(tokenized), tokenized

def dense_search(query: str, index, enc_model: SentenceTransformer, topk: int, normalize: bool):
    q = enc_model.encode([query], convert_to_numpy=True, normalize_embeddings=normalize)
    D, I = index.search(q.astype(np.float32), topk)
    return I[0], D[0]

def hybrid_merge(dense_ids: np.ndarray, dense_scores: np.ndarray,
                 bm25: BM25Okapi, tokenized_docs: List[List[str]],
                 query: str, alpha: float = 0.6, topk: int = 50):
    """
    Simple hybrid: normalize dense to [0,1], BM25 to [0,1], combine with alpha.
    """
    # BM25
    bm25_scores = bm25.get_scores(query.split())
    bm25_scores = np.asarray(bm25_scores, dtype=np.float32)
    # Take candidate union (dense topk ∪ top BM25 topk)
    bm25_top_idx = np.argsort(-bm25_scores)[:topk]
    cand = set(dense_ids.tolist()) | set(bm25_top_idx.tolist())
    cand = np.array(sorted(list(cand)))

    # min-max normalize
    def norm(x):
        if np.allclose(x.max(), x.min()):  # constant
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    dense_all = np.full_like(bm25_scores, fill_value=dense_scores.min() if len(dense_scores) else 0, dtype=np.float32)
    dense_all[dense_ids] = dense_scores
    d_norm = norm(dense_all)
    b_norm = norm(bm25_scores)

    combo = alpha * d_norm[cand] + (1 - alpha) * b_norm[cand]
    order = np.argsort(-combo)
    merged_ids = cand[order][:topk]
    merged_scores = combo[order][:topk]
    return merged_ids, merged_scores

def rerank(query: str, chunks: List[Chunk], cand_idx: np.ndarray, reranker: CrossEncoder, topn: int):
    pairs = [(query, chunks[i].text) for i in cand_idx]
    scores = reranker.predict(pairs)  # higher is better
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores)[:topn]
    return cand_idx[order], scores[order]

def pretty_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - (h*3600 + m*60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:06.3f}"
    return f"{m:02d}:{sec:06.3f}"

def search_pipeline(query: str, chunks: List[Chunk],
                    faiss_index, emb_model, bm25, tokenized_docs, reranker,
                    topk=TOPK_RETRIEVE, topn=TOPN_RETURN, normalize=True, hybrid=True):
    dense_ids, dense_scores = dense_search(query, faiss_index, emb_model, topk, normalize)
    if hybrid:
        cand_ids, _ = hybrid_merge(dense_ids, dense_scores, bm25, tokenized_docs, query, alpha=0.6, topk=topk)
    else:
        cand_ids = dense_ids

    # Optional quick fuzzy boost inside top candidates (lightweight heuristic)
    # (You can remove this if you prefer pure neural ranking.)
    texts = [chunks[i].text for i in cand_ids]
    fuzz_scores = np.array([fuzz.partial_ratio(query, t) for t in texts], dtype=np.float32) / 100.0
    # add a tiny boost to keep exact/near matches high before rerank
    # not used to filter; purely a nudge in case your reranker is lightweight
    # (we won't reorder here; reranker will finalize)

    reranked_ids, rer_scores = rerank(query, chunks, cand_ids, reranker, topn)
    results = []
    for rid, rs in zip(reranked_ids.tolist(), rer_scores.tolist()):
        c = chunks[rid]
        results.append({
            "doc_id": c.doc_id,
            "source_path": c.source_path,
            "start": c.start,
            "end": c.end,
            "time_range": f"{pretty_time(c.start)} → {pretty_time(c.end)}",
            "score": float(rs),
            "text": c.text
        })
    return results

def save_index(out_dir: str, chunks: List[Chunk], embs: np.ndarray, faiss_index):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(faiss_index, os.path.join(out_dir, "vectors.faiss"))
    np.save(os.path.join(out_dir, "embeddings.npy"), embs.astype(np.float32))
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

def load_index(out_dir: str):
    idx = faiss.read_index(os.path.join(out_dir, "vectors.faiss"))
    embs = np.load(os.path.join(out_dir, "embeddings.npy"))
    meta = []
    with open(os.path.join(out_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta.append(Chunk(**d))
    return idx, embs, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt_dir", type=str, help="Folder with .srt files")
    ap.add_argument("--build", action="store_true", help="Build (ingest + index)")
    ap.add_argument("--index_dir", type=str, default="./srt_index", help="Where to save/load the FAISS index & metadata")
    ap.add_argument("--query", type=str, help="Run a query after building / loading")
    ap.add_argument("--no_hybrid", action="store_true", help="Disable BM25+dense hybrid")
    args = ap.parse_args()

    if args.build:
        print("[info] Ingesting SRTs…")
        chunks = ingest_folder(args.srt_dir)
        print(f"[info] Got {len(chunks)} chunks")

        print("[info] Building dense index…")
        faiss_idx, embs, emb_model = build_dense_index(chunks, EMBED_MODEL, NORMALIZE_EMB)

        print("[info] Building BM25…")
        bm25, tokenized = build_bm25(chunks)

        print("[info] Loading reranker…")
        reranker = CrossEncoder(RERANK_MODEL)

        print("[info] Saving index…")
        save_index(args.index_dir, chunks, embs, faiss_idx)

        if args.query:
            print(f"[info] Query: {args.query}")
            res = search_pipeline(
                args.query, chunks, faiss_idx, emb_model, bm25, tokenized, reranker,
                topk=TOPK_RETRIEVE, topn=TOPN_RETURN, normalize=NORMALIZE_EMB, hybrid=not args.no_hybrid
            )
            for i, r in enumerate(res, 1):
                print(f"\n#{i} [{r['score']:.4f}] {r['doc_id']}  {r['time_range']}")
                print(r["text"])
    else:
        # Load & search only
        print("[info] Loading existing index…")
        faiss_idx, embs, chunks = load_index(args.index_dir)

        # You still need models for encoding & reranking at search time
        emb_model = SentenceTransformer(EMBED_MODEL)
        bm25, tokenized = build_bm25(chunks)
        reranker = CrossEncoder(RERANK_MODEL)

        while True:
            q = args.query if args.query else input("\nQuery (empty to quit): ").strip()
            if not q:
                break
            res = search_pipeline(
                q, chunks, faiss_idx, emb_model, bm25, tokenized, reranker,
                topk=TOPK_RETRIEVE, topn=TOPN_RETURN, normalize=NORMALIZE_EMB, hybrid=not args.no_hybrid
            )
            for i, r in enumerate(res, 1):
                print(f"\n#{i} [{r['score']:.4f}] {r['doc_id']}  {r['time_range']}")
                print(r["text"])
            if args.query:
                break

if __name__ == "__main__":
    main()
