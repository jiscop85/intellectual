from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


@dataclass
class SearchResult:
    source: str
    chunk_id: int
    text: str
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0


class HybridRAGStore:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedder = SentenceTransformer(embed_model)
        self.reranker = CrossEncoder(rerank_model)
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.bm25 = None
        self.tokenized_chunks = None

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.tokenized_chunks = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def _dense_search(self, query: str, top_k: int = 20):
        q = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, ids = self.index.search(q, top_k)
        items = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            item = dict(self.chunks[int(idx)])
            item["dense_score"] = float(score)
            items.append(item)
        return items

    def _bm25_search(self, query: str, top_k: int = 20):
        token_q = tokenize(query)
        scores = self.bm25.get_scores(token_q)
        top_idx = np.argsort(scores)[::-1][:top_k]

        items = []
        for idx in top_idx:
            item = dict(self.chunks[int(idx)])
            item["bm25_score"] = float(scores[idx])
            items.append(item)
        return items

    @staticmethod
    def _merge_candidates(dense_hits, bm25_hits, limit=30):
        by_key = {}

        for item in dense_hits:
            key = (item["source"], item["chunk_id"])
            by_key[key] = dict(item)

        for item in bm25_hits:
            key = (item["source"], item["chunk_id"])
            if key not in by_key:
                by_key[key] = dict(item)
            else:
                by_key[key]["bm25_score"] = item.get("bm25_score", 0.0)

        merged = list(by_key.values())
        merged.sort(
            key=lambda x: 0.7 * x.get("dense_score", 0.0) + 0.3 * x.get("bm25_score", 0.0),
            reverse=True,
        )
        return merged[:limit]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        dense_hits = self._dense_search(query, top_k=20)
        bm25_hits = self._bm25_search(query, top_k=20)
        candidates = self._merge_candidates(dense_hits, bm25_hits, limit=30)

        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        for c in candidates:
            c["score"] = float(
                0.55 * c.get("rerank_score", 0.0)
                + 0.30 * c.get("dense_score", 0.0)
                + 0.15 * c.get("bm25_score", 0.0)
            )

        return candidates[:top_k]

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(folder / "index.faiss"))
        (folder / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> "HybridRAGStore":
        folder = Path(folder)
        obj = cls()
        obj.index = faiss.read_index(str(folder / "index.faiss"))
        obj.chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))

        texts = [c["text"] for c in obj.chunks]
        obj.tokenized_chunks = [tokenize(t) for t in texts]
        obj.bm25 = BM25Okapi(obj.tokenized_chunks)
        return obj