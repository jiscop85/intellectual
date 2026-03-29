from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import json
import os
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def read_documents(folder: str | Path) -> List[Dict[str, str]]:
    folder = Path(folder)
    docs: List[Dict[str, str]] = []

    for path in sorted(folder.glob("**/*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append({"source": str(path), "text": text})

    return docs


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks


def build_chunks(docs: List[Dict[str, str]], chunk_size: int = 900, overlap: int = 150) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for doc in docs:
        parts = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, part in enumerate(parts):
            items.append(
                {
                    "source": doc["source"],
                    "chunk_id": i,
                    "text": part,
                }
            )

    return items


class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: List[Dict[str, Any]] = []

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index is empty. Build the index first.")

        q = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, ids = self.index.search(q, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            item = dict(self.chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)

        return results

    def save(self, folder: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Nothing to save. Build the index first.")

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(folder / "index.faiss"))
        (folder / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: str | Path, model_name: str = EMBED_MODEL_NAME) -> "VectorStore":
        folder = Path(folder)
        obj = cls(model_name=model_name)
        obj.index = faiss.read_index(str(folder / "index.faiss"))
        obj.chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        return obj