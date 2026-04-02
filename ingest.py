from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import re

from pypdf import PdfReader
from docx import Document


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_txt_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(parts)


def load_document(path: Path) -> Dict[str, str] | None:
    suffix = path.suffix.lower()

    if suffix in [".txt", ".md"]:
        text = read_txt_md(path)
    elif suffix == ".pdf":
        text = read_pdf(path)
    elif suffix == ".docx":
        text = read_docx(path)
    else:
        return None

    text = clean_text(text)
    if not text:
        return None

    return {"source": str(path), "text": text}


def load_folder(folder: str | Path) -> List[Dict[str, str]]:
    folder = Path(folder)
    docs = []

    for path in sorted(folder.rglob("*")):
        if path.is_file():
            item = load_document(path)
            if item:
                docs.append(item)

    return docs


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 180) -> List[str]:
    text = clean_text(text)
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


def build_chunks(docs: List[Dict[str, str]], chunk_size: int = 1000, overlap: int = 180):
    items = []
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