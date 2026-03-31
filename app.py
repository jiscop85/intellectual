from __future__ import annotations

from pathlib import Path
import shutil

import streamlit as st

from rag import read_documents, build_chunks, VectorStore
from generator import answer_question


STORAGE_DIR = Path("storage")
UPLOAD_DIR = STORAGE_DIR / "uploads"
INDEX_DIR = STORAGE_DIR / "index"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="SmartDoc AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 SmartDoc AI Assistant")
st.caption("یک دستیار هوش مصنوعی برای جست‌وجوی معنایی و پاسخ‌گویی روی اسناد شما")


@st.cache_resource
def load_store():
    index_file = INDEX_DIR / "index.faiss"
    chunks_file = INDEX_DIR / "chunks.json"
    if index_file.exists() and chunks_file.exists():
        return VectorStore.load(INDEX_DIR)
    return None


def save_uploaded_files(files):
    for file in files:
        target = UPLOAD_DIR / file.name
        target.write_bytes(file.getbuffer())


with st.sidebar:
    st.header("تنظیمات")
    files = st.file_uploader(
        "فایل‌های .txt یا .md را آپلود کن",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("اندازه هر بخش", 300, 1500, 900, 50)
    overlap = st.slider("هم‌پوشانی", 0, 400, 150, 10)
    top_k = st.slider("تعداد نتایج", 1, 10, 5, 1)

    build = st.button("ساخت / به‌روزرسانی ایندکس")

    if build:
        if files:
            save_uploaded_files(files)

        docs = read_documents(UPLOAD_DIR)
        if not docs:
            st.error("هیچ فایل معتبری برای ایندکس‌سازی پیدا نشد.")
        else:
            with st.spinner("در حال ساخت ایندکس..."):
                chunks = build_chunks(docs, chunk_size=chunk_size, overlap=overlap)
                store = VectorStore()
                store.fit(chunks)
                store.save(INDEX_DIR)
                st.session_state["store"] = store
            st.success("ایندکس با موفقیت ساخته شد.")


store = st.session_state.get("store") or load_store()

col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_input("سؤال خود را بنویس:", placeholder="مثلاً: هدف اصلی این سند چیست؟")

    ask = st.button("پاسخ بده")

    if ask:
        if not store:
            st.warning("ابتدا ایندکس را بساز یا یک ایندکس ذخیره‌شده را بارگذاری کن.")
        elif not question.strip():
            st.warning("یک سؤال وارد کن.")
        else:
            with st.spinner("در حال جست‌وجو و پاسخ‌سازی..."):
                hits = store.search(question, top_k=top_k)
                answer = answer_question(question, hits)

            st.subheader("پاسخ")
            st.write(answer)

            st.subheader("منابع نزدیک")
            for i, hit in enumerate(hits, start=1):
                with st.expander(f"منبع {i} | امتیاز: {hit['score']:.3f} | {hit['source']}"):
                    st.write(hit["text"])

with col2:
    st.subheader("وضعیت سیستم")
    if store:
        st.success("ایندکس آماده است.")
        st.write(f"تعداد بخش‌ها: {len(store.chunks)}")
    else:
        st.info("ایندکسی هنوز بارگذاری نشده است.")

    st.subheader("فایل‌های موجود")
    existing = list(UPLOAD_DIR.glob("*"))
    if existing:
        for p in existing:
            st.write(f"- {p.name}")
    else:
        st.write("فایلی وجود ندارد.")