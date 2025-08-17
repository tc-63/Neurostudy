import os, json
import streamlit as st
import pandas as pd
from openai import OpenAI

from app.prompts import (
    SYSTEM_SEARCH, SYSTEM_FLASHCARDS, SYSTEM_MCQS, SYSTEM_OSCE, SYSTEM_NOTEBOOK,
    FLASHCARD_FORMAT, MCQ_FORMAT
)
from app.utils import ensure_chroma, pdf_to_text, chunk_text, add_docs, search, join_context

st.set_page_config(page_title="NeuroStudy", page_icon="🧠")
st.title("NeuroStudy — education-only")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4o-mini","gpt-4o"])
client = OpenAI(api_key=api_key) if api_key else None

persist_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
_, coll = ensure_chroma(persist_dir)

st.header("Library")
files = st.file_uploader("Upload PDFs/TXTs", type=["pdf","txt"], accept_multiple_files=True)
if files:
    for f in files:
        if f.name.lower().endswith(".pdf"):
            text = pdf_to_text(f)
        else:
            text = f.read().decode("utf-8","ignore")
        chunks = chunk_text(text)
        add_docs(coll, chunks, [{"title": f.name}] * len(chunks))
    st.success("Added to library.")

st.header("Search")
q = st.text_input("Ask a question")
if st.button("Search & Answer", disabled=not (client and q)):
    out = search(coll, q, k=6)
    ctx,_ = join_context(out)
    msgs = [{"role":"system","content":SYSTEM_SEARCH},{"role":"user","content":f"Q: {q}\n\nContext:\n{ctx}"}]
    resp = client.chat.completions.create(model=model, messages=msgs)
    st.markdown(resp.choices[0].message.content)

st.header("Flashcards")
n = st.slider("How many?", 3, 20, 10)
if st.button("Generate Flashcards", disabled=not client):
    msgs = [{"role":"system","content":SYSTEM_FLASHCARDS},{"role":"user","content":f"Create {n} flashcards.\n{FLASHCARD_FORMAT}"}]
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.4)
    st.code(resp.choices[0].message.content)
