import re
import uuid
import numpy as np
import faiss
from typing import List, Dict, Tuple
from pypdf import PdfReader
from openai import OpenAI
import os

# Global FAISS index (in-memory)
_index = None
_texts = []
_metas = []

def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def pdf_to_text(file) -> str:
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        chunks.append(text[i:end])
        i = max(0, end - overlap)
    return [c for c in chunks if c.strip()]

def add_docs(texts: List[str], metadatas: List[Dict]):
    global _index, _texts, _metas
    client = get_client()
    embeds = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    vecs = np.array([e.embedding for e in embeds.data]).astype("float32")

    if _index is None:
        _index = faiss.IndexFlatL2(vecs.shape[1])
    _index.add(vecs)
    _texts.extend(texts)
    _metas.extend(metadatas)
    return list(range(len(_texts) - len(texts), len(_texts)))

def search(query: str, k: int = 6) -> Dict:
    global _index, _texts, _metas
    if _index is None or len(_texts) == 0:
        return {"documents":[[]], "metadatas":[[]]}

    client = get_client()
    q_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding
    q_vec = np.array([q_embed]).astype("float32")

    D, I = _index.search(q_vec, min(k, len(_texts)))
    docs = [ _texts[i] for i in I[0] ]
    metas = [ _metas[i] for i in I[0] ]
    return {"documents":[docs], "metadatas":[metas]}

def join_context(out) -> Tuple[str, List[str]]:
    docs = out.get("documents", [[]])[0]
    metas = out.get("metadatas", [[]])[0]
    parts = []
    titles = []
    for doc, m in zip(docs, metas):
        title = m.get("title", "Document")
        titles.append(title)
        parts.append(f"[{title}]\n{doc}")
    return "\n\n".join(parts), titles
