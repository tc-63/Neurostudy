import chromadb, uuid, re
from chromadb.config import Settings
from pypdf import PdfReader

def ensure_chroma(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    try:
        coll = client.get_or_create_collection("neurostudy")
    except Exception:
        coll = client.create_collection("neurostudy")
    return client, coll

def pdf_to_text(file) -> str:
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        chunks.append(text[i:end])
        i = max(0, end - overlap)
    return [c for c in chunks if c.strip()]

def add_docs(coll, texts, metadatas):
    ids = [str(uuid.uuid4()) for _ in texts]
    coll.add(documents=texts, metadatas=metadatas, ids=ids)
    return ids

def search(coll, query: str, k: int = 6):
    return coll.query(query_texts=[query], n_results=k, include=["documents","metadatas","distances"])

def join_context(out):
    docs = out.get("documents", [[]])[0]
    metas = out.get("metadatas", [[]])[0]
    parts = []
    for doc, m in zip(docs, metas):
        title = m.get("title","Document")
        parts.append(f"[{title}]\n{doc}")
    return "\n\n".join(parts), []
