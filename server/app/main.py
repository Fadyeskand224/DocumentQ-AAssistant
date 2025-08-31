import os
import uuid
import json
import time
import shutil
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pdfplumber
import numpy as np

import faiss
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ---------------------------
# CONFIG
# ---------------------------

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QA_MODEL_NAME = "deepset/roberta-base-squad2"

CHUNK_MAX_TOKENS = 900     # target chunk length
CHUNK_OVERLAP_TOKENS = 150 # keep context
TOP_N = 30                 # retrieve top-N for rerank
TOP_K = 5                  # rerank → top-K for QA

# ---------------------------
# APP
# ---------------------------

app = FastAPI(title="Document Q&A Assistant", version="0.1.0")

# CORS - dev only (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# MODELS (load once)
# ---------------------------

print("Loading models...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANK_MODEL_NAME)
qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
qa_model.eval()
print("Models loaded.")

# ---------------------------
# UTILS
# ---------------------------

def _doc_dir(doc_id: str) -> str:
    d = os.path.join(STORAGE_DIR, doc_id)
    os.makedirs(d, exist_ok=True)
    return d

def _normalize(vecs: np.ndarray) -> np.ndarray:
    # cosine similarity with FAISS IP requires normalized vectors
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def _chunk_text(text: str, max_tokens: int = CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    # Simple whitespace token-based chunking
    toks = text.split()
    chunks = []
    i = 0
    while i < len(toks):
        chunk = toks[i:i+max_tokens]
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks

def _extract_pdf_text_by_page(pdf_path: str) -> List[str]:
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            pages_text.append(t)
    return pages_text

def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index

def _save_index(index: faiss.IndexFlatIP, path: str) -> None:
    faiss.write_index(index, path)

def _load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

# ---------------------------
# SCHEMAS
# ---------------------------

class QARequest(BaseModel):
    doc_id: str
    question: str

class QAResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[Dict[str, Any]]
    retrieval_time_ms: int

# ---------------------------
# ROUTES
# ---------------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())[:8]
    ddir = _doc_dir(doc_id)
    pdf_path = os.path.join(ddir, "document.pdf")

    # save pdf
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # extract pages → chunk → embed → index
    pages = _extract_pdf_text_by_page(pdf_path)
    if sum(len(p) for p in pages) == 0:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    chunks_meta = []  # [{chunk_id, page, text}]
    all_chunks = []
    for p_idx, page_text in enumerate(pages):
        if not page_text.strip():
            continue
        page_chunks = _chunk_text(page_text)
        for ch in page_chunks:
            if ch.strip():
                chunk_id = str(uuid.uuid4())[:8]
                chunks_meta.append({
                    "chunk_id": chunk_id,
                    "page": p_idx + 1,
                    "text": ch
                })
                all_chunks.append(ch)

    # embeddings
    emb = embedder.encode(all_chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    emb = emb.astype(np.float32)

    # FAISS index (cosine via IP b/c we normalized)
    index = _build_faiss_index(emb)
    _save_index(index, os.path.join(ddir, "index.faiss"))

    # save metadata
    with open(os.path.join(ddir, "chunks.json"), "w") as f:
        json.dump(chunks_meta, f)

    return {"doc_id": doc_id, "pages": len(pages), "chunks": len(chunks_meta)}

@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    doc_id = req.doc_id
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    ddir = _doc_dir(doc_id)
    index_path = os.path.join(ddir, "index.faiss")
    chunks_path = os.path.join(ddir, "chunks.json")

    if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
        raise HTTPException(status_code=404, detail="Document not found or not indexed.")

    index = _load_index(index_path)
    with open(chunks_path, "r") as f:
        chunks_meta = json.load(f)

    # retrieve
    t0 = time.time()
    q_emb = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    D, I = index.search(q_emb, TOP_N)  # cosine similarities because normalized
    retrieved = [(int(idx), float(sim)) for idx, sim in zip(I[0], D[0]) if idx != -1]

    # prepare passages
    passages = [chunks_meta[i]["text"] for i, _ in retrieved]
    pairs = [(question, p) for p in passages]

    # rerank
    if len(pairs) > 0:
        rr_scores = reranker.predict(pairs)  # higher is better
        rr = list(zip(range(len(passages)), rr_scores))
        rr.sort(key=lambda x: x[1], reverse=True)
        top = rr[: min(TOP_K, len(rr))]
    else:
        top = []

    # extractive QA over top passages
    best_answer = ""
    best_conf = 0.0
    best_citation = None

    for idx, rr_score in top:
        p_text = passages[idx]
        inputs = qa_tokenizer(question, p_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)
            start_scores = outputs.start_logits[0].numpy()
            end_scores = outputs.end_logits[0].numpy()

        # find best span
        start_idx = int(np.argmax(start_scores))
        end_idx = int(np.argmax(end_scores))
        if end_idx < start_idx:
            continue

        input_ids = inputs["input_ids"][0].numpy().tolist()
        answer_ids = input_ids[start_idx : end_idx + 1]
        ans = qa_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        # approximate confidence
        span_score = (start_scores[start_idx] + end_scores[end_idx]) / 2.0
        # combine with rerank score (simple min-max scaling)
        rr_min, rr_max = float(min(r for _, r in top)), float(max(r for _, r in top))
        rr_norm = (rr_score - rr_min) / (rr_max - rr_min + 1e-12)
        conf = float(0.5 * rr_norm + 0.5 * (span_score / (abs(span_score) + 10)))  # crude calibration

        if ans and conf > best_conf:
            best_conf = conf
            best_answer = ans
            global_idx = retrieved[idx][0]
            best_citation = {
                "page": chunks_meta[global_idx]["page"],
                "chunk_id": chunks_meta[global_idx]["chunk_id"],
                "text": chunks_meta[global_idx]["text"][:500],  # snippet cap
            }

    latency_ms = int((time.time() - t0) * 1000)

    # fallback when unsure
    if not best_answer or best_conf < 0.25:
        # return top 3 passages as evidence
        citations = []
        for i, _sim in retrieved[:3]:
            citations.append({
                "page": chunks_meta[i]["page"],
                "chunk_id": chunks_meta[i]["chunk_id"],
                "text": chunks_meta[i]["text"][:500]
            })
        return QAResponse(
            answer="I'm not confident enough to answer. Here are the most relevant passages.",
            confidence=round(float(best_conf), 3),
            citations=citations,
            retrieval_time_ms=latency_ms
        )

    return QAResponse(
        answer=best_answer,
        confidence=round(float(best_conf), 3),
        citations=[best_citation] if best_citation else [],
        retrieval_time_ms=latency_ms
    )
