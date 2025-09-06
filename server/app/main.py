# server/app/main.py
import os
# Stability on macOS (no OpenMP storms)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid
import shutil
import re
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pdfplumber
import numpy as np

from sentence_transformers import SentenceTransformer

# ---------------------------
# CONFIG
# ---------------------------
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Extractive summarization settings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 160_000          # safety cap on text length processed
MAX_SENTENCES_EMBED = 800    # cap sentences to embed (speed)
SUMMARY_SENTENCES = 6        # how many sentences to include in summary/bullets
MMR_REDUNDANCY = 0.35        # redundancy penalty (0..1), higher = more diverse

# ---------------------------
# APP
# ---------------------------
app = FastAPI(title="Fast Document Summarizer", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------
# MODEL LOAD (once)
# ---------------------------
print("Loading sentence embedder…")
embedder = SentenceTransformer(EMBED_MODEL)
print("Embedder ready.")

# ---------------------------
# UTILS
# ---------------------------
def _extract_pdf_text_by_page(pdf_path: str) -> List[str]:
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            pages_text.append(t)
    return pages_text

_SENT_SPLIT = re.compile(r'(?<=[\.\?!])\s+')
def _split_sentences(text: str) -> List[str]:
    # basic sentence splitter; filters very short/empty strings
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    # drop extremely short fragments (e.g., “F/E”, headers)
    sents = [s for s in sents if len(s) > 20]
    return sents

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def _centroid_mmr(sentences: List[str], k: int, redundancy: float) -> List[int]:
    """
    Select k sentences using centroid similarity with MMR-style redundancy penalty.
    Returns indices into `sentences` preserving original order.
    Robust to single-sentence / 1D embeddings.
    """
    # Embed with MiniLM (fast); force numpy + normalized vectors
    embs = embedder.encode(
        sentences,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim == 1:  # (d,) -> (1, d)
        embs = embs.reshape(1, -1)

    # centroid similarity
    centroid = embs.mean(axis=0, keepdims=True)   # (1, d)
    # embs and centroid are already normalized, so dot = cosine
    centrality = (embs @ centroid.T).ravel()      # (n,)

    k = min(k, len(sentences))
    if k <= 0:
        return []

    selected: List[int] = []
    selected_set = set()

    # pick most central first
    first = int(np.argmax(centrality))
    selected.append(first)
    selected_set.add(first)

    # greedy MMR
    for _ in range(1, k):
        best_i = None
        best_score = -1e9
        # precompute selected matrix
        sel_idx = np.array(selected, dtype=int)
        sel_mat = embs[sel_idx, :]                 # (s, d)
        if sel_mat.ndim == 1:                      # (d,) -> (1, d)
            sel_mat = sel_mat.reshape(1, -1)

        for i in range(len(sentences)):
            if i in selected_set:
                continue
            # centrality
            c = centrality[i]
            # redundancy penalty: max cosine sim to any selected
            v = embs[i:i+1, :]                     # (1, d)
            sims = (v @ sel_mat.T).ravel()         # (s,)
            red = float(sims.max()) if sims.size else 0.0
            score = c - redundancy * red
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break
        selected.append(best_i)
        selected_set.add(best_i)

    selected.sort() 
    return selected

def _guess_author_name(pages: List[str]) -> str | None:
    """
    Very light heuristic to detect a name on page 1.
    """
    text = "\n".join(pages[:1])
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # look near signature
    sig_idx = None
    for i, l in enumerate(lines):
        if re.search(r'@', l) or re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', l):
            sig_idx = i
            break
    candidates = []
    if sig_idx is not None:
        for j in range(max(0, sig_idx-3), sig_idx):
            candidates.append(lines[j])
    candidates = lines[:4] + candidates
    for c in candidates:
        parts = [p for p in re.split(r'\s+', c) if p and p.isalpha()]
        if len(parts) >= 2 and all(p[0].isupper() for p in parts[:2]):
            name = " ".join(parts[:2])
            if name.isupper():
                name = name.title()
            return name
    return None

def _neutralize_first_person(text: str, author: str | None, doc_label: str = "the document") -> str:
    subj = author or "the applicant"
    rules = [
        (r"\bI am\b", f"{subj} is"),
        (r"\bI'm\b", f"{subj} is"),
        (r"\bI was\b", f"{subj} was"),
        (r"\bI have\b", f"{subj} has"),
        (r"\bI've\b", f"{subj} has"),
        (r"\bI had\b", f"{subj} had"),
        (r"\bI will\b", f"{subj} will"),
        (r"\bI'll\b", f"{subj} will"),
        (r"\bI can\b", f"{subj} can"),
        (r"\bI built\b", f"{subj} built"),
        (r"\bI led\b", f"{subj} led"),
        (r"\bI developed\b", f"{subj} developed"),
        (r"\bI designed\b", f"{subj} designed"),
        (r"\bI improved\b", f"{subj} improved"),
        (r"\bI\b", subj),
        (r"\bmy\b", f"{subj}'s"),
        (r"\bme\b", subj),
        (r"\bmine\b", f"{subj}'s"),
        (r"\bour\b", "the team’s"),
        (r"\bours\b", "the team’s"),
        (r"\bwe\b", "the team"),
        (r"\bWe\b", "The team"),
    ]
    out = text
    for pat, rep in rules:
        out = re.sub(pat, rep, out)
    out = out.strip()
    if out and not re.match(rf"^{re.escape(subj)}\b|^{doc_label}\b", out):
        out = f"{doc_label.capitalize()} states that {out[0].lower() + out[1:]}"
    return out

def summarize_extractive(pages: List[str]) -> Dict[str, Any]:
    """
    Fast extractive pipeline:
      1) join pages, cap length
      2) sentence split
      3) embed + centroid/MMR selection
      4) objective-voice rewrite (lightweight)
    """
    author = _guess_author_name(pages)
    full = " ".join(pages)
    full = " ".join(full.split())
    if not full:
        return {"summary": "", "key_points": []}

    if len(full) > MAX_CHARS:
        cut = full[:MAX_CHARS]
        last_dot = cut.rfind(".")
        if last_dot > 200:
            cut = cut[: last_dot + 1]
        full = cut

    sents = _split_sentences(full)
    if not sents:
        return {"summary": "", "key_points": []}

    # cap sentence count to keep embedding fast
    if len(sents) > MAX_SENTENCES_EMBED:
        sents = sents[:MAX_SENTENCES_EMBED]

    idxs = _centroid_mmr(sents, k=SUMMARY_SENTENCES, redundancy=MMR_REDUNDANCY)
    chosen = [sents[i] for i in idxs]

    # objective voice pass
    doc_label = "the file" 
    bullets = []
    for s in chosen:
        t = _neutralize_first_person(s, author, doc_label=doc_label)
        if not t.endswith("."):
            t += "."
        bullets.append("• " + t)

    paragraph = " ".join([re.sub(r"^[•\-\u2022]\s*", "", b).lstrip("• ").strip() for b in bullets])
    return {"summary": paragraph, "key_points": bullets}

# ---------------------------
# SCHEMAS
# ---------------------------
class UploadResponse(BaseModel):
    doc_id: str
    pages: int
    summary: str
    key_points: List[str]

# ---------------------------
# ROUTES
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save PDF
    doc_id = str(uuid.uuid4())[:8]
    ddir = os.path.join(STORAGE_DIR, doc_id)
    os.makedirs(ddir, exist_ok=True)
    pdf_path = os.path.join(ddir, "document.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract text
    try:
        pages = _extract_pdf_text_by_page(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF read failed: {e}")

    if sum(len(p) for p in pages) == 0:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF. (Scanned images need OCR — we can add that next.)"
        )

    # Fast extractive summary
    try:
        result = summarize_extractive(pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    # Optionally store text
    with open(os.path.join(ddir, "fulltext.txt"), "w") as f:
        f.write("\n\n".join(pages))

    return UploadResponse(
        doc_id=doc_id,
        pages=len(pages),
        summary=result["summary"],
        key_points=result["key_points"]
    )
