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
from fastapi.responses import FileResponse
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
MMR_REDUNDANCY = 0.35        # redundancy penalty (0..1), higher = more diverse

# ---------------------------
# APP
# ---------------------------
app = FastAPI(title="Fast Document Summarizer", version="2.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# MODEL LOAD (once)
# ---------------------------
print("Loading sentence embedder…")
embedder = SentenceTransformer(EMBED_MODEL)
print("Embedder ready.")

# ---------------------------
# DOC TYPE DETECTION & CLEANUP
# ---------------------------
DOC_LETTER = "letter"
DOC_RESUME = "resume"
DOC_SLIDES = "slides"
DOC_PAPER = "paper"
DOC_GENERIC = "document"

_BULLET_MARKS = r"[•▪●◦·\-–—•♦■✓▶→»·]"

def _detect_doc_type(pages: list[str]) -> str:
    text = "\n".join(pages[:2]).lower()
    first_page = (pages[0] if pages else "").lower()

    if first_page.startswith("dear ") or "sincerely" in text or "hiring team" in text:
        return DOC_LETTER

    resume_keys = ["experience", "education", "skills", "projects", "summary", "objective"]
    short_lines = sum(1 for ln in first_page.splitlines() if 0 < len(ln.strip()) <= 60)
    if any(k in text for k in resume_keys) and short_lines > 10:
        return DOC_RESUME

    slides_keys = ["slide", "©", "chapter", "edition", "lecture", "module", "unit"]
    bullets_density = sum(bool(re.search(_BULLET_MARKS, ln)) for ln in first_page.splitlines())
    if any(k in text for k in slides_keys) or bullets_density > 8:
        return DOC_SLIDES

    if "abstract" in text or "references" in text or ("introduction" in text and "doi" in text):
        return DOC_PAPER

    return DOC_GENERIC

def _subject_for_doc(doc_type: str, author: str | None) -> tuple[str, str]:
    if doc_type == DOC_LETTER:
        return ("the letter", author or "the applicant")
    if doc_type == DOC_RESUME:
        return ("the resume", author or "the candidate")
    if doc_type == DOC_SLIDES:
        return ("the slides", "the presenter")
    if doc_type == DOC_PAPER:
        return ("the paper", "the author")
    return ("the document", author or "the author")

def _clean_pages(pages: list[str]) -> str:
    text = "\n".join(pages)
    noise_patterns = [
        r"Operating System Concepts\s*–?\s*10(th)?\s*Edition.*",
        r"Copyright\s+20\d{2}.*",
        r"Silberschatz,?\s*Galvin\s*and\s*Gagne.*",
        r"^\s*Figure\s*\d+.*$",
        r"^\s*Table\s*\d+.*$",
        r"^\s*\d+(\.\d+)?\s*$",    # slide/page numbers
        r"Chapter\s+Introduction.*",
    ]
    for pat in noise_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)

    text = re.sub(_BULLET_MARKS, " ", text)
    text = text.replace("\r", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip()
    return text

def _smart_split_sentences(text: str) -> list[str]:
    parts = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        chunks = re.split(r"(?<=[\.\?\!])\s+", para)
        for c in chunks:
            c = c.strip()
            if c:
                parts.append(c)
    parts = [p for p in parts if len(p) > 25]
    seen = set()
    uniq = []
    for p in parts:
        key = re.sub(r"\W+", " ", p).lower().strip()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq

_VERB_HINTS = set("""
is are was were be been being has have had built led developed designed improved
manage managed manages coordinating coordinated coordinate integrate integrated
implement implemented implementing reduce reduced reducing increase increased increasing
optimize optimized optimizing analyze analyzed analyzing contribute contributed contributing
""".split())
_STOPWORDS_END = {"and", "or", "of", "to", "at", "for", "in", "on", "with", "a", "an", "the", "by"}

def _has_verbish(s: str) -> bool:
    toks = re.findall(r"[A-Za-z']+", s.lower())
    return any(t in _VERB_HINTS for t in toks) or any(t.endswith(("ed", "ing", "es", "s")) for t in toks)

def _polish_sentence(s: str) -> str | None:
    s = re.sub(rf"^{_BULLET_MARKS}\s*", "", s).strip()
    s = re.sub(r"^(and|or|but|so|also|plus|then)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*,\s*,+", ", ", s)
    s = re.sub(r"[;:]\s*(,|\.)", ". ", s)
    s = re.sub(r"\s+", " ", s).strip()
    last = re.findall(r"[A-Za-z']+$", s)
    if last and last[0].lower() in _STOPWORDS_END:
        return None
    if not s.endswith((".", "!", "?")):
        s += "."
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if len(s) < 30 or not _has_verbish(s):
        return None
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    return s.strip()

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

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def _centroid_mmr(sentences: List[str], k: int, redundancy: float) -> List[int]:
    embs = embedder.encode(
        sentences,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    centroid = embs.mean(axis=0, keepdims=True)
    centrality = (embs @ centroid.T).ravel()

    k = min(k, len(sentences))
    if k <= 0:
        return []

    selected: List[int] = []
    selected_set = set()

    first = int(np.argmax(centrality))
    selected.append(first)
    selected_set.add(first)

    for _ in range(1, k):
        best_i = None
        best_score = -1e9
        sel_idx = np.array(selected, dtype=int)
        sel_mat = embs[sel_idx, :]
        if sel_mat.ndim == 1:
            sel_mat = sel_mat.reshape(1, -1)
        for i in range(len(sentences)):
            if i in selected_set:
                continue
            c = centrality[i]
            v = embs[i:i+1, :]
            sims = (v @ sel_mat.T).ravel()
            red = float(sims.max()) if sims.size else 0.0
            score = c - MMR_REDUNDANCY * red
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
    text = "\n".join(pages[:1])
    lines = [l.strip() for l in text.splitlines() if l.strip()]
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

def _neutralize_first_person(text: str, subj: str) -> str:
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
    return out.strip()

def _pick_k_sentences(total_sents: int, length: str) -> int:
    length = (length or "auto").lower()
    if length == "short":
        k = 5
    elif length == "medium":
        k = 8
    elif length == "long":
        k = 14
    elif length in ("xlong", "xl", "verylong"):
        k = 22
    else:
        if total_sents <= 12:
            k = 6
        elif total_sents <= 50:
            k = 9
        elif total_sents <= 150:
            k = 14
        else:
            k = 20
    k = max(3, min(k, total_sents))
    return k

# --- NEW: map key points back to likely pages ---
def _map_points_to_pages(points: List[str], pages_text: List[str]) -> List[int]:
    """
    Heuristic mapping from key-point text to page index.
    """
    def tokenize(s: str) -> List[str]:
        return [t for t in re.findall(r"[a-zA-Z]{4,}", s.lower())]

    page_tokens = [set(tokenize(p)) for p in pages_text]
    out: List[int] = []
    for ptxt in points:
        toks = tokenize(ptxt)
        toks = sorted(toks, key=len, reverse=True)[:8] or toks[:4]
        best_i, best_score = 0, -1
        for i, ptoks in enumerate(page_tokens):
            hits = sum(1 for t in toks if t in ptoks)
            score = hits + (1 if hits >= 3 else 0)  # tiny density bonus
            if score > best_score:
                best_score, best_i = score, i
        out.append(best_i)
    return out

# ---------------------------
# SUMMARIZER
# ---------------------------
def summarize_extractive(pages: List[str], length: str = "auto") -> Dict[str, Any]:
    doc_type = _detect_doc_type(pages)
    author = _guess_author_name(pages)
    doc_label, subject_noun = _subject_for_doc(doc_type, author)

    raw = _clean_pages(pages)
    if not raw:
        return {"summary": "", "key_points": [], "key_point_pages": []}

    if len(raw) > MAX_CHARS:
        cut = raw[:MAX_CHARS]
        last_dot = cut.rfind(".")
        if last_dot > 200:
            cut = cut[: last_dot + 1]
        raw = cut

    sents = _smart_split_sentences(raw)
    if not sents:
        return {"summary": "", "key_points": [], "key_point_pages": []}

    if len(sents) > MAX_SENTENCES_EMBED:
        sents = sents[:MAX_SENTENCES_EMBED]

    k = _pick_k_sentences(len(sents), length)
    idxs = _centroid_mmr(sents, k=k, redundancy=MMR_REDUNDANCY)
    chosen_raw = [sents[i] for i in idxs]

    polished = []
    for s in chosen_raw:
        s = _neutralize_first_person(s, subject_noun)
        s = _polish_sentence(s)
        if s:
            polished.append(s)

    if not polished:
        return {"summary": "", "key_points": [], "key_point_pages": []}

    first = polished[0]
    needs_intro = not re.match(rf"^({re.escape(doc_label)}|{re.escape(subject_noun)})\b", first, flags=re.IGNORECASE)
    paragraph = (f"{doc_label.capitalize()} summarizes: " + " ".join(polished)) if needs_intro else " ".join(polished)
    bullets = [f"• {s}" if not s.startswith("•") else s for s in polished]

    # Map bullets to pages (use original per-page text)
    key_point_pages = _map_points_to_pages(polished, pages)

    return {"summary": paragraph, "key_points": bullets, "key_point_pages": key_point_pages}

# ---------------------------
# SCHEMAS
# ---------------------------
class UploadResponse(BaseModel):
    doc_id: str
    pages: int
    summary: str
    key_points: List[str]
    key_point_pages: List[int] = []

# ---------------------------
# ROUTES
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload", response_model=UploadResponse)
async def upload(length: str = "auto", file: UploadFile = File(...)):
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

    # Summarize
    try:
        result = summarize_extractive(pages, length=length)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    # Store raw text (optional)
    with open(os.path.join(ddir, "fulltext.txt"), "w") as f:
        f.write("\n\n".join(pages))

    return UploadResponse(
        doc_id=doc_id,
        pages=len(pages),
        summary=result["summary"],
        key_points=result["key_points"],
        key_point_pages=result.get("key_point_pages", []),
    )

@app.get("/resummarize", response_model=UploadResponse)
async def resummarize(doc_id: str, length: str = "auto"):
    ddir = os.path.join(STORAGE_DIR, doc_id)
    pdf_path = os.path.join(ddir, "document.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Document not found. Please upload again.")

    try:
        pages = _extract_pdf_text_by_page(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF read failed: {e}")

    if sum(len(p) for p in pages) == 0:
        raise HTTPException(status_code=400, detail="Stored PDF has no extractable text.")

    try:
        result = summarize_extractive(pages, length=length)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    return UploadResponse(
        doc_id=doc_id,
        pages=len(pages),
        summary=result["summary"],
        key_points=result["key_points"],
        key_point_pages=result.get("key_point_pages", []),
    )

# --- Stream the stored PDF inline (no download attachment) ---
@app.get("/pdf")
def get_pdf(doc_id: str):
    ddir = os.path.join(STORAGE_DIR, doc_id)
    pdf_path = os.path.join(ddir, "document.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Document not found.")
    # Do NOT set filename here; a filename header makes browsers download it.
    return FileResponse(pdf_path, media_type="application/pdf")
