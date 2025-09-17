import { useState, useEffect } from "react";
import { uploadPdf, resummarize, pdfUrlFor } from "./api";
import "./App.css";

function UploadIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path
        d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SparklesIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path
        d="M5 3l1.5 3L10 7.5 6.5 9 5 12l-1.5-3L0 7.5 3.5 6 5 3Zm14 6l2 4 4 2-4 2-2 4-2-4-4-2 4-2 2-4ZM9 14l1 2 2 1-2 1-1 2-1-2-2-1 2-1 1-2Z"
        fill="currentColor"
      />
    </svg>
  );
}

export default function App() {
  const [docId, setDocId] = useState<string>("");
  const [pages, setPages] = useState<number>(0);

  const [summary, setSummary] = useState<string>("");
  const [keyPoints, setKeyPoints] = useState<string[]>([]);
  const [keyPointPages, setKeyPointPages] = useState<number[]>([]); // 0-based page indexes

  const [length, setLength] = useState<
    "auto" | "short" | "medium" | "long" | "xlong"
  >("auto");

  const [uploading, setUploading] = useState(false);
  const [resumming, setResumming] = useState(false);

  // Preview sources
  const [localBlobUrl, setLocalBlobUrl] = useState<string | null>(null);
  const [serverPdfUrl, setServerPdfUrl] = useState<string | null>(null);

  // Parameters for PDF viewer
  const [viewerPage, setViewerPage] = useState<number | null>(null); // 1-based
  const [viewerSearch, setViewerSearch] = useState<string | null>(null);

  // Force the iframe to reload when we change page/search
  const [iframeKey, setIframeKey] = useState<number>(0);

  // Build the viewer URL in a stable order so browsers honor it
  function buildViewerUrl(): string | null {
    if (serverPdfUrl) {
      const params: string[] = [];
      if (viewerPage) params.push(`page=${viewerPage}`);
      // Helpful default: fit to width
      params.push("view=FitH");
      if (viewerSearch) params.push(`search=${encodeURIComponent(viewerSearch)}`);
      return `${serverPdfUrl}#${params.join("&")}`;
    }
    return localBlobUrl; // blob preview before upload response arrives
  }

  const previewUrl = buildViewerUrl();

  // cleanup blob URLs
  useEffect(() => {
    return () => {
      if (localBlobUrl) URL.revokeObjectURL(localBlobUrl);
    };
  }, [localBlobUrl]);

  function showError(e: any, fallback: string) {
    const msg = e?.response?.data?.detail || e?.message || fallback;
    alert(msg);
    console.error("Error:", e);
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files?.[0]) return;
    const file = e.target.files[0];

    // Immediate local preview
    if (localBlobUrl) URL.revokeObjectURL(localBlobUrl);
    setLocalBlobUrl(URL.createObjectURL(file));
    setServerPdfUrl(null);
    setViewerPage(null);
    setViewerSearch(null);
    setIframeKey((k) => k + 1); // ensure the blob loads fresh

    setUploading(true);
    setDocId("");
    setSummary("");
    setKeyPoints([]);
    setKeyPointPages([]);
    setPages(0);

    try {
      const res = await uploadPdf(file, length);
      setDocId(res.doc_id);
      setPages(res.pages);
      setSummary(res.summary);
      setKeyPoints(res.key_points || []);
      setKeyPointPages(res.key_point_pages || []);
      setServerPdfUrl(pdfUrlFor(res.doc_id)); // enable #page/#search
      setIframeKey((k) => k + 1);             // refresh to server URL
    } catch (err: any) {
      showError(err, "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleResummarize() {
    if (!docId) return;
    setResumming(true);
    try {
      const res = await resummarize(docId, length);
      setPages(res.pages);
      setSummary(res.summary);
      setKeyPoints(res.key_points || []);
      setKeyPointPages(res.key_point_pages || []);
      if (!serverPdfUrl) setServerPdfUrl(pdfUrlFor(docId));
      setIframeKey((k) => k + 1); // refresh if needed
    } catch (err: any) {
      showError(err, "Resummarize failed");
    } finally {
      setResumming(false);
    }
  }

  // Build a concise, “searchable” term from a key point
  function searchTermFromKeyPoint(text: string): string {
    const t = text.replace(/^•\s*/, "").trim();
    const words = t.split(/\s+/).filter((w) =>
      w.replace(/[^a-zA-Z]/g, "").length > 2
    );
    return words.slice(0, 8).join(" ");
  }

  // Jump to page & highlight in the embedded viewer
  function jumpToKeyPoint(idx: number) {
    const pageZeroBased = keyPointPages?.[idx];
    const kpText = keyPoints?.[idx] || "";
    if (pageZeroBased == null || !kpText) return;

    const oneBased = pageZeroBased + 1;
    setViewerPage(oneBased);
    setViewerSearch(searchTermFromKeyPoint(kpText));
    setIframeKey((k) => k + 1); // force iframe reload

    // Ensure the preview is in view (outer page scroll only)
    const el = document.querySelector(".preview");
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });

    // Subtle flash to signal navigation
    const p = document.querySelector(".preview");
    if (p) {
      p.classList.add("preview-flash");
      window.setTimeout(() => p.classList.remove("preview-flash"), 600);
    }
  }

  return (
    <div className="app">
      <div className="app-bg" />

      {/* Top-left logo */}
      <img
        src="src/assets/studyfireLogo.png"
        alt="Studify Logo"
        className="brand-logo"
      />

      {/* Header */}
      <div className="header">
        <div className="header-text">
          <h1 className="title">STUDYFIRE</h1>
        <p className="subtitle">Futuristic summaries for anything you upload.</p>
        </div>
      </div>

      {/* Controls Card */}
      <div className="card">
        <div className="card-head">
          <div className="badge" aria-live="polite">
            {docId ? (
              <>
                <span>📄</span>{" "}
                <span>
                  Doc ID: <strong>{docId}</strong>
                </span>
                <span style={{ marginLeft: 10, color: "var(--ok)" }}>
                  • {pages} pages
                </span>
              </>
            ) : (
              <>
                <span>📥</span> <span>Upload a PDF to begin</span>
              </>
            )}
          </div>

          {(uploading || resumming) && (
            <div className="loader" aria-label="Processing">
              <span></span>
              <span></span>
              <span></span>
            </div>
          )}
        </div>

        {/* PDF Preview Pane */}
        {previewUrl && (
          <div className="preview">
            {/* Use iframe so #page & #search work; 'key' forces reload on param change */}
            <iframe
              key={iframeKey}
              className="preview-frame"
              src={previewUrl}
              title="PDF preview"
            />
            <div className="preview-actions">
              <a
                className="open-link"
                href={serverPdfUrl ?? previewUrl}
                target="_blank"
                rel="noreferrer"
              >
                Open in new tab
              </a>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="controls">
          {/* Length selector */}
          <label className="select">
            <span>Summary length</span>
            <select
              value={length}
              onChange={(e) => setLength(e.target.value as any)}
              aria-label="Summary length"
            >
              <option value="auto">Auto</option>
              <option value="short">Short</option>
              <option value="medium">Medium</option>
              <option value="long">Long</option>
              <option value="xlong">Extra-long</option>
            </select>
          </label>

          {/* Upload */}
          <div className="file-wrap">
            <input
              id="file"
              className="file-hidden"
              type="file"
              accept="application/pdf"
              onChange={handleUpload}
              disabled={uploading}
            />
            <label htmlFor="file" className="file-label" title="Upload a PDF">
              <UploadIcon />
              <span>{uploading ? "Uploading…" : "Choose PDF"}</span>
            </label>
          </div>

          {/* Resummarize */}
          <button
            className="btn"
            onClick={handleResummarize}
            disabled={!docId || resumming || uploading}
            title={
              docId
                ? "Apply the selected length to the current document"
                : "Upload a document first"
            }
          >
            <SparklesIcon />
            {resumming ? "Resummarizing…" : "Resummarize"}
          </button>
        </div>

        {/* Results */}
        {summary && (
          <div className="results">
            <h3>Summary</h3>
            <p>{summary}</p>

            {keyPoints.length > 0 && (
              <>
                <h4>Key Points</h4>
                <ul>
                  {keyPoints.map((k, i) => (
                    <li key={i}>
                      <button
                        className="kp-link"
                        type="button"
                        onClick={() => jumpToKeyPoint(i)}
                        title={
                          keyPointPages?.[i] != null
                            ? `Highlight on page ${keyPointPages[i] + 1}`
                            : "Go to source"
                        }
                      >
                        {k}
                        {keyPointPages?.[i] != null && (
                          <span className="kp-page-pill">
                            p.{keyPointPages[i] + 1}
                          </span>
                        )}
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>

      <div className="footer">
        Pro tip: for big lecture decks, pick <strong>Long</strong> or{" "}
        <strong>Extra-long</strong>, then hit <em>Resummarize</em>.
      </div>
    </div>
  );
}
