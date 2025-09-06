import { useState } from "react";
import { uploadPdf, resummarize } from "./api";
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

  const [length, setLength] = useState<
    "auto" | "short" | "medium" | "long" | "xlong"
  >("auto");

  const [uploading, setUploading] = useState(false);
  const [resumming, setResumming] = useState(false);

  function showError(e: any, fallback: string) {
    const msg = e?.response?.data?.detail || e?.message || fallback;
    alert(msg);
    console.error("Error:", e);
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files?.[0]) return;
    setUploading(true);
    setDocId("");
    setSummary("");
    setKeyPoints([]);
    setPages(0);

    try {
      const res = await uploadPdf(e.target.files[0], length);
      setDocId(res.doc_id);
      setPages(res.pages);
      setSummary(res.summary);
      setKeyPoints(res.key_points || []);
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
    } catch (err: any) {
      showError(err, "Resummarize failed");
    } finally {
      setResumming(false);
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
          <p className="subtitle">
            Futuristic summaries for anything you upload.
          </p>
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
                    <li key={i}>{k}</li>
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
