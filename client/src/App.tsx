import { useState } from "react";
import { uploadPdf } from "./api";

function App() {
  const [docId, setDocId] = useState<string>("");
  const [uploading, setUploading] = useState(false);

  const [pages, setPages] = useState<number>(0);
  const [summary, setSummary] = useState<string>("");
  const [keyPoints, setKeyPoints] = useState<string[]>([]);

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
      const res = await uploadPdf(e.target.files[0]);
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

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", padding: "0 1rem", fontFamily: "Inter, system-ui, Arial" }}>
      <h1 style={{ fontSize: 28, marginBottom: 8 }}>Document Summarizer</h1>
      <p style={{ color: "#555", marginBottom: 24 }}>
        Upload a PDF to get a concise summary and key points.
      </p>

      <div style={{
        display: "flex",
        gap: 16,
        alignItems: "center",
        padding: 16,
        border: "1px solid #eee",
        borderRadius: 12,
        marginBottom: 16
      }}>
        <input type="file" accept="application/pdf" onChange={handleUpload} disabled={uploading} />
        {uploading && <span>Processing…</span>}
        {docId && <span style={{ fontSize: 14, color: "#0a0" }}>Doc ID: {docId} • {pages} pages</span>}
      </div>

      {summary && (
        <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 16, marginBottom: 16 }}>
          <h3 style={{ marginTop: 0, marginBottom: 8 }}>Summary</h3>
          <p style={{ whiteSpace: "pre-wrap", marginTop: 0 }}>{summary}</p>

          {keyPoints.length > 0 && (
            <>
              <h4 style={{ marginTop: 16 }}>Key Points</h4>
              <ul>
                {keyPoints.map((k, i) => (<li key={i}>{k}</li>))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
