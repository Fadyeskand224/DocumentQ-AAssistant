import { useState } from "react";
import { uploadPdf, askQuestion } from "./api";

function App() {
  const [docId, setDocId] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string>("");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [citations, setCitations] = useState<{ page: number; chunk_id: string; text: string }[]>([]);
  const [latency, setLatency] = useState<number | null>(null);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files?.[0]) return;
    setUploading(true);
    setAnswer("");
    setCitations([]);
    try {
      const res = await uploadPdf(e.target.files[0]);
      setDocId(res.doc_id);
    } catch (err: any) {
      alert(err?.response?.data?.detail || "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleAsk() {
    if (!docId) {
      alert("Upload a PDF first.");
      return;
    }
    if (!question.trim()) return;
    setAnswer("Thinking...");
    setCitations([]);
    setConfidence(null);
    setLatency(null);
    try {
      const t0 = performance.now();
      const res = await askQuestion(docId, question);
      const t1 = performance.now();
      setAnswer(res.answer);
      setConfidence(res.confidence);
      setCitations(res.citations);
      setLatency(res.retrieval_time_ms ?? Math.round(t1 - t0));
    } catch (err: any) {
      setAnswer("");
      alert(err?.response?.data?.detail || "QA failed");
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", padding: "0 1rem", fontFamily: "Inter, system-ui, Arial" }}>
      <h1 style={{ fontSize: 28, marginBottom: 8 }}>Semantic Document Q&A Assistant</h1>
      <p style={{ color: "#555", marginBottom: 24 }}>
        Upload a PDF, then ask grounded, cited questions about its content.
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
        {uploading && <span>Uploading & indexing…</span>}
        {docId && <span style={{ fontSize: 14, color: "#0a0" }}>Doc ID: {docId}</span>}
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr auto",
        gap: 8,
        alignItems: "center",
        marginBottom: 16
      }}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your document…"
          onKeyDown={(e) => (e.key === "Enter" ? handleAsk() : null)}
          style={{
            padding: "12px 14px",
            borderRadius: 10,
            border: "1px solid #ddd",
            fontSize: 16
          }}
        />
        <button
          onClick={handleAsk}
          style={{
            padding: "12px 20px",
            borderRadius: 10,
            border: "1px solid #222",
            background: "#111",
            color: "#fff",
            fontWeight: 600,
            cursor: "pointer"
          }}
        >
          Ask
        </button>
      </div>

      {answer && (
        <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 16 }}>
          <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 8 }}>
            <h3 style={{ margin: 0 }}>Answer</h3>
            {confidence !== null && (
              <span style={{
                fontSize: 12,
                color: confidence >= 0.6 ? "#0a0" : confidence >= 0.35 ? "#c80" : "#a00",
                border: "1px solid #ddd",
                padding: "2px 8px",
                borderRadius: 999
              }}>
                confidence: {confidence.toFixed(2)}
              </span>
            )}
            {latency !== null && (
              <span style={{ fontSize: 12, color: "#555" }}>
                • {latency} ms
              </span>
            )}
          </div>
          <p style={{ whiteSpace: "pre-wrap" }}>{answer}</p>

          {citations.length > 0 && (
            <>
              <h4 style={{ marginTop: 16 }}>Citations</h4>
              <ul>
                {citations.map((c, i) => (
                  <li key={i}>
                    <strong>Page {c.page}</strong> — <code>{c.chunk_id}</code>
                    <div style={{
                      background: "#fafafa",
                      border: "1px solid #eee",
                      padding: 8,
                      borderRadius: 8,
                      marginTop: 6,
                      fontSize: 14,
                      color: "#333"
                    }}>
                      {c.text}…
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
