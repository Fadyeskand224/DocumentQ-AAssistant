import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function uploadPdf(file: File): Promise<{ doc_id: string; pages: number; chunks: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await axios.post(`${API_BASE}/upload`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function askQuestion(docId: string, question: string) {
  const res = await axios.post(`${API_BASE}/qa`, { doc_id: docId, question });
  return res.data as {
    answer: string;
    confidence: number;
    citations: { page: number; chunk_id: string; text: string }[];
    retrieval_time_ms: number;
  };
}
