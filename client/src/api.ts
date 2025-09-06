import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function uploadPdf(file: File, length: string = "auto") {
  const form = new FormData();
  form.append("file", file);
  const res = await axios.post(
    `${API_BASE}/upload?length=${encodeURIComponent(length)}`,
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return res.data as {
    doc_id: string;
    pages: number;
    summary: string;
    key_points: string[];
  };
}

export async function resummarize(docId: string, length: string = "auto") {
  const res = await axios.get(
    `${API_BASE}/resummarize?doc_id=${encodeURIComponent(docId)}&length=${encodeURIComponent(length)}`
  );
  return res.data as {
    doc_id: string;
    pages: number;
    summary: string;
    key_points: string[];
  };
}
