import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function uploadPdf(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await axios.post(`${API_BASE}/upload`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data as {
    doc_id: string;
    pages: number;
    summary: string;
    key_points: string[];
  };
}
