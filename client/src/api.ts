import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

/**
 * Upload a PDF and get back doc_id, pages, summary, key_points, and (optionally) key_point_pages.
 */
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
    /** 0-based page index per key point; older servers may not return this */
    key_point_pages?: number[];
  };
}

/**
 * Re-run summarization with a new length for an existing doc.
 */
export async function resummarize(docId: string, length: string = "auto") {
  const res = await axios.get(
    `${API_BASE}/resummarize?doc_id=${encodeURIComponent(
      docId
    )}&length=${encodeURIComponent(length)}`
  );
  return res.data as {
    doc_id: string;
    pages: number;
    summary: string;
    key_points: string[];
    /** 0-based page index per key point; older servers may not return this */
    key_point_pages?: number[];
  };
}

/**
 * Build the inline-viewer URL for the stored PDF.
 * (Works with the backend /pdf route that streams inline.)
 */
export function pdfUrlFor(docId: string) {
  return `${API_BASE}/pdf?doc_id=${encodeURIComponent(docId)}`;
}
