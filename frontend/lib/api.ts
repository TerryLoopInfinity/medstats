import type { AnalysisMethod, AnalysisResult, UploadResponse } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ────────────────────────────────────────────────────────────
// Upload
// ────────────────────────────────────────────────────────────

export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/api/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `上传失败 (${res.status})`);
  }

  return res.json() as Promise<UploadResponse>;
}

// ────────────────────────────────────────────────────────────
// Analyze
// ────────────────────────────────────────────────────────────

export async function analyze(
  method: AnalysisMethod,
  fileId: string,
  params: Record<string, unknown> = {}
): Promise<AnalysisResult> {
  const res = await fetch(`${BASE_URL}/api/analyze/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId, params }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `分析失败 (${res.status})`);
  }

  return res.json() as Promise<AnalysisResult>;
}
