// Thin API client. Adjust URLs to match your Python backend when you add HTTP endpoints.

export interface LogAnalysisResponse {
  severity: string;
  root_cause: string;
  summary: string[];
}

export interface DocsUploadResponse {
  count: number;
}

export interface IncidentResolutionResponse {
  root_cause: string;
  solution: string;
  sources: string[];
}

export async function uploadLog(file: File): Promise<LogAnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/api/log/analyse", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? "Failed to analyse log");
  }
  return res.json();
}

export async function uploadDocs(files: File[]): Promise<DocsUploadResponse> {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));

  const res = await fetch("/api/docs/upload", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? "Failed to upload documentation");
  }
  return res.json();
}

export async function runIncidentResolution(
  context: Record<string, unknown>,
  rootCauseExpln: string
): Promise<IncidentResolutionResponse> {
  const res = await fetch("/api/incident/resolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ _context: context, _root_cause_expln: rootCauseExpln }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? "Failed to generate incident resolution");
  }
  return res.json();
}

