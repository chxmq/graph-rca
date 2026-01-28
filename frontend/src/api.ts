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

  if (!res.ok) throw new Error("Failed to analyse log");
  return res.json();
}

export async function uploadDocs(files: File[]): Promise<DocsUploadResponse> {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));

  const res = await fetch("/api/docs/upload", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Failed to upload documentation");
  return res.json();
}

export async function runIncidentResolution(): Promise<IncidentResolutionResponse> {
  const res = await fetch("/api/incident/resolve", {
    method: "POST",
  });

  if (!res.ok) throw new Error("Failed to generate incident resolution");
  return res.json();
}

