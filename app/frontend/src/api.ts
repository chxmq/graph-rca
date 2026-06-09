// Thin API client. Adjust URLs to match your Python backend when you add HTTP endpoints.

export interface LogAnalysisResponse {
  analysis_id?: string;
  severity: string;
  root_cause: string;
  summary: string[];
  context: Record<string, unknown>;
  root_cause_expln: string;
  summary_parse_failed?: boolean;
  parse_errors?: string[];
  parsed_lines?: number;
  total_lines?: number;
  truncated?: boolean;
  max_log_lines?: number;
}

export interface AnalysisProgressResponse {
  status: "processing" | "completed" | "failed";
  processed_batches?: number;
  total_batches?: number;
  progress: number;
  parse_errors?: string[];
  parsed_lines?: number;
  total_lines?: number;
}

export interface DocsUploadResponse {
  count: number;
}

export interface IncidentResolutionResponse {
  root_cause: string;
  solution: string;
  sources: string[];
}

export type ApiErrorBody = {
  detail?: string;
  error?: string;
  [key: string]: unknown;
};

export class ApiError extends Error {
  status: number;
  body: ApiErrorBody;
  constructor(message: string, status: number, body: ApiErrorBody) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function parseError(res: Response, fallback: string): Promise<never> {
  const body = await res.json().catch(() => ({} as ApiErrorBody));
  const detail = body.detail ?? body.error ?? fallback;
  throw new ApiError(detail, res.status, body);
}

export async function uploadLog(
  file: File,
  signal?: AbortSignal,
  analysisId?: string
): Promise<LogAnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/api/log/analyse", {
    method: "POST",
    headers: analysisId ? { "X-Analysis-ID": analysisId } : undefined,
    body: formData,
    signal,
  });

  if (!res.ok) {
    await parseError(res, "Failed to analyse log");
  }
  return res.json();
}

export async function getAnalysisProgress(
  analysisId: string,
  signal?: AbortSignal
): Promise<AnalysisProgressResponse> {
  const res = await fetch(`/api/analysis/${encodeURIComponent(analysisId)}/progress`, {
    signal,
  });

  if (!res.ok) {
    await parseError(res, "Failed to load analysis progress");
  }
  return res.json();
}

export async function getAnalysisContext(
  analysisId: string,
  signal?: AbortSignal
): Promise<Record<string, unknown>> {
  const res = await fetch(`/api/analysis/${encodeURIComponent(analysisId)}/context`, {
    signal,
  });

  if (!res.ok) {
    await parseError(res, "Failed to load analysis context");
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
    await parseError(res, "Failed to upload documentation");
  }
  return res.json();
}

export async function runIncidentResolution(
  context: Record<string, unknown>,
  rootCauseExpln: string,
  signal?: AbortSignal
): Promise<IncidentResolutionResponse> {
  const res = await fetch("/api/incident/resolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context, root_cause_expln: rootCauseExpln }),
    signal,
  });

  if (!res.ok) {
    await parseError(res, "Failed to generate incident resolution");
  }
  return res.json();
}

