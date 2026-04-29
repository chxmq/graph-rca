import { useState } from "react";
import { FiUploadCloud } from "react-icons/fi";
import { uploadLog, getAnalysisProgress, ApiError } from "../api";
import { useAppStore } from "../store";
import { AnalysisHistory } from "./AnalysisHistory";

export function LogUploadPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const { 
    isProcessing, 
    setProcessing, 
    addProcessingLog, 
    clearProcessingLogs, 
    setProcessingProgress,
    setLogStatus,
    analysisResult,
    setAnalysisResult,
    addToHistory,
    setResolutionResult,
  } = useAppStore();

  const severityClass = (severity?: string) => {
    const value = (severity ?? "").trim().toLowerCase();
    if (value === "critical") return "text-neon-pink";
    if (value === "high") return "text-neon-purple";
    return "text-green-100";
  };

  // Backend rejects non-UUIDv4 X-Analysis-ID values, so always emit one.
  // crypto.randomUUID is universally available in modern browsers and Node.
  const createAnalysisId = () =>
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : null;

  const handleSubmit = async () => {
    if (!file) {
      setError("Please choose a log file first.");
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setError("File too large. Maximum allowed size is 5 MB.");
      return;
    }
    setError(null);
    setAnalysisResult(null);
    setResolutionResult(null);
    setProcessing(true);
    clearProcessingLogs();
    
    addProcessingLog({
      message: `[▶] Uploading: ${file.name}`,
      type: "info",
    });

    addProcessingLog({
      message: `[◆] Backend is parsing log entries with LLM — progress will update below`,
      type: "info",
    });

    const controller = new AbortController();
    const analysisId = createAnalysisId();
    setAbortController(controller);

    // Guard against overlapping polls: if a previous poll is still in
    // flight (slow backend, transient stall), skip this tick rather than
    // double-fetching and racing the progress bar.
    let inflight = false;
    const pollProgress = async () => {
      if (inflight || !analysisId) return;
      inflight = true;
      try {
        const progress = await getAnalysisProgress(analysisId, controller.signal);
        useAppStore.getState().setProcessingProgress(progress.progress);
      } catch (progressError) {
        if (progressError instanceof ApiError && progressError.status === 404) {
          return;
        }
        if (progressError instanceof DOMException && progressError.name === "AbortError") {
          return;
        }
      } finally {
        inflight = false;
      }
    };
    const progressInterval = analysisId ? setInterval(pollProgress, 500) : null;

    try {
      void pollProgress();
      const res = await uploadLog(file, controller.signal, analysisId ?? undefined);
      
      if (progressInterval !== null) clearInterval(progressInterval);
      setProcessingProgress(100);

      addProcessingLog({ message: `[✓] Analysis complete!`, type: "success" });
      addProcessingLog({ message: `[✓] Severity: ${res.severity}`, type: "success" });
      addProcessingLog({ message: `[✓] Root cause: ${res.root_cause}`, type: "success" });
      if (res.truncated) {
        addProcessingLog({
          message: `[⚠] Log truncated to ${res.max_log_lines ?? 500} lines (of ${res.total_lines ?? "?"})`,
          type: "error",
        });
      }
      if (res.parse_errors && res.parse_errors.length > 0) {
        addProcessingLog({
          message: `[⚠] ${res.parse_errors.length} line(s) could not be parsed; analysis may be partial`,
          type: "error",
        });
      }
      
      setAnalysisResult(res);
      try {
        addToHistory(file.name, res);
      } catch (historyError) {
        const historyMsg = historyError instanceof Error ? historyError.message : "Failed to save history entry";
        addProcessingLog({ message: `[✗] History save failed: ${historyMsg}`, type: "error" });
      }
      setLogStatus({
        analysed: true,
        rootCause: res.root_cause,
        severity: res.severity,
      });
    } catch (e) {
      if (progressInterval !== null) clearInterval(progressInterval);
      const errorMsg =
        e instanceof DOMException && e.name === "AbortError"
          ? "Analysis cancelled by user."
          : e instanceof ApiError
            ? e.message
            : e instanceof Error
              ? e.message
              : "Unexpected error while analysing log file.";
      setError(errorMsg);
      addProcessingLog({ 
        message: `[✗] Error: ${errorMsg}`, 
        type: "error" 
      });
    } finally {
      setAbortController(null);
      setProcessing(false);
    }
  };

  const handleCancel = () => {
    abortController?.abort();
  };

  return (
    <div className="space-y-4">
      {/* Analysis History */}
      <AnalysisHistory />
      
      <div className="flex items-center justify-between gap-4 border-b border-green-500/30 pb-4">
        <div>
          <h2 className="text-xl font-bold text-green-50 flex items-center gap-3 uppercase tracking-wider">
            <FiUploadCloud className="text-green-500 animate-pulse" />
            <span className="terminal-text">Log Ingestion</span>
          </h2>
          <p className="text-xs text-green-400 mt-2 font-mono">
            &gt; Supported formats: <span className="text-green-300">.log</span> | <span className="text-green-300">.txt</span>
            <br />
            &gt; Parser will extract events and build causal graph
          </p>
        </div>
        <button
          className="secondary-button text-xs"
          type="button"
          onClick={() => {
            abortController?.abort();
            setAbortController(null);
            setFile(null);
            setAnalysisResult(null);
            setResolutionResult(null);
            setError(null);
            clearProcessingLogs();
            setProcessingProgress(0);
            setLogStatus({ analysed: false, rootCause: "", severity: "" });
          }}
          disabled={isProcessing}
        >
          [RESET]
        </button>
      </div>

      {!isProcessing && (
      <label className="flex flex-col items-center justify-center border-2 border-dashed border-green-500/30 hover:border-green-500 transition rounded-none px-6 py-10 cursor-pointer bg-black/40 group">
        <input
          type="file"
          accept=".log,.txt"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) setFile(f);
          }}
          disabled={isProcessing}
        />
        <FiUploadCloud className="text-4xl text-green-500 mb-4 group-hover:animate-pulse" />
        <p className="text-sm text-green-100 font-bold font-mono uppercase tracking-wide">
          {file ? `[ ${file.name} ]` : "[ DRAG LOG FILE OR CLICK ]"}
        </p>
        <p className="text-xs text-green-500 mt-2 font-mono">
          &gt; Local processing | Zero data transmission
        </p>
      </label>
      )}

      <ProcessingView />

      <div className="flex justify-end gap-3 items-center">
        {isProcessing && (
          <div className="flex items-center gap-2 text-green-500 font-mono text-sm">
            <div className="h-4 w-4 border-2 border-green-500 border-t-transparent loading-spinner" />
            <span className="animate-pulse">ANALYSING...</span>
          </div>
        )}
        <button
          className="primary-button"
          type="button"
          onClick={handleSubmit}
          disabled={isProcessing || !file}
        >
          {isProcessing ? "[PROCESSING]" : "[ANALYSE LOG]"}
        </button>
        {isProcessing && (
          <button className="secondary-button" type="button" onClick={handleCancel}>
            [CANCEL]
          </button>
        )}
      </div>

      {error && (
        <div className="border-2 border-neon-pink bg-black px-4 py-3 text-xs text-neon-pink font-mono">
          <span className="font-bold">ERROR:</span> {error}
        </div>
      )}

      {analysisResult && (analysisResult.truncated || (analysisResult.parse_errors?.length ?? 0) > 0) && (
        <div className="border-2 border-yellow-500/60 bg-black px-4 py-3 text-xs text-yellow-300 font-mono">
          <div className="font-bold uppercase mb-1">&gt; Partial analysis</div>
          {analysisResult.truncated && (
            <p>
              Log truncated to {analysisResult.max_log_lines ?? 500} of{" "}
              {analysisResult.total_lines ?? "?"} lines. Re-run with a smaller window
              if the root cause is past the cutoff.
            </p>
          )}
          {(analysisResult.parse_errors?.length ?? 0) > 0 && (
            <details className="mt-1">
              <summary className="cursor-pointer">
                {analysisResult.parse_errors!.length} line(s) failed to parse — click to expand
              </summary>
              <ul className="mt-2 space-y-1 max-h-40 overflow-auto pl-4">
                {analysisResult.parse_errors!.slice(0, 50).map((err, idx) => (
                  <li key={idx} className="text-yellow-400">
                    &gt; {err}
                  </li>
                ))}
                {analysisResult.parse_errors!.length > 50 && (
                  <li className="text-yellow-600">
                    … and {analysisResult.parse_errors!.length - 50} more
                  </li>
                )}
              </ul>
            </details>
          )}
        </div>
      )}

      {analysisResult && (
        <div className="glass-panel p-6 space-y-6 animate-in">
          <div className="grid gap-6 md:grid-cols-3">
            <div className="md:col-span-1 space-y-4">
              <h3 className="text-sm font-bold text-green-50 uppercase tracking-wider border-b border-green-500/30 pb-2">
                [STATUS REPORT]
              </h3>
              <div className="space-y-3 text-xs text-green-300 font-mono">
                <div>
                  <span className="text-green-500">&gt; SEVERITY:</span>{" "}
                  <span className={`font-bold ${
                    severityClass(analysisResult.severity)
                  }`}>
                    {analysisResult.severity ?? "UNKNOWN"}
                  </span>
                </div>
                <div>
                  <span className="text-green-500">&gt; ROOT CAUSE:</span>{" "}
                  <span className="text-green-100">{analysisResult.root_cause ?? "NOT IDENTIFIED"}</span>
                </div>
              </div>
            </div>
            <div className="md:col-span-2 space-y-4">
              <h3 className="text-sm font-bold text-green-50 uppercase tracking-wider border-b border-green-500/30 pb-2">
                [ANALYSIS OUTPUT]
              </h3>
              <div className="space-y-2">
                {analysisResult.summary?.map((line, idx) => (
                  <div key={`${line}-${idx}`} className="text-xs text-green-300 font-mono flex gap-2">
                    <span className="text-green-500">&gt;</span>
                    <span>{line}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ProcessingView() {
  const { isProcessing, processingLogs, processingProgress } = useAppStore();

  if (!isProcessing && processingLogs.length === 0) {
    return null;
  }

  return (
    <div className="glass-panel p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-green-50 uppercase tracking-wider">
          [PROCESSING STATUS]
        </h3>
        <span className="text-xs text-green-400 font-mono">
          {processingProgress}%
        </span>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-black border border-green-500/30 h-3 relative overflow-hidden">
        <div
          className="h-full bg-green-500 transition-all duration-300 relative"
          style={{ width: `${processingProgress}%` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-green-300/50 to-transparent animate-scan" />
        </div>
      </div>

      {/* Log Output */}
      <div aria-live="polite" className="bg-black border border-green-500/30 p-4 max-h-48 overflow-y-auto font-mono text-xs space-y-2">
        {processingLogs.map((log, idx) => (
          <div
            key={`${log.timestamp}-${idx}`}
            className={`p-2 border-l-2 ${
              log.type === "error"
                ? "text-red-400 border-red-500 bg-red-950/20"
                : log.type === "success"
                ? "text-green-400 border-green-500 bg-green-950/20"
                : "text-green-400 border-green-700 bg-green-950/10"
            }`}
          >
            <span>{log.message}</span>
          </div>
        ))}
        {isProcessing && (
          <div className="flex gap-1 text-green-400 animate-pulse pt-2">
            <span>&gt;</span>
            <span className="animate-pulse">_</span>
          </div>
        )}
      </div>
    </div>
  );
}

