import { useState } from "react";
import { FiZap, FiAlertTriangle } from "react-icons/fi";
import ReactMarkdown from "react-markdown";
import { getAnalysisContext, runIncidentResolution } from "../api";
import { useAppStore } from "../store";

function hasResolutionContext(context: Record<string, unknown> | undefined): context is Record<string, unknown> {
  return Boolean(
    context &&
      typeof context.dag_id === "string" &&
      typeof context.root_cause === "string" &&
      Array.isArray(context.causal_chain)
  );
}

export function IncidentResolutionPanel() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const {
    logStatus,
    docsStatus,
    analysisResult,
    resolutionResult,
    setResolutionResult,
    mergeAnalysisResult,
  } = useAppStore();
  const contextLoadedFromHistory =
    Boolean(analysisResult?.loadedFromHistory) && !hasResolutionContext(analysisResult?.context);
  const canFetchHistoryContext = contextLoadedFromHistory && Boolean(analysisResult?.analysis_id);

  const handleRun = async () => {
    setError(null);
    setLoading(true);
    const controller = new AbortController();
    setAbortController(controller);
    try {
      if (!analysisResult?.root_cause_expln) {
        throw new Error(
          "Missing analysis context. Please re-analyse a log file first.",
        );
      }
      let context = analysisResult.context;
      if (!hasResolutionContext(context) && analysisResult.analysis_id) {
        context = await getAnalysisContext(analysisResult.analysis_id, controller.signal);
        mergeAnalysisResult({ context, loadedFromHistory: false });
      }
      if (!hasResolutionContext(context)) {
        throw new Error(
          "This history entry does not include resolution context. Re-analyse the log file first.",
        );
      }
      const res = await runIncidentResolution(
        context,
        analysisResult.root_cause_expln,
        controller.signal,
      );
      setResolutionResult(res);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Unexpected error while generating incident resolution.",
      );
    } finally {
      setAbortController(null);
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4 border-b border-green-500/30 pb-4">
        <div>
          <h2 className="text-xl font-bold text-green-50 flex items-center gap-3 uppercase tracking-wider">
            <FiZap className="text-green-500 animate-pulse" />
            <span className="terminal-text">Incident Resolution</span>
          </h2>
          <p className="text-xs text-green-400 mt-2 font-mono">
            &gt; Combines causal chain + documentation to generate remediation
            steps
          </p>
        </div>

        {/* Documentation Status Badge */}
        {docsStatus.uploaded ? (
          <div className="pill">
            <span className="h-2 w-2 bg-green-500 animate-pulse" />
            <span>DOCS LOADED</span>
          </div>
        ) : (
          <div className="pill">
            <FiAlertTriangle className="text-yellow-500" />
            <span className="text-yellow-400">NO DOCS</span>
          </div>
        )}
      </div>

      {!logStatus.analysed && (
        <div className="border-2 border-yellow-500/50 bg-black px-4 py-3 text-xs text-yellow-400 font-mono flex items-start gap-2">
          <FiAlertTriangle className="mt-0.5 shrink-0" />
          <div>
            <p className="font-bold uppercase">&gt; Log analysis required</p>
            <p>
              Analyse a log file in step 1 before running incident resolution.
            </p>
          </div>
        </div>
      )}

      {docsStatus.uploaded === false && (
        <p className="text-xs text-green-600 font-mono">
          &gt; No docs uploaded — resolution still works, but runbooks improve
          accuracy.
        </p>
      )}

      {contextLoadedFromHistory && (
        <div className="border-2 border-yellow-500/50 bg-black px-4 py-3 text-xs text-yellow-400 font-mono flex items-start gap-2">
          <FiAlertTriangle className="mt-0.5 shrink-0" />
          <div>
            <p className="font-bold uppercase">&gt; Loaded from history</p>
            <p>
              {canFetchHistoryContext
                ? "Resolution will fetch the saved backend context before running."
                : "This entry has no saved context ID. Re-analyse the log before running resolution."}
            </p>
          </div>
        </div>
      )}

      <div className="flex justify-end gap-3 items-center">
        {loading && (
          <div className="flex items-center gap-2 text-green-500 font-mono text-sm">
            <div className="h-4 w-4 border-2 border-green-500 border-t-transparent loading-spinner" />
            <span className="animate-pulse">RESOLVING...</span>
          </div>
        )}
        <button
          className="primary-button"
          type="button"
          onClick={handleRun}
          disabled={loading || !logStatus.analysed || (contextLoadedFromHistory && !canFetchHistoryContext)}
        >
          {loading ? "[PROCESSING]" : "[RUN RESOLUTION]"}
        </button>
        {loading && (
          <div className="flex flex-col items-start gap-1">
            <button className="secondary-button" type="button" onClick={() => abortController?.abort()}>
              [CANCEL]
            </button>
            <span className="text-[10px] text-gray-500 font-mono">
              Cancels the UI request only — the backend LLM call continues until it finishes.
            </span>
          </div>
        )}
      </div>

      {error && (
        <div className="border-2 border-neon-pink bg-black px-4 py-3 text-xs text-neon-pink font-mono">
          <span className="font-bold">ERROR:</span> {error}
        </div>
      )}

      {resolutionResult && (
        <div className="glass-panel p-6 space-y-6 animate-in">
          <div className="grid gap-6 md:grid-cols-3">
            <div className="md:col-span-1 space-y-4">
              <h3 className="text-sm font-bold text-green-50 uppercase tracking-wider border-b border-green-500/30 pb-2">
                [ROOT CAUSE]
              </h3>
              <p className="text-xs text-green-300 whitespace-pre-wrap font-mono">
                <span className="text-green-500">&gt;</span> {resolutionResult.root_cause}
              </p>
            </div>
            <div className="md:col-span-2 space-y-4">
              <h3 className="text-sm font-bold text-green-50 uppercase tracking-wider border-b border-green-500/30 pb-2">
                [REMEDIATION STEPS]
              </h3>
              <div className="text-xs text-green-300 bg-black border border-green-500/30 p-4 font-mono prose prose-invert prose-sm max-w-none">
                <ReactMarkdown>{resolutionResult.solution}</ReactMarkdown>
              </div>
              {resolutionResult.sources?.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-green-50 uppercase tracking-wider">
                    [REFERENCED DOCS]
                  </h4>
                  <ul className="space-y-1">
                    {resolutionResult.sources.map((s, idx) => (
                      <li
                        key={`${s}-${idx}`}
                        className="text-xs text-green-400 font-mono"
                      >
                        <span className="text-green-600">├─</span> {s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
