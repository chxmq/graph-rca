import { useState } from "react";
import { FiClock, FiChevronDown, FiChevronUp, FiTrash2 } from "react-icons/fi";
import { getAnalysisContext, ApiError } from "../api";
import { useAppStore } from "../store";

export function AnalysisHistory() {
  const severityBadge = (severity?: string) => {
    const value = (severity ?? "").trim().toLowerCase();
    if (value === "critical") return "text-red-400 border-red-500/50";
    if (value === "high") return "text-yellow-400 border-yellow-500/50";
    return "text-green-400 border-green-500/50";
  };

  const [isExpanded, setIsExpanded] = useState(false);
  const { analysisHistory, loadFromHistory, clearHistory, mergeAnalysisResult } = useAppStore();

  // Pre-fetch saved context immediately on history click so the resolve
  // panel doesn't show a "missing context" flash before its own fetch
  // resolves. Failures are silent — IncidentResolutionPanel re-fetches
  // on demand if the prefetch didn't land.
  const handleLoad = async (id: string, analysisId: string | undefined) => {
    loadFromHistory(id);
    if (!analysisId) return;
    try {
      const context = await getAnalysisContext(analysisId);
      mergeAnalysisResult({ context, loadedFromHistory: false });
    } catch (err) {
      // 404 is expected when the backend's TTL expired the context;
      // anything else we leave for IncidentResolutionPanel to surface.
      if (!(err instanceof ApiError && err.status === 404)) {
        console.warn("[AnalysisHistory] context prefetch failed", err);
      }
    }
  };

  if (analysisHistory.length === 0) {
    return null;
  }

  return (
    <div className="glass-panel p-4">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          <FiClock className="text-green-500" />
          <h3 className="text-sm font-bold text-green-50 uppercase tracking-wider">
            [ANALYSIS HISTORY]
          </h3>
          <span className="text-xs text-green-500 font-mono">
            ({analysisHistory.length})
          </span>
        </div>
        {isExpanded ? (
          <FiChevronUp className="text-green-500" />
        ) : (
          <FiChevronDown className="text-green-500" />
        )}
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-2">
          <div className="flex justify-end mb-2">
            <button
              onClick={clearHistory}
              className="flex items-center gap-1 px-2 py-1 text-xs text-red-400 hover:text-red-300 border border-red-500/30 hover:border-red-500 transition"
            >
              <FiTrash2 className="h-3 w-3" />
              <span>Clear All</span>
            </button>
          </div>

          <div className="space-y-2 max-h-64 overflow-y-auto">
            {analysisHistory.map((entry) => (
              <button
                key={entry.id}
                onClick={() => void handleLoad(entry.id, entry.analysis_id)}
                className="w-full text-left p-3 bg-black/40 border border-green-500/20 hover:border-green-500/50 transition group"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-mono text-green-300 truncate">
                        {entry.fileName}
                      </span>
                      <span
                        className={`text-xs px-1.5 py-0.5 border ${severityBadge(entry.severity)}`}
                      >
                        {entry.severity}
                      </span>
                    </div>
                    <p className="text-xs text-green-500 line-clamp-2 font-mono">
                      {entry.root_cause}
                    </p>
                  </div>
                  <div className="text-xs text-green-700 font-mono whitespace-nowrap">
                    {new Date(entry.timestamp).toLocaleString()}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
