import { useState } from "react";
import { FiZap, FiAlertTriangle } from "react-icons/fi";
import { runIncidentResolution } from "../api";
import { useAppStore } from "../store";

export function IncidentResolutionPanel() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    root_cause: string;
    solution: string;
    sources: string[];
  } | null>(null);

  const { logStatus, docsStatus } = useAppStore();

  const handleRun = async () => {
    setError(null);
    setLoading(true);
    try {
      const res = await runIncidentResolution();
      setResult(res);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Unexpected error while generating incident resolution.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <h2 className="text-lg font-semibold text-slate-50 flex items-center gap-2">
            <FiZap className="text-brand-400" />
            <span>Automatic incident resolution</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            We combine the analysed causal chain from your logs with any uploaded
            documentation to propose a concrete set of remediation steps.
          </p>
        </div>
        
        {/* Documentation Status Badge */}
        {docsStatus.uploaded ? (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-green-50 border border-green-200 rounded-lg">
            <div className="h-2 w-2 bg-green-500 rounded-full"></div>
            <span className="text-xs font-medium text-green-700">Docs Loaded</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-50 border border-yellow-200 rounded-lg">
            <FiAlertTriangle className="h-3 w-3 text-yellow-600" />
            <span className="text-xs font-medium text-yellow-700">No Docs</span>
          </div>
        )}
        
        <button
          className="primary-button"
          type="button"
          onClick={handleRun}
          disabled={loading || !logStatus.analysed}
        >
          {loading ? "Generating…" : "Run resolution"}
        </button>
      </div>

      {!logStatus.analysed && (
        <div className="rounded-xl border border-amber-500/70 bg-amber-950/40 px-3 py-2 text-xs text-amber-100 flex items-start gap-2">
          <FiAlertTriangle className="mt-0.5" />
          <div>
            <p className="font-semibold">Log analysis required first</p>
            <p>
              Please analyse a log file in step 1 before running the incident
              resolution.
            </p>
          </div>
        </div>
      )}

      {docsStatus.uploaded === false && (
        <p className="text-xs text-slate-400">
          You can still run without documentation, but adding runbooks usually makes
          solutions more actionable.
        </p>
      )}

      {error && (
        <div className="rounded-xl border border-red-500/60 bg-red-950/40 px-3 py-2 text-xs text-red-200">
          {error}
        </div>
      )}

      {result && (
        <div className="grid gap-4 md:grid-cols-3">
          <div className="md:col-span-1 space-y-2">
            <h3 className="text-sm font-semibold text-slate-100">
              Root cause (from analysis)
            </h3>
            <p className="text-xs text-slate-300 whitespace-pre-wrap">
              {result.root_cause}
            </p>
          </div>
          <div className="md:col-span-2 space-y-3">
            <h3 className="text-sm font-semibold text-slate-100">
              Recommended resolution
            </h3>
            <div className="text-xs text-slate-200 bg-slate-950/60 rounded-xl p-3 space-y-2 whitespace-pre-wrap">
              {result.solution.split("\n\n").map((block, idx) => (
                <p key={idx}>{block}</p>
              ))}
            </div>
            {result.sources?.length > 0 && (
              <div className="space-y-1">
                <h4 className="text-xs font-semibold text-slate-300">
                  Referenced documents
                </h4>
                <ul className="list-disc list-inside text-xs text-slate-400">
                  {result.sources.map((s, idx) => (
                    <li key={idx}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

