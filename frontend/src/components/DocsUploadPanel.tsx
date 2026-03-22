import { useState } from "react";
import { FiBookOpen, FiUploadCloud } from "react-icons/fi";
import { uploadDocs } from "../api";
import { useAppStore } from "../store";

export function DocsUploadPanel() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const setDocsStatus = useAppStore((s) => s.setDocsStatus);

  const handleSubmit = async () => {
    if (!files.length) {
      setError("Please choose at least one documentation file.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await uploadDocs(files);
      setDocsStatus({ uploaded: true, count: res.count });

      // Load preview from first file locally
      const first = files[0];
      const text = await first.text();
      setPreview(text.slice(0, 800));
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Unexpected error while uploading documentation.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-50 flex items-center gap-2">
            <FiBookOpen className="text-brand-400" />
            <span>Add documentation</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Upload runbooks, incident guides or service docs as{" "}
            <span className="font-mono">.txt</span> or{" "}
            <span className="font-mono">.md</span>. These are used as context when we
            generate a fix.
          </p>
        </div>
        <button
          className="secondary-button text-xs"
          type="button"
          onClick={() => {
            setFiles([]);
            setPreview(null);
            setError(null);
          }}
        >
          Reset
        </button>
      </div>

      <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 hover:border-brand-500/60 transition rounded-2xl px-6 py-8 cursor-pointer bg-slate-950/40">
        <input
          type="file"
          accept=".txt,.md"
          multiple
          className="hidden"
          onChange={(e) => {
            const list = e.target.files;
            if (!list) return;
            setFiles(Array.from(list));
          }}
        />
        <FiUploadCloud className="text-3xl text-slate-400 mb-3" />
        <p className="text-sm text-slate-100 font-medium">
          {files.length
            ? `${files.length} file${files.length > 1 ? "s" : ""} selected`
            : "Drop docs here or click to browse"}
        </p>
        <p className="text-xs text-slate-400 mt-1">
          Try including SLOs, alert runbooks and architecture overviews for richer
          answers.
        </p>
      </label>

      <div className="flex justify-end">
        <button
          className="primary-button"
          type="button"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? "Indexing…" : "Index documentation"}
        </button>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/60 bg-red-950/40 px-3 py-2 text-xs text-red-200">
          {error}
        </div>
      )}

      {preview && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-slate-100">
            Preview from first document
          </h3>
          <pre className="text-xs text-slate-200 bg-slate-950/60 rounded-xl p-3 max-h-60 overflow-auto whitespace-pre-wrap">
            {preview}
          </pre>
        </div>
      )}
    </div>
  );
}

