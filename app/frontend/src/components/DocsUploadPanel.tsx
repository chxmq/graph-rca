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
    const oversize = files.find((f) => f.size > 5 * 1024 * 1024);
    if (oversize) {
      setError(`File '${oversize.name}' is too large. Maximum allowed size is 5 MB.`);
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await uploadDocs(files);
      setDocsStatus({ uploaded: true, count: res.count });

      // Load preview from first file locally
      const first = files[0];
      if (!first) {
        setPreview(null);
        return;
      }
      const text = await first.text();
      const prefix =
        files.length > 1
          ? `[Previewing file 1/${files.length}: ${first.name}]\n\n`
          : `[Previewing: ${first.name}]\n\n`;
      setPreview(`${prefix}${text.slice(0, 800)}`);
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
      <div className="flex items-center justify-between gap-4 border-b border-green-500/30 pb-4">
        <div>
          <h2 className="text-xl font-bold text-green-50 flex items-center gap-3 uppercase tracking-wider">
            <FiBookOpen className="text-green-500 animate-pulse" />
            <span className="terminal-text">Documentation Ingestion</span>
          </h2>
          <p className="text-xs text-green-400 mt-2 font-mono">
            &gt; Upload runbooks, incident guides, or service docs as{" "}
            <span className="text-green-300">.txt</span> |{" "}
            <span className="text-green-300">.md</span>
            <br />
            &gt; Used as context for resolution generation
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
          [RESET]
        </button>
      </div>

      <label className="flex flex-col items-center justify-center border-2 border-dashed border-green-500/30 hover:border-green-500 transition rounded-none px-6 py-10 cursor-pointer bg-black/40 group">
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
        <FiUploadCloud className="text-4xl text-green-500 mb-4 group-hover:animate-pulse" />
        <p className="text-sm text-green-100 font-bold font-mono uppercase tracking-wide">
          {files.length
            ? `[ ${files.length} FILE${files.length > 1 ? "S" : ""} SELECTED ]`
            : "[ DROP DOCS HERE OR CLICK ]"}
        </p>
        <p className="text-xs text-green-500 mt-2 font-mono">
          &gt; Include SLOs, alert runbooks, architecture overviews for richer
          answers
        </p>
      </label>

      <div className="flex justify-end gap-3 items-center">
        {loading && (
          <div className="flex items-center gap-2 text-green-500 font-mono text-sm">
            <div className="h-4 w-4 border-2 border-green-500 border-t-transparent loading-spinner" />
            <span className="animate-pulse">INDEXING...</span>
          </div>
        )}
        <button
          className="primary-button"
          type="button"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? "[PROCESSING]" : "[INDEX DOCUMENTATION]"}
        </button>
      </div>

      {error && (
        <div className="border-2 border-neon-pink bg-black px-4 py-3 text-xs text-neon-pink font-mono">
          <span className="font-bold">ERROR:</span> {error}
        </div>
      )}

      {preview && (
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-green-50 uppercase tracking-wider">
            [DOCUMENT PREVIEW]
          </h3>
          <pre className="text-xs text-green-300 bg-black border border-green-500/30 p-4 max-h-60 overflow-auto whitespace-pre-wrap font-mono">
            {preview}
          </pre>
        </div>
      )}
    </div>
  );
}
