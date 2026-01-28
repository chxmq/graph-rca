import { useAppStore } from "../store";
import { FiActivity, FiBookOpen, FiAlertTriangle } from "react-icons/fi";

export function StatusSidebar() {
  const { logStatus, docsStatus } = useAppStore();

  return (
    <aside className="hidden lg:flex w-80 border-r border-green-500/30 bg-black px-6 py-6 flex-col gap-6 scanlines">
      <div>
        <div className="pill mb-3">
          <FiActivity className="text-green-500 animate-pulse" />
          <span className="terminal-text">SYSTEM STATUS</span>
        </div>
        <h2 className="text-lg font-bold text-green-50 uppercase tracking-wider">Execution Queue</h2>
        <p className="text-xs text-green-400 mt-2 font-mono">
          &gt; Sequential analysis protocol
        </p>
      </div>

      <div className="space-y-3">
        <StepItem
          index={1}
          title="Analyse log file"
          description="Upload a .log or .txt file for causal analysis."
          done={logStatus.analysed}
        />
        <StepItem
          index={2}
          title="Ingest documentation"
          description="Add runbooks & guides to improve solution quality."
          done={docsStatus.uploaded}
        />
        <StepItem
          index={3}
          title="Generate resolution"
          description="Use the analysed context plus docs to resolve."
          done={false}
        />
      </div>

      <div className="mt-auto glass-panel p-4 space-y-2">
        <div className="flex items-center gap-2 text-sm font-bold text-neon-pink terminal-text uppercase">
          <FiAlertTriangle className="animate-pulse" />
          <span>Protocol Notice</span>
        </div>
        <p className="text-xs text-green-300 font-mono leading-relaxed">
          &gt; Optimal results: Single-incident logs + stack-specific documentation
        </p>
      </div>
    </aside>
  );
}

interface StepItemProps {
  index: number;
  title: string;
  description: string;
  done: boolean;
}

function StepItem({ index, title, description, done }: StepItemProps) {
  return (
    <div className="flex gap-3 items-start group">
      <div
        className={`flex h-8 w-8 items-center justify-center border text-xs font-bold transition-all ${
          done
            ? "border-green-500 bg-green-900/50 text-green-50 shadow-neon-sm"
            : "border-green-700 bg-black text-green-600"
        }`}
      >
        {done ? "✓" : index}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          {index === 1 && <FiActivity className={done ? "text-green-500" : "text-green-700"} />}
          {index === 2 && <FiBookOpen className={done ? "text-green-500" : "text-green-700"} />}
          {index === 3 && <FiAlertTriangle className={done ? "text-green-500" : "text-green-700"} />}
          <h3 className={`text-sm font-bold uppercase tracking-wide ${done ? "text-green-50" : "text-green-600"}`}>
            {title}
          </h3>
        </div>
        <p className="text-xs text-green-500 mt-1 font-mono">&gt; {description}</p>
      </div>
    </div>
  );
}

