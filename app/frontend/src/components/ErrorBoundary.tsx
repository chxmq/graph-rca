import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Top-level error boundary. Without this, a render error anywhere in the
 * tree blanks the page with no fallback. We catch render-time exceptions,
 * log them to the console (so they're still surfaced for devs), and show
 * a minimal terminal-styled error panel.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[ErrorBoundary] uncaught:", error, info);
  }

  reset = () => this.setState({ hasError: false, error: null });

  render() {
    if (!this.state.hasError) return this.props.children;
    if (this.props.fallback) return this.props.fallback;

    const message = this.state.error?.message ?? "Unknown rendering error";
    return (
      <div className="min-h-screen flex items-center justify-center bg-black p-6 scanlines">
        <div className="glass-panel max-w-2xl w-full p-6 space-y-4">
          <h2 className="text-lg font-bold text-neon-pink uppercase tracking-wider terminal-text">
            [APP CRASH]
          </h2>
          <p className="text-xs text-green-300 font-mono">
            &gt; The UI hit an unrecoverable error and was halted.
          </p>
          <pre className="text-xs text-red-300 bg-black border border-red-500/30 p-4 max-h-48 overflow-auto whitespace-pre-wrap font-mono">
            {message}
          </pre>
          <p className="text-xs text-green-500 font-mono">
            &gt; Reloading the page is usually safe; persistent state lives in localStorage.
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              className="primary-button"
              onClick={() => window.location.reload()}
            >
              [RELOAD]
            </button>
            <button type="button" className="secondary-button" onClick={this.reset}>
              [DISMISS]
            </button>
          </div>
        </div>
      </div>
    );
  }
}
