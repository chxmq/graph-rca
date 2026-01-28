import { ReactNode } from "react";
import { StatusSidebar } from "./StatusSidebar";

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen flex bg-black scanlines">
      <StatusSidebar />
      <main className="flex-1 px-6 py-6 md:px-10 md:py-8">
        <div className="max-w-7xl mx-auto space-y-6">
          <header className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between border-b border-green-500/30 pb-6">
            <div>
              <div className="pill mb-3">
                <span className="h-2 w-2 bg-green-500 animate-pulse shadow-neon-sm" />
                <span className="terminal-text">SYSTEM ONLINE</span>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold tracking-wider text-green-50 animate-glow uppercase">
                GRAPH-RCA
              </h1>
              <p className="text-green-300 mt-2 text-sm md:text-base font-mono">
                &gt; Root Cause Analysis Engine v1.0
              </p>
            </div>
            <div className="flex gap-2">
              <div className="px-3 py-1 border border-neon-pink/50 text-neon-pink text-xs font-mono">
                AI-POWERED
              </div>
              <div className="px-3 py-1 border border-neon-blue/50 text-neon-blue text-xs font-mono">
                REAL-TIME
              </div>
            </div>
          </header>

          {children}
        </div>
      </main>
    </div>
  );
}

