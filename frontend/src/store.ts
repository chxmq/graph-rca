import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

type LogStatus = {
  analysed: boolean;
  severity?: string;
  rootCause?: string;
};

type DocsStatus = {
  uploaded: boolean;
  count?: number;
};

type ProcessingLog = {
  timestamp: string;
  message: string;
  type: "info" | "success" | "error" | "progress";
};

type AnalysisResult = {
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
  loadedFromHistory?: boolean;
};

type AnalysisHistory = {
  id: string;
  timestamp: string;
  fileName: string;
  analysis_id?: string;
  severity: string;
  root_cause: string;
  summary: string[];
  root_cause_expln: string;
};

type ResolutionResult = {
  root_cause: string;
  solution: string;
  sources: string[];
};

interface AppState {
  logStatus: LogStatus;
  docsStatus: DocsStatus;
  analysisResult: AnalysisResult | null;
  analysisHistory: AnalysisHistory[];
  resolutionResult: ResolutionResult | null;
  isProcessing: boolean;
  processingLogs: ProcessingLog[];
  processingProgress: number;
  setLogStatus: (status: Partial<LogStatus>) => void;
  setDocsStatus: (status: Partial<DocsStatus>) => void;
  setAnalysisResult: (result: AnalysisResult | null) => void;
  mergeAnalysisResult: (result: Partial<AnalysisResult>) => void;
  setResolutionResult: (result: ResolutionResult | null) => void;
  addToHistory: (fileName: string, result: AnalysisResult) => void;
  loadFromHistory: (id: string) => void;
  clearHistory: () => void;
  setProcessing: (isProcessing: boolean) => void;
  addProcessingLog: (log: Omit<ProcessingLog, "timestamp">) => void;
  clearProcessingLogs: () => void;
  setProcessingProgress: (progress: number | ((prev: number) => number)) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      logStatus: { analysed: false },
      docsStatus: { uploaded: false },
      analysisResult: null,
      analysisHistory: [],
      resolutionResult: null,
      isProcessing: false,
      processingLogs: [],
      processingProgress: 0,
      setLogStatus: (status) =>
        set((state) => ({ logStatus: { ...state.logStatus, ...status } })),
      setDocsStatus: (status) =>
        set((state) => ({ docsStatus: { ...state.docsStatus, ...status } })),
      setAnalysisResult: (result) => set({ analysisResult: result }),
      mergeAnalysisResult: (result) =>
        set((state) => ({
          analysisResult: state.analysisResult
            ? { ...state.analysisResult, ...result }
            : null,
        })),
      setResolutionResult: (result) => set({ resolutionResult: result }),
      addToHistory: (fileName, result) =>
        set((state) => {
          const newEntry: AnalysisHistory = {
            id:
              typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
                ? crypto.randomUUID()
                : `${Date.now()}-${Math.random().toString(36).slice(2)}`,
            timestamp: new Date().toISOString(),
            fileName,
            analysis_id: result.analysis_id,
            severity: result.severity,
            root_cause: result.root_cause,
            summary: result.summary,
            root_cause_expln: result.root_cause_expln,
          };
          // Keep only last 10 analyses
          const updatedHistory = [newEntry, ...state.analysisHistory].slice(0, 10);
          return { analysisHistory: updatedHistory };
        }),
      loadFromHistory: (id) =>
        set((state) => {
          const entry = state.analysisHistory.find((h) => h.id === id);
          if (entry) {
            return {
              analysisResult: {
                analysis_id: entry.analysis_id,
                severity: entry.severity,
                root_cause: entry.root_cause,
                summary: entry.summary,
                context: {},
                root_cause_expln: entry.root_cause_expln ?? "",
                loadedFromHistory: true,
              },
              logStatus: {
                analysed: true,
                severity: entry.severity,
                rootCause: entry.root_cause,
              },
            };
          }
          return state;
        }),
      clearHistory: () => set({ analysisHistory: [] }),
      setProcessing: (isProcessing) => set({ isProcessing }),
      addProcessingLog: (log) =>
        set((state) => ({
          processingLogs: [
            ...state.processingLogs,
            { ...log, timestamp: new Date().toISOString() },
          ],
        })),
      clearProcessingLogs: () => set({ processingLogs: [], processingProgress: 0 }),
      setProcessingProgress: (next) =>
        set((state) => ({
          processingProgress:
            typeof next === "function" ? next(state.processingProgress) : next,
        })),
    }),
    {
      name: "graph-rca-store",
      // Bump version whenever a breaking schema change is made to persisted state.
      // v1 → v2: analysis_id was added to AnalysisHistory entries.  Entries
      //           without it can never open a resolve panel (canFetchHistoryContext
      //           returns false), so migrate drops them rather than leaving
      //           permanently broken history entries.
      version: 2,
      storage: createJSONStorage(() => localStorage),
      // Only persist history — transient UI state is always reset on load
      partialize: (state) => ({
        analysisHistory: state.analysisHistory,
      }),
      migrate: (persisted: unknown, version: number) => {
        const state = persisted as Partial<AppState>;
        if (version < 2) {
          state.analysisHistory = (state.analysisHistory ?? []).filter(
            (entry) => Boolean(entry.analysis_id)
          );
        }
        return state;
      },
    }
  )
);

