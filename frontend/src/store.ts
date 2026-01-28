import { create } from "zustand";

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
  severity: string;
  root_cause: string;
  summary: string[];
};

type AnalysisHistory = {
  id: string;
  timestamp: string;
  fileName: string;
  severity: string;
  root_cause: string;
  summary: string[];
};

interface AppState {
  logStatus: LogStatus;
  docsStatus: DocsStatus;
  analysisResult: AnalysisResult | null;
  analysisHistory: AnalysisHistory[];
  isProcessing: boolean;
  processingLogs: ProcessingLog[];
  processingProgress: number;
  setLogStatus: (status: Partial<LogStatus>) => void;
  setDocsStatus: (status: Partial<DocsStatus>) => void;
  setAnalysisResult: (result: AnalysisResult | null) => void;
  addToHistory: (fileName: string, result: AnalysisResult) => void;
  loadFromHistory: (id: string) => void;
  clearHistory: () => void;
  setProcessing: (isProcessing: boolean) => void;
  addProcessingLog: (log: Omit<ProcessingLog, "timestamp">) => void;
  clearProcessingLogs: () => void;
  setProcessingProgress: (progress: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  logStatus: { analysed: false },
  docsStatus: { uploaded: false },
  analysisResult: null,
  analysisHistory: [],
  isProcessing: false,
  processingLogs: [],
  processingProgress: 0,
  setLogStatus: (status) =>
    set((state) => ({ logStatus: { ...state.logStatus, ...status } })),
  setDocsStatus: (status) =>
    set((state) => ({ docsStatus: { ...state.docsStatus, ...status } })),
  setAnalysisResult: (result) => set({ analysisResult: result }),
  addToHistory: (fileName, result) =>
    set((state) => {
      const newEntry: AnalysisHistory = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        fileName,
        ...result,
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
            severity: entry.severity,
            root_cause: entry.root_cause,
            summary: entry.summary,
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
  setProcessingProgress: (progress) => set({ processingProgress: progress }),
}));

