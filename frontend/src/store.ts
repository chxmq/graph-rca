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

interface AppState {
  logStatus: LogStatus;
  docsStatus: DocsStatus;
  isProcessing: boolean;
  processingLogs: ProcessingLog[];
  processingProgress: number;
  setLogStatus: (status: Partial<LogStatus>) => void;
  setDocsStatus: (status: Partial<DocsStatus>) => void;
  setProcessing: (isProcessing: boolean) => void;
  addProcessingLog: (log: Omit<ProcessingLog, "timestamp">) => void;
  clearProcessingLogs: () => void;
  setProcessingProgress: (progress: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  logStatus: { analysed: false },
  docsStatus: { uploaded: false },
  isProcessing: false,
  processingLogs: [],
  processingProgress: 0,
  setLogStatus: (status) =>
    set((state) => ({ logStatus: { ...state.logStatus, ...status } })),
  setDocsStatus: (status) =>
    set((state) => ({ docsStatus: { ...state.docsStatus, ...status } })),
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

