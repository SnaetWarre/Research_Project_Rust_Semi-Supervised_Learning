import { writable } from 'svelte/store';

// Current page/route
export const currentPage = writable<string>('dashboard');

// Model state
export interface ModelInfo {
  loaded: boolean;
  path: string | null;
  numClasses: number;
  inputSize: number;
}

export const modelInfo = writable<ModelInfo>({
  loaded: false,
  path: null,
  numClasses: 0,
  inputSize: 0,
});

// Dataset info
export interface DatasetInfo {
  path: string;
  totalSamples: number;
  numClasses: number;
  classNames: string[];
  classCounts: number[];
}

export const datasetInfo = writable<DatasetInfo | null>(null);

// Training state
export interface TrainingState {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  epoch: number;
  totalEpochs: number;
  batch: number;
  totalBatches: number;
  currentLoss: number;
  currentAccuracy: number;
  lossHistory: number[];
  accuracyHistory: number[];
  learningRateHistory: number[];
  errorMessage?: string;
}

export const trainingState = writable<TrainingState>({
  status: 'idle',
  epoch: 0,
  totalEpochs: 0,
  batch: 0,
  totalBatches: 0,
  currentLoss: 0,
  currentAccuracy: 0,
  lossHistory: [],
  accuracyHistory: [],
  learningRateHistory: [],
});

// Simulation state
export interface SimulationState {
  status: 'idle' | 'running' | 'completed' | 'error';
  day: number;
  totalDays: number;
  pseudoLabels: number;
  currentAccuracy: number;
  accuracyHistory: { day: number; accuracy: number }[];
  errorMessage?: string;
}

export const simulationState = writable<SimulationState>({
  status: 'idle',
  day: 0,
  totalDays: 0,
  pseudoLabels: 0,
  currentAccuracy: 0,
  accuracyHistory: [],
});

// Diagnostics state
export interface DiagnosticResult {
  class_predictions: { [key: number]: number };
  class_confidences: { [key: number]: number[] };
  total_predictions: number;
  most_predicted_class: number;
  most_predicted_class_name: string;
  prediction_bias_score: number;
  low_confidence_count: number;
  class_distribution: { [key: number]: number };
  input_distribution: { [key: string]: number };
}

export interface DiagnosticsState {
  result: DiagnosticResult | null;
  lastRunAt: Date | null;
  config: {
    numSamples: number;
    confidenceThreshold: number;
  };
}

export const diagnosticsState = writable<DiagnosticsState>({
  result: null,
  lastRunAt: null,
  config: {
    numSamples: 100,
    confidenceThreshold: 0.7,
  },
});

// Activity log
export interface ActivityItem {
  id: number;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  timestamp: Date;
}

export const activityLog = writable<ActivityItem[]>([]);

let activityId = 0;
export function addActivity(type: ActivityItem['type'], message: string) {
  activityLog.update(log => {
    const newLog = [
      { id: ++activityId, type, message, timestamp: new Date() },
      ...log,
    ].slice(0, 50); // Keep last 50 items
    return newLog;
  });
}
