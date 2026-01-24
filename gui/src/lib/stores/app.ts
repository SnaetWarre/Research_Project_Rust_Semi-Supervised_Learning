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

// Interactive Demo state
export interface DemoState {
  status: 'idle' | 'initialized' | 'running' | 'completed' | 'error';
  currentDay: number;
  totalImagesAvailable: number;
  imagesProcessed: number;
  pseudoLabelsAccumulated: number;
  totalPseudoLabelsGenerated: number;
  retrainingCount: number;
  currentAccuracy: number;
  initialAccuracy: number;
  pseudoLabelPrecision: number;
  accuracyHistory: { day: number; accuracy: number }[];
  lastDayResult: DayResult | null;
  errorMessage?: string;
}

export interface DayResult {
  day: number;
  images_processed_today: number;
  pseudo_labels_accepted_today: number;
  pseudo_labels_accumulated: number;
  did_retrain: boolean;
  accuracy_before_retrain: number | null;
  accuracy_after_retrain: number | null;
  current_accuracy: number;
  pseudo_label_precision: number;
  sample_images: DayImage[];
  remaining_images: number;
}

export interface DayImage {
  path: string;
  predicted_label: number;
  confidence: number;
  accepted: boolean;
  ground_truth: number;
  is_correct: boolean;
  base64_thumbnail?: string; // Base64 encoded thumbnail for display
  is_farmer_image?: boolean; // Whether this is from farmer demo upload
}

export interface FarmerImportResult {
  images_processed: number;
  pseudo_labels_accepted: number;
  pseudo_labels_accumulated: number;
  sample_images: DayImage[];
  current_accuracy: number;
  pseudo_label_precision: number;
}

// Load persisted demoState from localStorage (for state persistence across navigation)
const loadPersistedDemoState = (): DemoState => {
  if (typeof window === 'undefined') {
    return {
      status: 'idle',
      currentDay: 0,
      totalImagesAvailable: 0,
      imagesProcessed: 0,
      pseudoLabelsAccumulated: 0,
      totalPseudoLabelsGenerated: 0,
      retrainingCount: 0,
      currentAccuracy: 0,
      initialAccuracy: 0,
      pseudoLabelPrecision: 0,
      accuracyHistory: [],
      lastDayResult: null,
    };
  }

  try {
    const saved = localStorage.getItem('demoState');
    if (saved) {
      return JSON.parse(saved);
    }
  } catch (e) {
    console.warn('Failed to load persisted demo state:', e);
  }

  return {
    status: 'idle',
    currentDay: 0,
    totalImagesAvailable: 0,
    imagesProcessed: 0,
    pseudoLabelsAccumulated: 0,
    totalPseudoLabelsGenerated: 0,
    retrainingCount: 0,
    currentAccuracy: 0,
    initialAccuracy: 0,
    pseudoLabelPrecision: 0,
    accuracyHistory: [],
    lastDayResult: null,
  };
};

export const demoState = writable<DemoState>(loadPersistedDemoState());

// Persist demoState to localStorage whenever it changes
if (typeof window !== 'undefined') {
  demoState.subscribe(state => {
    try {
      localStorage.setItem('demoState', JSON.stringify(state));
    } catch (e) {
      console.warn('Failed to persist demo state:', e);
    }
  });
}

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
