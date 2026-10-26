import { invoke } from "@tauri-apps/api/core";

export interface ModelInfo {
  dir: string;
  classes: number;
  image_size: number;
}

export interface ClassScore {
  class: string;
  probability: number;
}

export interface Classification {
  scores: ClassScore[];
  millis: number;
}

export interface EpochLog {
  epoch: number;
  train_loss: number;
  val_accuracy: number;
  seconds: number;
}

export interface RoundLog {
  round: number;
  revealed: number;
  confident: number;
  accepted: number;
  precision: number;
  val_accuracy: number;
}

export interface TestReport {
  images: number;
  accuracy: number;
  macro_f1: number;
  classes: { class: string; precision: number; recall: number; f1: number; support: number }[];
}

export interface Run {
  name: string;
  train: { best_epoch: number; epochs: EpochLog[] } | null;
  ssl: { best_round: number; rounds: RoundLog[] } | null;
  test: TestReport | null;
}

export interface SslRequest {
  split: string;
  init: string;
  out: string;
  rounds: number;
  threshold: number;
  epochs_per_round: number;
  fresh_student: boolean;
}

export type SslEvent =
  | { kind: "pass"; label: string; images: number }
  | { kind: "progress"; done: number }
  | ({ kind: "epoch" } & EpochLog)
  | ({ kind: "round" } & RoundLog)
  | { kind: "finished"; best_round: number; out: string }
  | { kind: "stopped" }
  | { kind: "failed"; message: string };

export const startSsl = (request: SslRequest) => invoke<void>("start_ssl", { request });

export const stopSsl = () => invoke<void>("stop_ssl");

export const loadModel = (dir: string) => invoke<ModelInfo>("load_model", { dir });

export const classify = (bytes: Uint8Array) => invoke<Classification>("classify", bytes);

export const listRuns = (dir: string) => invoke<Run[]>("list_runs", { dir });

/** `Tomato___Late_blight` becomes `Tomato: Late blight`. */
export function className(raw: string): string {
  const [plant, condition = ""] = raw.split("___");
  const clean = (s: string) => s.replaceAll("_", " ").replace(/\s+/g, " ").trim();
  return condition ? `${clean(plant)}: ${clean(condition)}` : clean(plant);
}

export const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
