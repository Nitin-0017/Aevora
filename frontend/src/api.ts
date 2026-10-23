export type ModelID = "baseline" | "ridge" | "gradient_boosting" | "lstm";
export interface Model {
  id: ModelID;
  name: string;
  schema_version: number;
  preprocessing_version: number;
  sequence_length: number;
  feature_count: number;
}
export interface Engine {
  engine_id: number;
  available_cycles: number;
}
export interface Engines {
  engines: Engine[];
  sensors: string[];
  min_cycle: number;
  max_cycle: number;
  disclosure: string;
}
export interface Prediction {
  engine_id: number;
  model: ModelID;
  raw_rul: number | null;
  valid: boolean;
  status: string;
  level: string;
  recommendation: string;
  reason: string;
  latest_cycle: number;
  available_cycles: number;
  model_version: number;
  preprocessing_version: number;
  generated_at: string;
  warning: string;
  guidance: string;
}
export interface Fleet {
  predictions: Prediction[];
  generated_at: string;
  disclosure: string;
}
export interface Metrics {
  model: ModelID;
  mae: number;
  rmse: number;
  r2: number;
  within_10_pct: number;
  within_20_pct: number;
  mean_signed_error: number;
  p95_absolute_error: number;
  observations: number;
  engines: number;
}
export interface Diagnostics {
  observations: number;
  points: {
    unit_id: number;
    cycle: number;
    rul: number;
    predicted_rul: number;
  }[];
  residual_bins: { cycles: number; count: number }[];
  per_engine: { engine_id: number; mae: number; observations: number }[];
}
export interface History {
  engine_id: number;
  sensor: string;
  units: string;
  readings: { cycle: number; value: number }[];
}
export interface Reliability {
  mean_bias: number;
  middle_bias: number;
  largest_overprediction: number;
  p95_absolute_error: number;
  negative_predictions: { model: ModelID; count: number }[];
  noise_probe: {
    model: ModelID;
    mean_absolute_prediction_change: number;
    max_absolute_prediction_change: number;
  }[];
  scope: string;
  perturbation: { seed: number; fraction: number };
}
export const MODEL_LABELS: Record<ModelID, string> = {
  baseline: "Median baseline",
  ridge: "Ridge",
  gradient_boosting: "Boosted trees",
  lstm: "LSTM",
};
export const DISCLOSURE =
  "NASA C-MAPSS FD001 · Demonstration data · Not live equipment";
export async function request<T>(
  url: string,
  signal?: AbortSignal,
  body?: unknown,
): Promise<T> {
  const response = await fetch(url, {
    signal,
    method: body === undefined ? "GET" : "POST",
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      data?.error?.message ??
        "The local API is unavailable. Check that the backend is running and the audited artifacts are installed.",
    );
  if (data === null)
    throw new Error("The API returned an unreadable response. Please retry.");
  if (
    url.includes("/history?") &&
    (!Array.isArray(data.readings) ||
      !data.readings.every(
        (row: { cycle: unknown; value: unknown }) =>
          row &&
          typeof row.cycle === "number" &&
          Number.isFinite(row.cycle) &&
          typeof row.value === "number" &&
          Number.isFinite(row.value),
      ))
  ) {
    throw new Error(
      "The API returned invalid sensor history. Check the backend data.",
    );
  }
  if (
    (url === "/api/fleet-predictions" || url === "/api/predict") &&
    (!Array.isArray(data.predictions) ||
      !data.predictions.every(
        (row: Prediction) =>
          row &&
          Number.isInteger(row.engine_id) &&
          typeof row.raw_rul === "number" &&
          Number.isFinite(row.raw_rul) &&
          typeof row.valid === "boolean" &&
          row.model in MODEL_LABELS &&
          Number.isInteger(row.latest_cycle) &&
          Number.isFinite(Date.parse(row.generated_at)),
      ))
  ) {
    throw new Error(
      "The API returned an invalid prediction response. Unable to assess.",
    );
  }
  return data as T;
}
export function hoursFromCycles(
  rul: number | null,
  rate: string,
): number | null {
  const n = Number(rate);
  if (
    rul === null ||
    !Number.isFinite(rul) ||
    rul < 0 ||
    !rate.trim() ||
    !Number.isFinite(n) ||
    n <= 0
  )
    return null;
  const result = rul / n;
  return Number.isFinite(result) ? result : null;
}
