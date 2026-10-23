import { useState } from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  Clock3,
  Info,
  LoaderCircle,
  ShieldAlert,
} from "lucide-react";
import { MODEL_LABELS, hoursFromCycles, type Prediction } from "./api";
export function LoadState({
  error,
  loading,
  empty = false,
}: {
  error?: string;
  loading?: boolean;
  empty?: boolean;
}) {
  if (error)
    return (
      <div className="notice danger" role="alert">
        <AlertTriangle size={20} />
        <div>
          <strong>Unable to load this view</strong>
          <p>{error}</p>
        </div>
      </div>
    );
  if (loading)
    return (
      <div className="loading" role="status">
        <LoaderCircle className="spin" size={22} /> Reading verified local data…
      </div>
    );
  if (empty)
    return (
      <div className="notice">
        No eligible observations are available for this selection.
      </div>
    );
  return null;
}
export function Status({
  value,
  valid,
  level,
}: {
  value: string;
  valid: boolean;
  level: string;
}) {
  return (
    <span className={`badge ${valid ? level : "invalid"}`}>
      {valid ? <Info size={13} /> : <ShieldAlert size={13} />}{" "}
      {valid ? value : "Unable to assess"}
    </span>
  );
}
export function HealthCard({ prediction }: { prediction: Prediction }) {
  const [rate, setRate] = useState("");
  const [convert, setConvert] = useState(false);
  const finite =
    prediction.raw_rul !== null && Number.isFinite(prediction.raw_rul);
  const valid = finite && prediction.raw_rul! >= 0 && prediction.valid;
  const hours =
    valid && convert ? hoursFromCycles(prediction.raw_rul, rate) : null;
  return (
    <section
      className="panel health-card"
      aria-label={`Engine ${prediction.engine_id} condition`}
    >
      <div className="section-top">
        <span className="eyebrow">
          Engine {prediction.engine_id} · Latest cycle:{" "}
          {prediction.latest_cycle}
        </span>
        <span className={`validity ${valid ? "valid" : "invalid"}`}>
          {valid ? "● Valid prediction" : "● Unable to assess"}
        </span>
      </div>
      <div className="health-grid">
        <div>
          <div className="machine-status">
            <span
              className={`status-emblem ${valid ? prediction.level : "invalid"}`}
            >
              <Info size={38} />
            </span>
            <div>
              <span className="eyebrow">MACHINE STATUS</span>
              <h2>{valid ? prediction.status : "Unable to assess"}</h2>
              <p className="muted small">
                Illustrative demonstration band · not a safety assessment.
              </p>
            </div>
          </div>
          <p className="muted">Estimated remaining useful life</p>
          <div className="rul">
            {finite ? Math.round(prediction.raw_rul!) : "—"}
            <span>{finite ? "operating cycles" : "Unavailable"}</span>
            <div className="raw-output">
              Model output (raw)
              <strong>
                {finite ? prediction.raw_rul!.toFixed(2) : "Unavailable"}
              </strong>
              <small>cycles · shown to 2 decimals</small>
            </div>
          </div>
          <p className="muted small">
            Raw model output · rounded to 2 decimals
            <br />
            Operating cycles, not a guaranteed failure date.
          </p>
        </div>
        <div className="action-box">
          <span className="eyebrow">
            <ArrowUpRight size={16} /> Recommended next step
          </span>
          <h3>
            {valid
              ? prediction.level === "high"
                ? "Arrange a qualified inspection"
                : prediction.level === "elevated"
                  ? "Review and keep monitoring"
                  : "Continue scheduled inspections"
              : "Do not rely on this estimate"}
          </h3>
          <p>
            {valid
              ? prediction.recommendation
              : "Ask a qualified technician to review the inputs and model output. Negative or non-finite estimates cannot describe physical remaining life."}
          </p>
          <span className="small muted">
            Decision support only · not a validated operational procedure.
          </span>
        </div>
      </div>
      <div className="card-meta">
        <span>
          <Clock3 size={15} /> Latest cycle <b>{prediction.latest_cycle}</b>
        </span>
        <span>
          Model <b>{MODEL_LABELS[prediction.model]}</b>
        </span>
        <span>
          History <b>{prediction.available_cycles} cycles</b>
        </span>
      </div>
      <details>
        <summary>Why this estimate?</summary>
        <p>
          {valid
            ? prediction.reason
            : "The model output is invalid and needs manual review."}
        </p>
        <p>
          AEVORA learns patterns in historical sensor readings. It can be wrong.
          These status bands are illustrative, not safety limits or failure
          probabilities. A concerning result warrants qualified inspection, not
          automatic shutdown. No causal explanation or confidence percentage is
          claimed.
        </p>
      </details>
      <details>
        <summary>Optional operating-hours estimate</summary>
        <label className="check">
          <input
            type="checkbox"
            checked={convert}
            onChange={(e) => setConvert(e.target.checked)}
          />{" "}
          I have a verified average operating rate
        </label>
        {convert && (
          <label>
            Cycles per hour
            <input
              type="number"
              min="0"
              step="any"
              placeholder="Enter measured rate"
              value={rate}
              onChange={(e) => setRate(e.target.value)}
            />
          </label>
        )}
        {hours !== null ? (
          <p>
            Approximately <b>{hours.toFixed(2)} operating hours</b>. Assumes a
            constant user-supplied rate; not a guaranteed failure time.
          </p>
        ) : (
          <p className="small muted">
            No time estimate without a positive verified rate and a valid
            prediction.
          </p>
        )}
      </details>
      <p className="timestamp">
        Generated {new Date(prediction.generated_at).toLocaleString()} ·
        application time, not measurement time · schema v
        {prediction.model_version} / preprocessing v
        {prediction.preprocessing_version}
      </p>
    </section>
  );
}
