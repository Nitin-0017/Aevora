import { useState } from "react";
import {
  MODEL_LABELS,
  type Engines,
  type History,
  type Prediction,
} from "./api";
import { useRemote } from "./useRemote";
import { LoadState } from "./components";
import { SensorChart } from "./charts";

export function BrandAsset() {
  const [missing, setMissing] = useState(false);
  return missing ? (
    <span className="brand-pending">Logo unavailable</span>
  ) : (
    <img
      className="brand-image"
      src="/assets/aevora-logo.jpg"
      alt="AEVORA silver A logo"
      onError={() => setMissing(true)}
    />
  );
}
export function TurbinePanel() {
  const [missing, setMissing] = useState(false);
  return (
    <section
      className="turbine-panel"
      aria-label="Industrial engine illustration"
    >
      {missing ? (
        <div className="image-pending">
          <p>Industrial turbofan visual</p>
          <span>Illustration unavailable.</span>
        </div>
      ) : (
        <img
          src="/assets/turbofan.png"
          alt="Illustrative industrial turbofan engine, not the selected demonstration engine"
          onError={() => setMissing(true)}
        />
      )}
      <div className="turbine-caption">
        <strong>AEVORA</strong>
        <p>Intelligence Before Failure.</p>
        <small>AI-generated illustration · not live equipment</small>
      </div>
    </section>
  );
}
export function OverviewDetails({
  engines,
  selected,
  cycle,
  prediction,
}: {
  engines: Engines;
  selected: number;
  cycle: number;
  prediction?: Prediction;
}) {
  const [sensor, setSensor] = useState(engines.sensors[0] ?? "sensor_2");
  const history = useRemote<History>(
    `/api/engines/${selected}/history?cycle=${cycle}&sensor=${encodeURIComponent(sensor)}`,
  );
  const valid =
    prediction?.valid &&
    prediction.raw_rul !== null &&
    Number.isFinite(prediction.raw_rul) &&
    prediction.raw_rul >= 0;
  return (
    <div className="overview-details">
      <section className="panel">
        <div className="section-top">
          <h2>Sensor trends</h2>
          <label className="sensor-control">
            Recorded sensor
            <select value={sensor} onChange={(e) => setSensor(e.target.value)}>
              {engines.sensors.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
        </div>
        <LoadState loading={history.loading} error={history.error} />
        {history.data && history.data.readings.length === 0 && (
          <LoadState empty />
        )}
        {history.data && history.data.readings.length > 0 && (
          <>
            <SensorChart data={history.data.readings} sensor={sensor} />
            <p className="small muted">
              Engine {selected} · {history.data.readings.length} recorded
              cycles. Dataset units; no inferred physical meaning.
            </p>
          </>
        )}
      </section>
      <section className="panel engine-information">
        <h2>Engine information</h2>
        {prediction ? (
          <dl>
            <dt>Engine ID</dt>
            <dd>{prediction.engine_id}</dd>
            <dt>Latest cycle</dt>
            <dd>{prediction.latest_cycle}</dd>
            <dt>Selected model</dt>
            <dd>{MODEL_LABELS[prediction.model]}</dd>
            <dt>Prediction time</dt>
            <dd>{new Date(prediction.generated_at).toLocaleString()}</dd>
            <dt>Status</dt>
            <dd className={valid ? "valid" : "amber"}>
              {valid ? "Valid prediction" : "Unable to assess"}
            </dd>
            <dt>Available history</dt>
            <dd>{prediction.available_cycles} cycles</dd>
          </dl>
        ) : (
          <LoadState empty />
        )}
        <p className="small muted">
          Timestamp records inference, not telemetry. Inputs must be finite and
          chronological with at least 20 cycles; frozen preprocessing is reused.
        </p>
      </section>
    </div>
  );
}
export function DataInformation() {
  return (
    <>
      <header className="page-header">
        <div className="eyebrow">DATA & LIMITATIONS</div>
        <h1>Data Information</h1>
        <p>Understand what the demonstration can—and cannot—tell you.</p>
      </header>
      <section className="panel data-information">
        <h2>NASA C-MAPSS FD001</h2>
        <p>
          Simulated turbofan engine run-to-failure trajectories provide
          operating cycles and recorded sensor fields. These are demonstration
          engines, not connected equipment.
        </p>
        <h2>What remaining useful life means</h2>
        <p>
          RUL is the number of operating cycles remaining until the dataset
          trajectory ends. Predictions estimate that quantity from historical
          sensor patterns; they are not guaranteed failure dates.
        </p>
        <h2>How the evaluation is separated</h2>
        <p>
          Training and validation use different engines. Preprocessing is fitted
          only on training engines. Sequences contain 20 chronological timesteps
          and cannot cross engine boundaries. Validation was used for model
          selection, so these metrics are not an unbiased final estimate of
          generalization.
        </p>
        <h2>Why this is not production-ready</h2>
        <p>
          Some estimates are negative or substantially overestimate remaining
          life. Status thresholds are illustrative; neither calibrated failure
          probabilities nor validated safety limits are available. Real
          deployment needs untouched evaluation, representative field data,
          operational validation and qualified human oversight.
        </p>
        <p>
          Official test RUL targets are not accessed by this dashboard. No
          maintenance events, causal explanations or confidence percentages are
          fabricated.
        </p>
      </section>
    </>
  );
}
