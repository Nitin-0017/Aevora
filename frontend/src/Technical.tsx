import { useState } from "react";
import { useRemote } from "./useRemote";
import { LoadState } from "./components";
import { Bars, ActualChart } from "./charts";
import {
  MODEL_LABELS,
  type ModelID,
  type Metrics,
  type Diagnostics,
  type Reliability,
} from "./api";
export function Comparison() {
  const [model, setModel] = useState<ModelID>("gradient_boosting");
  const scores = useRemote<{ scope: string; metrics: Metrics[] }>(
    "/api/model-metrics",
  );
  const diagnostic = useRemote<Diagnostics>(`/api/diagnostics/${model}`);
  return (
    <>
      <header className="page-header">
        <div className="eyebrow">THE EVIDENCE</div>
        <h1>Model performance</h1>
        <p>Compare what the models got right — and where they missed.</p>
      </header>
      <div className="notice">
        <strong>Development validation, not final test performance.</strong>{" "}
        These 3,472 observations from 20 engines helped select the models.
        Overlapping windows are correlated; these are not unbiased
        generalization estimates.
      </div>
      <LoadState {...scores} />
      {scores.data && (
        <>
          <div className="panel table-wrap">
            <table>
              <caption>
                Same validation observations for every model. Errors in
                operating cycles.
              </caption>
              <thead>
                <tr>
                  <th>Model</th>
                  <th title="Mean absolute error in cycles">MAE</th>
                  <th title="Root mean squared error: emphasizes large errors">
                    RMSE
                  </th>
                  <th title="Variance-relative fit, not prediction accuracy">
                    R²
                  </th>
                  <th>Within ±10</th>
                  <th>Within ±20</th>
                  <th title="Prediction minus actual RUL">Bias</th>
                  <th>P95 error</th>
                </tr>
              </thead>
              <tbody>
                {scores.data.metrics.map((m) => (
                  <tr key={m.model}>
                    <th>{MODEL_LABELS[m.model]}</th>
                    <td>{m.mae.toFixed(2)}</td>
                    <td>{m.rmse.toFixed(2)}</td>
                    <td>{m.r2.toFixed(3)}</td>
                    <td>{m.within_10_pct.toFixed(1)}%</td>
                    <td>{m.within_20_pct.toFixed(1)}%</td>
                    <td>{m.mean_signed_error.toFixed(2)}</td>
                    <td>{m.p95_absolute_error.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="two-col">
            <section className="panel">
              <h3>Mean absolute error · cycles</h3>
              <Bars
                data={scores.data.metrics.map((m) => ({
                  model: MODEL_LABELS[m.model],
                  mae: m.mae,
                }))}
                x="model"
                y="mae"
                label="MAE (cycles)"
              />
            </section>
            <section className="panel">
              <h3>Root mean squared error · cycles</h3>
              <Bars
                data={scores.data.metrics.map((m) => ({
                  model: MODEL_LABELS[m.model],
                  rmse: m.rmse,
                }))}
                x="model"
                y="rmse"
                label="RMSE (cycles)"
              />
            </section>
          </div>
        </>
      )}
      <div className="section-top">
        <h2>Look beyond the average</h2>
        <label>
          Diagnostic model
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as ModelID)}
          >
            {Object.entries(MODEL_LABELS).map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>
      <LoadState {...diagnostic} />
      {diagnostic.data && (
        <>
          <div className="two-col">
            <section className="panel">
              <h3>
                Actual vs predicted · n=
                {diagnostic.data.observations.toLocaleString()}
              </h3>
              <p className="small muted">
                Vertical: predicted cycles. Dashed line: ideal prediction.
              </p>
              <ActualChart data={diagnostic.data.points} />
            </section>
            <section className="panel">
              <h3>Error distribution · same observations</h3>
              <p className="small muted">
                Positive error overstates remaining life. Vertical:
                observations.
              </p>
              <Bars
                data={diagnostic.data.residual_bins}
                x="cycles"
                y="count"
                label="Observations"
              />
            </section>
          </div>
          <section className="panel">
            <h3>Mean absolute error by engine · cycles</h3>
            <Bars
              data={diagnostic.data.per_engine}
              x="engine_id"
              y="mae"
              label="MAE (cycles)"
            />
          </section>
        </>
      )}
      <p className="small muted">
        MAE describes average error magnitude; RMSE emphasizes extremes. R² is
        not prediction accuracy. Tolerance rates count predictions within the
        stated error, and P95 is an observed error quantile — not a confidence
        interval.
      </p>
    </>
  );
}
export function ReliabilityPage() {
  const r = useRemote<Reliability>("/api/reliability");
  return (
    <>
      <header className="page-header">
        <div className="eyebrow">KNOW THE LIMITS</div>
        <h1>Reliability insights</h1>
        <p>Useful estimates need honest boundaries.</p>
      </header>
      <div className="notice">
        Verified development-validation findings. These models are research
        prototypes, not validated maintenance decision systems.
      </div>
      <LoadState {...r} />
      {r.data && (
        <>
          <div className="stats reliability-stats">
            {[
              ["Overall overestimation", r.data.mean_bias],
              ["Middle-life overestimation", r.data.middle_bias],
              ["Largest overprediction", r.data.largest_overprediction],
              ["P95 absolute error", r.data.p95_absolute_error],
            ].map(([label, value]) => (
              <div className="stat" key={String(label)}>
                <span>{label}</span>
                <strong>
                  {Number(value).toFixed(2)} <small>cycles</small>
                </strong>
                <p>Boosted trees · validation</p>
              </div>
            ))}
          </div>
          <div className="two-col">
            <section className="panel">
              <h2>Why overprediction matters</h2>
              <p>
                A model can suggest more life remains than actually does. That
                could postpone inspection. Middle life here means actual RUL
                above 30 through 100 cycles, known only retrospectively.
              </p>
              <p>
                Do not translate these averages into a safe operating allowance
                or a guaranteed error bound for an individual engine.
              </p>
            </section>
            <section className="panel">
              <h2>Invalid negative estimates</h2>
              {r.data.negative_predictions.map((row) => (
                <div className="list-row" key={row.model}>
                  <span>{MODEL_LABELS[row.model]}</span>
                  <b>{row.count}</b>
                </div>
              ))}
              <p className="small muted">
                Counts across 3,472 observations per model. Negative remaining
                life is physically invalid, even when the software returns a
                finite number.
              </p>
            </section>
          </div>
          <section className="panel">
            <h2>Small sensor changes can move the estimate</h2>
            <p>
              Saved synthetic probe: {r.data.perturbation.fraction * 100}% of
              training sensor standard deviation, independent Gaussian noise,
              seed {r.data.perturbation.seed}. Not rerun by this application.
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Mean absolute shift (cycles)</th>
                    <th>Maximum shift (cycles)</th>
                  </tr>
                </thead>
                <tbody>
                  {r.data.noise_probe.map((row) => (
                    <tr key={row.model}>
                      <th>{MODEL_LABELS[row.model]}</th>
                      <td>{row.mean_absolute_prediction_change.toFixed(2)}</td>
                      <td>{row.max_absolute_prediction_change.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="notice">
              Sensitivity is not calibrated uncertainty, failure probability, or
              proof of field robustness.
            </div>
          </section>
          <section className="panel">
            <h2>Before operational use</h2>
            <p>
              Freeze evaluation choices before an approved official test
              evaluation; validate on independent engines and field conditions;
              define qualified inspection and monitoring procedures. Official
              test targets remain untouched. Passing software tests alone does
              not establish model reliability.
            </p>
          </section>
        </>
      )}
    </>
  );
}
