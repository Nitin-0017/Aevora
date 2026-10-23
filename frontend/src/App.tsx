import { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  BookOpen,
  ChevronRight,
  Layers3,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";
import {
  DISCLOSURE,
  type Engines,
  type Fleet,
  type History,
  type Model,
  type ModelID,
  type Prediction,
} from "./api";
import { useRemote } from "./useRemote";
import { HealthCard, LoadState } from "./components";
import { SensorChart } from "./charts";
import { Comparison, ReliabilityPage } from "./Technical";
const pages = [
  { id: "fleet", name: "Fleet overview", icon: Layers3 },
  { id: "engines", name: "Engine explorer", icon: Activity },
  { id: "models", name: "Model performance", icon: BarChart3 },
  { id: "reliability", name: "Reliability insights", icon: ShieldCheck },
] as const;
type Page = (typeof pages)[number]["id"];
function currentPage(): Page {
  const key = window.location.hash.replace("#/", "");
  return pages.some((p) => p.id === key) ? (key as Page) : "fleet";
}
export function EngineExplorer({
  engines,
  selected,
  onSelect,
  cycle,
  prediction,
}: {
  engines: Engines;
  selected: number;
  onSelect: (id: number) => void;
  cycle: number;
  prediction?: Prediction;
}) {
  const [sensor, setSensor] = useState(engines.sensors[0] ?? "sensor_2");
  const history = useRemote<History>(
    `/api/engines/${selected}/history?cycle=${cycle}&sensor=${encodeURIComponent(sensor)}`,
  );
  return (
    <>
      <header className="page-header">
        <div className="eyebrow">RECORDED ENGINE HISTORY</div>
        <h1>Engine explorer</h1>
        <p>Follow the readings behind the estimate.</p>
      </header>
      <div className="controls panel">
        <label>
          Demonstration engine
          <select
            value={selected}
            onChange={(e) => onSelect(Number(e.target.value))}
          >
            {engines.engines.map((e) => (
              <option key={e.engine_id} value={e.engine_id}>
                Engine {e.engine_id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Recorded sensor
          <select value={sensor} onChange={(e) => setSensor(e.target.value)}>
            {engines.sensors.map((s) => (
              <option value={s} key={s}>
                {s.replace("sensor_", "Sensor ")}
              </option>
            ))}
          </select>
        </label>
        <span className="small muted">
          History through cycle {cycle}. Earlier-ended engines use their last
          available record.
        </span>
      </div>
      <LoadState {...history} />
      {history.data && (
        <section className="panel">
          <div className="section-top">
            <h2>
              {sensor.replace("sensor_", "Sensor ")} · engine {selected}
            </h2>
            <span className="badge neutral">
              {history.data.readings.length} recorded cycles
            </span>
          </div>
          <SensorChart data={history.data.readings} />
          <p className="small muted">
            Recorded dataset units. Sensor IDs are shown without invented
            engineering meanings. No causal fault explanation is inferred from
            this trace.
          </p>
        </section>
      )}
      {prediction && <HealthCard key={selected} prediction={prediction} />}
      <div className="notice">
        Data checks require complete chronological, finite input history and at
        least 20 cycles. The frozen preprocessing uses 20 × 75 ordered sensor
        features. No new scaler is fitted. Only the current estimate is shown
        here; no future readings or maintenance events are fabricated.
      </div>
    </>
  );
}
export default function App() {
  const [page, setPage] = useState<Page>(currentPage);
  const [model, setModel] = useState<ModelID>("gradient_boosting");
  const [selected, setSelected] = useState(3);
  const [revision, setRevision] = useState(0);
  const [settings, setSettings] = useState({
    cycle: 100,
    review_threshold: 30,
    monitor_threshold: 100,
  });
  const [draft, setDraft] = useState(settings);
  const [formError, setFormError] = useState("");
  const engines = useRemote<Engines>("/api/engines", undefined, revision);
  const models = useRemote<{ models: Model[] }>(
    "/api/models",
    undefined,
    revision,
  );
  const fleet = useRemote<Fleet>(
    "/api/fleet-predictions",
    { model, ...settings },
    revision,
  );
  useEffect(() => {
    const listener = () => setPage(currentPage());
    window.addEventListener("hashchange", listener);
    return () => window.removeEventListener("hashchange", listener);
  }, []);
  const rows = fleet.data?.predictions ?? [];
  const prediction = rows.find((p) => p.engine_id === selected);
  const valid = rows.filter(
    (p) =>
      p.valid &&
      p.raw_rul !== null &&
      Number.isFinite(p.raw_rul) &&
      p.raw_rul >= 0,
  ).length;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#/fleet" aria-label="AEVORA home">
          <div className="brand-symbol">
            <Activity size={25} />
          </div>
          <div>
            AEVORA<span>Intelligence Before Failure.</span>
          </div>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {pages.map((p) => (
            <a
              key={p.id}
              href={`#/${p.id}`}
              aria-current={page === p.id ? "page" : undefined}
              className={page === p.id ? "active" : ""}
            >
              <p.icon size={18} />
              {p.name}
              {page === p.id && <ChevronRight size={14} />}
            </a>
          ))}
        </nav>
        <div className="sidebar-note">
          <BookOpen size={20} />
          <b>Built for informed decisions.</b>
          <p>
            Estimates support inspection. They do not replace a qualified
            technician.
          </p>
          <span>FD001 / LOCAL DEMONSTRATION</span>
        </div>
      </aside>
      <main>
        <div className="topbar">
          <span>Predictive Maintenance Intelligence</span>
          <span className="environment">
            <span className="dot" /> Offline-capable demo
          </span>
        </div>
        <div className="content">
          <div className="disclosure">
            <InfoMark />
            {DISCLOSURE}
          </div>
          {(page === "fleet" || page === "engines") && (
            <>
              <header className="page-header">
                {page === "fleet" && (
                  <>
                    <div className="eyebrow">
                      OPERATING INSIGHT, BEFORE FAILURE
                    </div>
                    <h1>A clearer view of remaining life.</h1>
                    <p>
                      Choose an engine. Review its estimate. Know the next step.
                    </p>
                  </>
                )}
              </header>
              <form
                className="controls panel"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (
                    !Number.isFinite(draft.cycle) ||
                    !Number.isInteger(draft.cycle) ||
                    draft.cycle < 20 ||
                    !Number.isFinite(draft.review_threshold) ||
                    !Number.isFinite(draft.monitor_threshold) ||
                    draft.review_threshold < 0 ||
                    draft.review_threshold >= draft.monitor_threshold
                  ) {
                    setFormError(
                      "Use a cycle of at least 20 and thresholds with 0 ≤ review < monitor.",
                    );
                    return;
                  }
                  setFormError("");
                  setSettings(draft);
                  setRevision((n) => n + 1);
                }}
              >
                <label>
                  Saved prediction model
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value as ModelID)}
                  >
                    {(models.data?.models ?? []).map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Replay through cycle
                  <input
                    aria-label="Replay through cycle"
                    type="number"
                    min={20}
                    max={engines.data?.max_cycle ?? 10000}
                    value={draft.cycle}
                    onChange={(e) =>
                      setDraft({ ...draft, cycle: Number(e.target.value) })
                    }
                  />
                </label>
                <button
                  className="primary"
                  type="submit"
                  disabled={fleet.loading}
                >
                  <RefreshCw
                    size={16}
                    className={fleet.loading ? "spin" : ""}
                  />
                  {fleet.loading ? "Estimating…" : "Run predictions"}
                </button>
                <details className="threshold-control">
                  <summary>
                    <SlidersHorizontal size={16} /> Demo thresholds
                  </summary>
                  <label>
                    Needs review at or below
                    <input
                      type="number"
                      min={0}
                      value={draft.review_threshold}
                      onChange={(e) =>
                        setDraft({
                          ...draft,
                          review_threshold: Number(e.target.value),
                        })
                      }
                    />
                  </label>
                  <label>
                    Monitor at or below
                    <input
                      type="number"
                      min={0}
                      value={draft.monitor_threshold}
                      onChange={(e) =>
                        setDraft({
                          ...draft,
                          monitor_threshold: Number(e.target.value),
                        })
                      }
                    />
                  </label>
                  <p className="small muted">
                    Illustrative cycles, not validated safety limits. Apply with
                    Run predictions.
                  </p>
                </details>
              </form>
              {formError && (
                <div role="alert" className="notice danger">
                  {formError}
                </div>
              )}
              <LoadState
                error={engines.error ?? models.error ?? fleet.error}
                loading={engines.loading || models.loading || fleet.loading}
              />
            </>
          )}
          {page === "fleet" && engines.data && fleet.data && (
            <>
              <div className="stats">
                <div className="stat">
                  <span>Demonstration engines</span>
                  <strong>
                    {engines.data.engines.length.toString().padStart(2, "0")}
                  </strong>
                  <p>Held-out validation engines</p>
                </div>
                <div className="stat">
                  <span>Valid estimates</span>
                  <strong>{valid.toString().padStart(2, "0")}</strong>
                  <p>Finite, non-negative outputs</p>
                </div>
                <div className="stat">
                  <span>Invalid · needs manual review</span>
                  <strong className={rows.length - valid ? "amber" : ""}>
                    {(rows.length - valid).toString().padStart(2, "0")}
                  </strong>
                  <p>Never treated as reassuring</p>
                </div>
              </div>
              <div className="fleet-grid">
                <div>
                  {prediction ? (
                    <HealthCard key={selected} prediction={prediction} />
                  ) : (
                    <LoadState empty />
                  )}
                  <section className="panel explanation">
                    <h3>Read the estimate, not a failure clock.</h3>
                    <p>
                      Remaining useful life is the model’s estimate of how many
                      operating cycles may remain before its learned end-of-life
                      condition. Actual failure timing may differ.
                    </p>
                    <a href="#/reliability">
                      Understand the limitations <ArrowRight size={15} />
                    </a>
                  </section>
                </div>
                <section className="panel fleet-list">
                  <div className="section-top">
                    <h2>Engine priority list</h2>
                    <span className="small muted">{rows.length} engines</span>
                  </div>
                  <p className="small muted">
                    Ordered by raw estimated life. Select an engine.
                  </p>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Engine</th>
                          <th>Raw cycles</th>
                          <th>Interpretation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((row) => (
                          <tr
                            className={
                              row.engine_id === selected ? "selected" : ""
                            }
                            key={row.engine_id}
                          >
                            <th>
                              <button
                                aria-label={`Select engine ${row.engine_id}`}
                                aria-pressed={row.engine_id === selected}
                                onClick={() => setSelected(row.engine_id)}
                              >
                                E-{String(row.engine_id).padStart(3, "0")}
                              </button>
                            </th>
                            <td>
                              {row.raw_rul === null ||
                              !Number.isFinite(row.raw_rul)
                                ? "Unavailable"
                                : row.raw_rul.toFixed(1)}
                            </td>
                            <td>
                              <span className={`status-dot ${row.level}`} />
                              {row.status}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="small muted">
                    Complete demonstration trajectories include failure
                    endpoints. Replay uses each engine’s last available cycle at
                    or before {settings.cycle}; it is not a live fleet.
                  </p>
                </section>
              </div>
            </>
          )}
          {page === "engines" && engines.data && (
            <EngineExplorer
              engines={engines.data}
              selected={selected}
              onSelect={setSelected}
              cycle={settings.cycle}
              prediction={prediction}
            />
          )}
          {page === "models" && <Comparison />}
          {page === "reliability" && <ReliabilityPage />}
          <footer>
            <span>AEVORA / Intelligence Before Failure.</span>
            <span>Research demonstration · official test targets unused</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
function InfoMark() {
  return <span className="disclosure-mark">i</span>;
}
