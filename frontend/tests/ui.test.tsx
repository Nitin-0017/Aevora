// Synthetic fixtures live only in tests; production always uses the FastAPI service.
import test from "node:test";
import assert from "node:assert/strict";
import { renderToStaticMarkup } from "react-dom/server";
import { createRoot } from "react-dom/client";
import { act } from "react";
import { JSDOM } from "jsdom";
import { HealthCard, LoadState } from "../src/components";
import {
  request,
  hoursFromCycles,
  DISCLOSURE,
  type Prediction,
} from "../src/api";
import App from "../src/App";
const prediction: Prediction = {
  engine_id: 3,
  model: "baseline",
  raw_rul: 95,
  valid: true,
  status: "Monitor",
  level: "elevated",
  recommendation: "Review the estimate and continue routine monitoring.",
  reason: "Illustrative monitoring band.",
  latest_cycle: 100,
  available_cycles: 100,
  model_version: 1,
  preprocessing_version: 1,
  generated_at: "2026-10-24T00:00:00Z",
  warning: "Not a guarantee",
  guidance: "Decision support",
};

test("health card presents raw cycles and qualified guidance", () => {
  const html = renderToStaticMarkup(<HealthCard prediction={prediction} />);
  assert.match(html, /95\.00/);
  assert.match(html, /Monitor/);
  assert.match(html, /not a validated operational procedure/);
  assert.match(html, /Operating cycles, not a guaranteed failure date/);
});
test("negative and non-finite predictions are not reassuring", () => {
  for (const raw of [-2, NaN, Infinity, null]) {
    const html = renderToStaticMarkup(
      <HealthCard
        prediction={{ ...prediction, raw_rul: raw, valid: raw !== null }}
      />,
    );
    assert.match(html, /Unable to assess/);
    assert.match(html, /Do not rely on this estimate/);
    if (raw === -2) assert.match(html, /-2\.00/);
  }
});
test("conversion requires valid RUL and user supplied positive rate", () => {
  assert.equal(hoursFromCycles(100, "20"), 5);
  assert.equal(hoursFromCycles(100, ""), null);
  for (const rate of ["0", "-2", "Infinity", "invalid"])
    assert.equal(hoursFromCycles(100, rate), null);
  assert.equal(hoursFromCycles(-2, "20"), null);
  assert.equal(hoursFromCycles(null, "20"), null);
});
test("loading, empty and error states contain actionable text", () => {
  assert.match(renderToStaticMarkup(<LoadState loading />), /role="status"/);
  assert.match(
    renderToStaticMarkup(<LoadState error="Restore the audited bundle" />),
    /Restore the audited bundle/,
  );
  assert.match(
    renderToStaticMarkup(<LoadState empty />),
    /No eligible observations/,
  );
});
test("typed API client renders server errors and rejects unreadable data", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () =>
      ({
        ok: false,
        json: async () => ({ error: { message: "Restore saved models" } }),
      }) as Response;
    await assert.rejects(request("/api/health"), /Restore saved models/);
    globalThis.fetch = async () =>
      ({
        ok: true,
        json: async () => {
          throw Error("invalid");
        },
      }) as Response;
    await assert.rejects(request("/api/health"), /unreadable/);
  } finally {
    globalThis.fetch = original;
  }
});
async function browserTest(callback: (host: HTMLElement) => Promise<void>) {
  const dom = new JSDOM('<div id="root"></div>', {
    url: "http://localhost:5173/#/fleet",
  });
  const originalFetch = globalThis.fetch;
  Object.defineProperty(globalThis, "window", {
    value: dom.window,
    configurable: true,
  });
  Object.defineProperty(globalThis, "document", {
    value: dom.window.document,
    configurable: true,
  });
  Object.defineProperty(globalThis, "navigator", {
    value: dom.window.navigator,
    configurable: true,
  });
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const host = dom.window.document.getElementById("root")!;
  try {
    await callback(host);
  } finally {
    globalThis.fetch = originalFetch;
    dom.window.close();
  }
}
test("fleet API response, demo disclosure and engine selection render correctly", async () =>
  browserTest(async (host) => {
    globalThis.fetch = async (url) =>
      ({
        ok: true,
        json: async () =>
          String(url) === "/api/engines"
            ? {
                engines: [
                  { engine_id: 3, available_cycles: 150 },
                  { engine_id: 5, available_cycles: 170 },
                ],
                sensors: ["sensor_2"],
                min_cycle: 20,
                max_cycle: 170,
                disclosure: DISCLOSURE,
              }
            : String(url) === "/api/models"
              ? { models: [{ id: "baseline", name: "Median baseline" }] }
              : {
                  predictions: [
                    prediction,
                    {
                      ...prediction,
                      engine_id: 5,
                      raw_rul: -2,
                      valid: false,
                      status: "Unable to assess",
                      level: "invalid",
                    },
                  ],
                  generated_at: prediction.generated_at,
                  disclosure: DISCLOSURE,
                },
      }) as Response;
    const root = createRoot(host);
    try {
      await act(async () => root.render(<App />));
      assert.ok(host.textContent?.includes(DISCLOSURE));
      assert.match(host.textContent ?? "", /95\.00/);
      const select = host.querySelector(
        'button[aria-label="Select engine 5"]',
      ) as HTMLButtonElement;
      assert.ok(select);
      await act(async () => select.click());
      assert.match(
        host.querySelector(".health-card")?.textContent ?? "",
        /Engine 005/,
      );
      assert.match(
        host.querySelector(".health-card")?.textContent ?? "",
        /-2\.00/,
      );
      assert.match(
        host.querySelector(".health-card")?.textContent ?? "",
        /Unable to assess/,
      );
    } finally {
      await act(async () => root.unmount());
    }
  }));
test("initial loading and backend failure do not invent fleet values", async () =>
  browserTest(async (host) => {
    globalThis.fetch = async () =>
      ({
        ok: false,
        json: async () => ({ error: { message: "Backend unavailable" } }),
      }) as Response;
    const root = createRoot(host);
    try {
      await act(async () => root.render(<App />));
      assert.match(host.textContent ?? "", /Backend unavailable/);
      assert.equal(host.querySelector(".health-card"), null);
      assert.equal(host.querySelector(".stats"), null);
    } finally {
      await act(async () => root.unmount());
    }
  }));
