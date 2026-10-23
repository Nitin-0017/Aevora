import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Scatter,
  ScatterChart,
  ReferenceLine,
} from "recharts";
const grid = "#243141",
  ink = "#9aabbe",
  cyan = "#65d7df";
const tip = {
  backgroundColor: "#172231",
  border: "1px solid #34465d",
  borderRadius: 6,
  color: "#edf4fc",
};
export function Bars({
  data,
  x,
  y,
  label,
}: {
  data: Record<string, unknown>[];
  x: string;
  y: string;
  label: string;
}) {
  return (
    <div className="chart" role="img" aria-label={label}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ left: 5, right: 16, bottom: 24 }}>
          <CartesianGrid stroke={grid} vertical={false} />
          <XAxis
            dataKey={x}
            stroke={ink}
            tick={{ fontSize: 11 }}
            label={{
              value:
                x === "model"
                  ? "Model"
                  : x === "cycles"
                    ? "Error (cycles)"
                    : "Engine",
              position: "bottom",
              fill: ink,
            }}
          />
          <YAxis stroke={ink} tick={{ fontSize: 11 }} />
          <Tooltip contentStyle={tip} />
          <Bar
            dataKey={y}
            name={label}
            fill={cyan}
            radius={[3, 3, 0, 0]}
            maxBarSize={46}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
export function SensorChart({
  data,
  sensor = "Sensor reading",
}: {
  data: { cycle: number; value: number }[];
  sensor?: string;
}) {
  return (
    <div
      className="chart"
      role="img"
      aria-label="Recorded sensor readings by operating cycle"
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ left: 8, right: 16, bottom: 24 }}>
          <CartesianGrid stroke={grid} vertical={false} />
          <XAxis
            dataKey="cycle"
            stroke={ink}
            label={{ value: "Operating cycle", position: "bottom", fill: ink }}
          />
          <YAxis
            domain={["auto", "auto"]}
            label={{
              value: sensor,
              angle: -90,
              position: "insideLeft",
              fill: ink,
            }}
            stroke={ink}
            tick={{ fontSize: 11 }}
          />
          <Tooltip contentStyle={tip} />
          <Line
            type="linear"
            dataKey="value"
            name="Recorded reading"
            stroke={cyan}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
export function ActualChart({
  data,
}: {
  data: { rul: number; predicted_rul: number }[];
}) {
  const low = Math.min(0, ...data.map((p) => p.predicted_rul));
  const high = Math.max(...data.map((p) => Math.max(p.rul, p.predicted_rul)));
  return (
    <div
      className="chart"
      role="img"
      aria-label="Actual versus predicted RUL with ideal prediction reference line"
    >
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ left: 4, right: 12, bottom: 24 }}>
          <CartesianGrid stroke={grid} />
          <XAxis
            type="number"
            dataKey="rul"
            name="Actual"
            domain={[low, high]}
            stroke={ink}
            label={{
              value: "Actual RUL (cycles)",
              position: "bottom",
              fill: ink,
            }}
          />
          <YAxis
            type="number"
            dataKey="predicted_rul"
            name="Predicted"
            domain={[low, high]}
            stroke={ink}
          />
          <Tooltip contentStyle={tip} />
          <ReferenceLine
            segment={[
              { x: low, y: low },
              { x: high, y: high },
            ]}
            stroke="#e9b35f"
            strokeDasharray="4 4"
          />
          <Scatter
            data={data}
            fill={cyan}
            fillOpacity={0.3}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
