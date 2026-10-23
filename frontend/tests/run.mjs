import { build } from "esbuild";
import { mkdir } from "node:fs/promises";
import { spawnSync } from "node:child_process";
await mkdir(".test-build", { recursive: true });
await build({
  entryPoints: ["tests/ui.test.tsx"],
  outfile: ".test-build/ui.test.mjs",
  bundle: true,
  platform: "node",
  format: "esm",
  packages: "external",
  jsx: "automatic",
});
const result = spawnSync(
  process.execPath,
  ["--test", ".test-build/ui.test.mjs"],
  { stdio: "inherit" },
);
process.exit(result.status ?? 1);
