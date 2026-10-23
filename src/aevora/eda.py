"""Reproducible training-only FD001 exploration; never opens official targets."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from .config import Settings, load_config
from .data.loader import load_split
from .data.validation import SETTINGS, SENSORS, quality_report


def run_eda(settings: Settings) -> Path:
    # Validate before importing plotting or creating output directories.
    frame = load_split(settings, "train")
    output = settings.artifact_dir / "reports" / settings.dataset / "train"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(settings.artifact_dir / ".matplotlib"))
    # Prefer bundled fonts in supporting Matplotlib versions for headless runs.
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = quality_report(frame)
    sizes = frame.groupby("unit_id", sort=True).size().rename("observed_cycles")
    report["cycle_count_summary"] = {key: float(value) for key, value in sizes.describe().items()}
    report["source_sha256"] = hashlib.sha256(settings.train_path.read_bytes()).hexdigest()
    report["source_file"] = str(settings.train_path)
    report["scope"] = "training trajectories only; no RUL labels or test targets"
    report["versions"] = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "matplotlib"]}
    report["versions"]["python"] = platform.python_version()
    variable = [name for name in SENSORS if name not in report["constant_sensors"]]
    selected = variable[:3] or SENSORS[:3]
    units = sorted(frame.unit_id.unique())[:3]
    report["plot_selection"] = {"sensors": selected, "engines": [int(x) for x in units]}
    (output / "quality_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    frame[SETTINGS + SENSORS].describe().T.to_csv(output / "sensor_setting_summary.csv")
    sizes.to_csv(output / "engine_cycle_counts.csv")
    frame.groupby("unit_id")[SENSORS].agg(["mean", "std", "min", "max"]).to_csv(output / "per_engine_sensor_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sizes, bins="auto", color="#0891b2", edgecolor="white")
    ax.set(xlabel="Observed cycles per training engine", ylabel="Engines", title="AEVORA · Training history lengths")
    fig.tight_layout()
    fig.savefig(output / "engine_cycle_distribution.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(selected), figsize=(12, 3.5), squeeze=False)
    for ax, sensor in zip(axes[0], selected):
        ax.hist(frame[sensor], bins=40, color="#0891b2")
        ax.set(title=sensor, xlabel="Recorded value", ylabel="Rows")
    fig.suptitle("Selected sensor distributions · pooled training rows")
    fig.tight_layout()
    fig.savefig(output / "sensor_distributions.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(len(selected), 1, figsize=(9, 8), squeeze=False)
    for ax, sensor in zip(axes[:, 0], selected):
        for unit in units:
            history = frame.loc[frame.unit_id == unit]
            ax.plot(history.cycle, history[sensor], label=f"Engine {unit}", linewidth=1)
        ax.set(xlabel="Operating cycle", ylabel=sensor)
        ax.legend()
    fig.suptitle("Selected training histories · observations, not failure predictions")
    fig.tight_layout()
    fig.savefig(output / "sensor_histories.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(selected), figsize=(12, 3.5), squeeze=False)
    stds = frame.groupby("unit_id")[selected].std()
    for ax, sensor in zip(axes[0], selected):
        ax.hist(stds[sensor].dropna(), bins="auto", color="#d97706")
        ax.set(title=sensor, xlabel="Within-engine standard deviation", ylabel="Engines")
    fig.tight_layout()
    fig.savefig(output / "sensor_variation_by_engine.png", dpi=140)
    plt.close(fig)

    findings = (
        "# AEVORA FD001 training exploration\n\n"
        f"Observed {len(frame):,} rows, {len(frame.columns)} columns, {len(sizes)} engines.\n\n"
        f"History lengths: min {sizes.min()}, max {sizes.max()}, mean {sizes.mean():.2f}, "
        f"median {sizes.median():.2f} cycles.\n\n"
        f"Missing: {report['missing_values']}; nonfinite: {report['nonfinite_values']}; "
        f"duplicate engine-cycle records: {report['duplicate_engine_cycle_records']}.\n\n"
        f"Exactly constant sensors: {', '.join(report['constant_sensors']) or 'none'}.\n\n"
        f"Near-constant, excluding constant: {', '.join(report['near_constant_sensors']) or 'none'}. "
        "Threshold: one exact recorded value accounts for at least 99% of training rows.\n\n"
        "Plot selection is deterministic: first three nonconstant sensors and lowest three engine IDs. "
        "Pooled histograms give longer histories more weight. Per-engine summaries are also provided.\n\n"
        "These are descriptive observations. Variation does not establish a physical cause or prove "
        "impending failure. Constant detection is not automatic feature selection. No training labels, "
        "features, models, test-target analysis, or performance estimates were produced.\n\n"
        f"Source SHA256: `{report['source_sha256']}`\n"
    )
    (output / "findings.md").write_text(findings)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        output = run_eda(load_config(args.config))
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"EDA unavailable: {exc}. See docs/dataset.md and environment setup.\n")
    print(f"Training-only EDA written to {output}")


if __name__ == "__main__":
    main()
