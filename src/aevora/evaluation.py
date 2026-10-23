"""Compare saved classical models on shared development-validation endpoints."""

import argparse
import json
import os
import numpy as np
import pandas as pd
from .classical import Predictor
from .config import load_config
from .preprocessing import META
from .training import digest, load_prepared, metrics

MODELS = ("baseline", "ridge", "gradient_boosting")


def align_predictions(reference, candidate):
    """Reject reordered/missing/duplicate endpoints or different target vectors."""
    columns = META + ["rul"]
    if (candidate.duplicated(META).any() or not candidate[columns].equals(reference[columns])):
        raise ValueError("Validation target alignment differs between models")
    metrics(reference.rul.to_numpy(), candidate.predicted_rul.to_numpy())


def error_tables(frame):
    """Near failure [0,30], middle (30,100], early (100,infinity), in cycles."""
    engines, bands = [], []
    for unit, group in frame.groupby("unit_id", sort=True):
        engines.append({"unit_id": int(unit), "observations": len(group),
                        **metrics(group.rul.to_numpy(), group.predicted_rul.to_numpy())})
    for name, mask in [("near_failure_0_30", frame.rul <= 30),
                       ("middle_31_100", (frame.rul > 30) & (frame.rul <= 100)),
                       ("early_over_100", frame.rul > 100)]:
        group = frame[mask]
        bands.append({"band": name, "observations": len(group),
                      "engines": int(group.unit_id.nunique()),
                      **(metrics(group.rul.to_numpy(), group.predicted_rul.to_numpy()) if len(group)
                         else {"mae": None, "rmse": None})})
    return engines, bands


def compare_models(settings):
    manifest, batches = load_prepared(settings)
    batch = batches["validation"]
    reference = batch.endpoints.copy()
    reference["rul"] = batch.y
    frames, reports, engine_rows, band_rows = {}, [], [], []
    prepared_dir = settings.artifact_dir / "prepared" / settings.dataset
    hashes = {p.name: digest(p) for p in sorted(prepared_dir.iterdir()) if p.is_file()}
    for name in MODELS:
        directory = settings.artifact_dir / "models" / settings.dataset / name
        report = json.loads((directory / "metadata.json").read_text())
        if report["prepared_sha256"] != hashes:
            raise ValueError(f"{name} was trained on different preparation artifacts")
        frame = pd.read_csv(directory / "validation_predictions.csv")
        align_predictions(reference, frame)
        predicted = Predictor.load(directory).predict(batch.x, manifest["feature_names"])
        np.testing.assert_allclose(predicted, frame.predicted_rul, rtol=1e-12, atol=1e-12)
        score = metrics(batch.y, predicted)
        for key in score:
            if not np.isclose(score[key], report[key], rtol=1e-12, atol=1e-12):
                raise ValueError(f"{name} metadata metrics differ from reloaded predictions")
        frame["predicted_rul"] = predicted
        frames[name] = frame
        engines, bands = error_tables(frame)
        engine_rows.extend({"model": name, **row} for row in engines)
        band_rows.extend({"model": name, **row} for row in bands)
        reports.append({"model": name, **score, "observations": len(frame),
                        "engines": int(frame.unit_id.nunique()),
                        "engine_macro_mae": float(np.mean([row["mae"] for row in engines])),
                        "negative_predictions": int((predicted < 0).sum()),
                        "fit_seconds": report["fit_seconds"], "search_fit_seconds": report["search_fit_seconds"],
                        "selected_parameters": report["trials"][report["selected_trial"]]["parameters"]})
    output = settings.artifact_dir / "reports" / settings.dataset / "validation"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(engine_rows).to_csv(output / "errors_by_engine.csv", index=False)
    pd.DataFrame(band_rows).to_csv(output / "errors_by_rul_band.csv", index=False)
    pd.DataFrame(reports).drop(columns="selected_parameters").to_csv(output / "metrics.csv", index=False)
    result = {"scope": "Development validation used for hyperparameter selection; not final test performance",
              "residual_definition": "prediction minus actual RUL",
              "source_sha256": manifest["source_sha256"], "models": reports,
              "rul_bands": band_rows, "prepared_sha256": hashes}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_errors(frames, pd.DataFrame(engine_rows), output, settings.artifact_dir)
    return result


def plot_errors(frames, engines, output, artifact_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(artifact_dir / ".matplotlib"))
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    all_values = np.concatenate([f[["rul", "predicted_rul"]].to_numpy().ravel() for f in frames.values()])
    limits = (min(0, float(all_values.min())), float(all_values.max()) + 5)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, (name, frame) in zip(axes, frames.items()):
        ax.scatter(frame.rul, frame.predicted_rul, s=5, alpha=.15, rasterized=True)
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set(title=name, xlabel="Actual RUL (cycles)", ylabel="Predicted RUL (cycles)", xlim=limits, ylim=limits)
    fig.suptitle("Development validation · all 20-cycle endpoints")
    fig.tight_layout()
    fig.savefig(output / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    residuals = {name: frame.predicted_rul - frame.rul for name, frame in frames.items()}
    bins = np.linspace(min(r.min() for r in residuals.values()), max(r.max() for r in residuals.values()), 51)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, (name, residual) in zip(axes, residuals.items()):
        ax.hist(residual, bins=bins)
        ax.axvline(0, color="black", linestyle="--")
        ax.set(title=name, xlabel="Prediction − actual (cycles)", ylabel="Observations")
    fig.tight_layout()
    fig.savefig(output / "residual_distributions.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    engines.pivot(index="unit_id", columns="model", values="mae").plot.bar(ax=ax, width=.8)
    ax.set(xlabel="Validation engine", ylabel="Mean absolute error (cycles)", title="Error by engine · development validation")
    fig.tight_layout()
    fig.savefig(output / "absolute_error_by_engine.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(compare_models(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
