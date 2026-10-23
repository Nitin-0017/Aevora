"""Foundation check only: does not load datasets or train models."""

import argparse
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check AEVORA configuration (no dataset access)")
    parser.add_argument("--config", required=True, help="Path to a TOML configuration")
    args = parser.parse_args()
    settings = load_config(args.config)
    print("AEVORA — Intelligence Before Failure.")
    print(f"Configuration valid | dataset={settings.dataset} | seed={settings.random_seed}")
    print(f"Expected training data: {settings.train_path}")
    print("Phase 1 foundation only. No data read, model trained, or performance measured.")


if __name__ == "__main__":
    main()
