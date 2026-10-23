"""Validate one trajectory split; never reads official RUL targets."""

import argparse
import json
from aevora.config import load_config
from .loader import load_split
from .validation import DataValidationError, quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()
    try:
        frame = load_split(load_config(args.config), args.split)
    except (OSError, ValueError) as exc:
        if isinstance(exc, DataValidationError) and exc.report:
            print(json.dumps(exc.report, indent=2, allow_nan=False))
        parser.exit(2, f"Validation failed: {exc}\n")
    print(json.dumps(quality_report(frame), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
