from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

FILE_RE = re.compile(
    r"(?P<router>.+?)_(?P<label>Throughput|Signal Strength) for (?P<floor>Ground Floor|Lower Floor|Upper Floor) on (?P<band>2\.4 GHz|5 GHz) band_output\.csv",
    re.IGNORECASE,
)


def infer_metric(filename: str) -> str | None:
    match = FILE_RE.match(filename)
    if not match:
        return None
    label = match.group("label").lower()
    return "throughput" if "throughput" in label else "signal_strength"


def main() -> None:
    parser = argparse.ArgumentParser(description="Move flat CSV exports into topology/metric/router folders.")
    parser.add_argument("source", help="Folder containing flat CSV files")
    parser.add_argument("destination", help="Root folder, e.g. data/compare_inputs/with_mesh")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.destination)
    if not source.exists():
        raise SystemExit(f"Source folder not found: {source}")

    moved = 0
    for csv_path in source.glob("*.csv"):
        metric = infer_metric(csv_path.name)
        match = FILE_RE.match(csv_path.name)
        if not metric or not match:
            print(f"Skipped: {csv_path.name}")
            continue
        router = match.group("router")
        target_dir = destination / metric / router
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / csv_path.name
        if args.copy:
            shutil.copy2(csv_path, target_path)
        else:
            shutil.move(str(csv_path), str(target_path))
        moved += 1
        print(f"Placed: {csv_path.name} -> {target_path}")

    print(f"Done. Processed {moved} CSV files.")


if __name__ == "__main__":
    main()
