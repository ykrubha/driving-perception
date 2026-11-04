"""Score the tuned tracking methods.

    python scripts/evaluate_tracking.py --split test

Settings come from results/tracker_settings.json, and nothing is tuned here.
Writes results/tracking_<split>.json and results/tracking_<split>.md.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from perception.factory import METHODS, make_tracker
from perception.kitti import IMAGE_HEIGHT, IMAGE_WIDTH, TEST_SEQUENCES, TUNE_SEQUENCES, load_sequence
from perception.metrics import run_sequence, summarize


def to_markdown(report: dict) -> str:
    lines = [
        "| Method | IDF1 | MOTA | ID switches | Fragments | Recall | Precision | Mostly tracked | Mostly lost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, data in report["methods"].items():
        o = data["metrics"]["OVERALL"]
        lines.append(
            f"| {method} | {o['idf1']:.3f} | {o['mota']:.3f} | {int(o['num_switches'])} | "
            f"{int(o['num_fragmentations'])} | {o['recall']:.3f} | {o['precision']:.3f} | "
            f"{int(o['mostly_tracked'])} | {int(o['mostly_lost'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ab3dmot")
    parser.add_argument("--split", default="tune", choices=["tune", "test"])
    parser.add_argument("--settings", default="results/tracker_settings.json")
    args = parser.parse_args()

    tuned = json.loads(Path(args.settings).read_text())
    indices = TEST_SEQUENCES if args.split == "test" else TUNE_SEQUENCES
    sequences = [load_sequence(args.root, index) for index in indices]

    report = {"split": args.split, "sequences": indices, "methods": {}}
    for method in METHODS:
        settings = tuned["methods"][method]["settings"]
        make = partial(make_tracker, method, settings, IMAGE_HEIGHT, IMAGE_WIDTH, tuned["kalman_params"])
        accumulators = {sequence.index: run_sequence(sequence, make) for sequence in sequences}
        report["methods"][method] = {"settings": settings, "metrics": summarize(accumulators)}

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"tracking_{args.split}.json").write_text(json.dumps(report, indent=1))
    table = to_markdown(report)
    (out / f"tracking_{args.split}.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
