"""Compare motion models at predicting future boxes on held-out KITTI trajectories.

    python scripts/evaluate_forecast.py --split test

Each model sees five frames of a trajectory, then predicts 1, 3 and 5 frames ahead with no
further measurements. That's what happens when a tracker coasts through a missed detection.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from perception.boxes import cxcywh_to_xyxy, iou_matrix, xyxy_to_cxcywh
from perception.factory import motion_factory
from perception.kitti import (
    IMAGE_HEIGHT, IMAGE_WIDTH, TEST_SEQUENCES, TUNE_SEQUENCES, car_trajectories, load_sequence,
)

OBSERVED = 5
HORIZONS = (1, 3, 5)
NOISE = 0.03


class ConstantVelocity:
    """Baseline: repeat the motion between the last two boxes."""

    def __init__(self, box):
        self.current = xyxy_to_cxcywh(box)
        self.velocity = np.zeros(4)
        self.predicted = box.copy()

    def update(self, box):
        measured = xyxy_to_cxcywh(box)
        self.velocity = measured - self.current
        self.current = measured
        self._set_prediction()

    def coast(self):
        self.current = self.current + self.velocity
        self._set_prediction()

    def _set_prediction(self):
        following = self.current + self.velocity
        following[2:] = np.maximum(following[2:], 1.0)
        self.predicted = cxcywh_to_xyxy(following)


def forecast(make_motion, measured: np.ndarray) -> dict[int, np.ndarray]:
    motion = make_motion(measured[0])
    for step in range(1, OBSERVED):
        motion.update(measured[step])
    predictions = {}
    for horizon in range(1, max(HORIZONS) + 1):
        predictions[horizon] = motion.predicted.copy()
        motion.coast()
    return predictions


def bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int = 11, samples: int = 2000) -> list[float]:
    """95% interval for mean(a) - mean(b), resampling trajectories in pairs."""
    rng = np.random.default_rng(seed)
    differences = a - b
    picks = rng.integers(0, len(differences), (samples, len(differences)))
    means = differences[picks].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ab3dmot")
    parser.add_argument("--split", default="tune", choices=["tune", "test"])
    parser.add_argument("--settings", default="results/tracker_settings.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sequences = TEST_SEQUENCES if args.split == "test" else TUNE_SEQUENCES
    runs = []
    for index in sequences:
        runs += car_trajectories(load_sequence(args.root, index, with_detections=False).labels,
                                 min_length=OBSERVED + max(HORIZONS) + 1)
    # Use non-overlapping windows, so one long track doesn't produce near-duplicate samples.
    span = OBSERVED + max(HORIZONS)
    windows = [run[start:start + span] for run in runs for start in range(0, len(run) - span + 1, span)]

    kalman_params = json.loads(Path(args.settings).read_text())["kalman_params"]
    models = {
        "last_position": motion_factory("stationary", IMAGE_HEIGHT, IMAGE_WIDTH),
        "constant_velocity": ConstantVelocity,
        "kalman": partial(motion_factory("kalman", IMAGE_HEIGHT, IMAGE_WIDTH, kalman_params)),
        "gru": motion_factory("gru", IMAGE_HEIGHT, IMAGE_WIDTH),
    }

    report = {"split": args.split, "sequences": sequences, "windows": len(windows), "noise": NOISE, "models": {}}
    per_window = {}
    for condition, noise in (("clean", 0.0), ("noisy", NOISE)):
        rng = np.random.default_rng(5)
        inputs = []
        for window in windows:
            w, h = window[:, 2] - window[:, 0], window[:, 3] - window[:, 1]
            inputs.append(window + rng.normal(0, 1, window.shape) * noise * np.stack([w, h, w, h], axis=1))
        for name, make in models.items():
            ious = {h: [] for h in HORIZONS}
            centers = {h: [] for h in HORIZONS}
            for window, measured in zip(windows, inputs):
                predictions = forecast(make, measured)
                for horizon in HORIZONS:
                    truth = window[OBSERVED + horizon - 1]
                    predicted = predictions[horizon]
                    ious[horizon].append(iou_matrix(predicted[None], truth[None])[0, 0])
                    centers[horizon].append(np.linalg.norm(
                        xyxy_to_cxcywh(predicted)[:2] - xyxy_to_cxcywh(truth)[:2]))
            report["models"].setdefault(condition, {})[name] = {
                f"h{h}": {"iou": round(float(np.mean(ious[h])), 4),
                          "center_px": round(float(np.mean(centers[h])), 2),
                          "center_px_median": round(float(np.median(centers[h])), 2)}
                for h in HORIZONS
            }
            per_window[(condition, name)] = {h: np.array(ious[h]) for h in HORIZONS}
        report.setdefault("gru_minus_kalman_iou_ci95", {})[condition] = {
            f"h{h}": bootstrap_difference(per_window[(condition, "gru")][h], per_window[(condition, "kalman")][h])
            for h in HORIZONS
        }

    out = Path(args.out or f"results/forecast_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
