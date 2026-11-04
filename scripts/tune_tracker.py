"""Choose settings for each tracking method using only the tuning sequences.

    python scripts/tune_tracker.py

Kalman noise is tuned first, so the learned model competes with a fair baseline and not an
untuned one. Test sequences are never loaded here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from perception.boxes import iou_matrix
from perception.factory import METHODS, make_tracker
from perception.kitti import (
    IMAGE_HEIGHT, IMAGE_WIDTH, MOTION_VAL_SEQUENCES, TUNE_SEQUENCES, car_trajectories, load_sequence,
)
from perception.metrics import run_sequence, summarize
from perception.motion import KalmanMotion

KALMAN_GRID = {
    "position_noise": [0.01, 0.025, 0.05, 0.1],
    "velocity_noise": [0.00625, 0.0125, 0.025, 0.05, 0.1],
    "measurement_noise": [0.01, 0.025, 0.05, 0.1],
}
MATCH_IOU = [0.1, 0.2, 0.3]
MAX_AGE = [1, 3, 5]
MIN_HITS = [1, 2, 3]
HIGH_SCORE = [1.0, 3.0, 5.0, 7.0]
LOW_SCORE = [-1.0, 0.0, 1.0]


def tune_kalman(root: str) -> dict:
    """Pick the noise settings that give the best next-frame overlap on validation trajectories."""
    runs = []
    for index in MOTION_VAL_SEQUENCES:
        runs += car_trajectories(load_sequence(root, index, with_detections=False).labels, min_length=6)
    rng = np.random.default_rng(3)
    noisy = []
    for run in runs:
        w, h = run[:, 2] - run[:, 0], run[:, 3] - run[:, 1]
        jitter = rng.normal(0, 0.03, run.shape) * np.stack([w, h, w, h], axis=1)
        noisy.append(run + jitter)

    best, best_score = None, -1.0
    for values in itertools.product(*KALMAN_GRID.values()):
        params = dict(zip(KALMAN_GRID, values))
        scores = []
        for truth, measured in zip(runs, noisy):
            motion = KalmanMotion(measured[0], **params)
            for step in range(1, len(truth)):
                scores.append(iou_matrix(motion.predicted[None], truth[step][None])[0, 0])
                motion.update(measured[step])
        score = float(np.mean(scores))
        if score > best_score:
            best, best_score = params, score
    print(f"Kalman noise: {best}, next-frame IoU {best_score:.4f}", flush=True)
    return best


def evaluate_settings(job: tuple, root: str, kalman_params: dict) -> dict:
    method, settings = job
    sequences = [load_sequence_cached(root, index) for index in TUNE_SEQUENCES]
    accumulators = {}
    for sequence in sequences:
        make = partial(make_tracker, method, settings, IMAGE_HEIGHT, IMAGE_WIDTH, kalman_params)
        accumulators[sequence.index] = run_sequence(sequence, make)
    overall = summarize(accumulators)["OVERALL"]
    return {"method": method, "settings": settings, "idf1": overall["idf1"], "mota": overall["mota"],
            "switches": overall["num_switches"], "score": round((overall["idf1"] + overall["mota"]) / 2, 5)}


_CACHE: dict = {}


def load_sequence_cached(root: str, index: int):
    if index not in _CACHE:
        _CACHE[index] = load_sequence(root, index)
    return _CACHE[index]


def grid(method: str) -> list[tuple[str, dict]]:
    two_stage = METHODS[method][1]
    jobs = []
    for high, iou, age, hits in itertools.product(HIGH_SCORE, MATCH_IOU, MAX_AGE, MIN_HITS):
        for low in (LOW_SCORE if two_stage else [high]):
            jobs.append((method, {"high_score": high, "low_score": low, "match_iou": iou,
                                  "max_age": age, "min_hits": hits}))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ab3dmot")
    parser.add_argument("--out", default="results/tracker_settings.json")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    started = time.time()
    kalman_params = tune_kalman(args.root)
    best = {}
    with Pool(args.workers) as pool:
        for method in METHODS:
            work = partial(evaluate_settings, root=args.root, kalman_params=kalman_params)
            results = pool.map(work, grid(method), chunksize=4)
            # Ties go to the earliest setting in the grid, so the choice is repeatable.
            top = max(results, key=lambda r: r["score"])
            best[method] = top
            print(f"{method}: {len(results)} settings tried, best {top}, {time.time() - started:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "tune_sequences": TUNE_SEQUENCES, "kalman_params": kalman_params, "methods": best,
    }, indent=1))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
