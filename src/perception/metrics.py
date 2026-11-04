"""KITTI-style CLEAR MOT and identity metrics, computed with py-motmetrics."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from .boxes import intersection_over_second, iou_matrix
from .kitti import Sequence
from .tracker import TrackOutput

MATCH_IOU = 0.5
IGNORE_IOU = 0.5

METRICS = [
    "idf1", "mota", "motp", "recall", "precision", "num_switches", "num_fragmentations",
    "num_false_positives", "num_misses", "mostly_tracked", "mostly_lost", "num_unique_objects",
    "num_objects",
]


def filter_ignored(tracks: list[TrackOutput], truth) -> list[TrackOutput]:
    """Drop outputs that overlap an ignored object and no scored car, so they aren't false positives."""
    if not tracks:
        return []
    boxes = np.array([t.box for t in tracks])
    on_care = iou_matrix(boxes, truth.care_boxes).max(axis=1, initial=0.0) >= IGNORE_IOU
    on_ignore = iou_matrix(boxes, truth.ignore_boxes).max(axis=1, initial=0.0) >= IGNORE_IOU
    in_dontcare = intersection_over_second(boxes, truth.dontcare_boxes).max(axis=1, initial=0.0) >= IGNORE_IOU
    return [t for t, care, ignored, dontcare in zip(tracks, on_care, on_ignore, in_dontcare)
            if care or not (ignored or dontcare)]


def run_sequence(sequence: Sequence, make_tracker: Callable[[], object]):
    """Run a tracker on one sequence and accumulate its results against the ground truth."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import motmetrics as mm

    tracker = make_tracker()
    accumulator = mm.MOTAccumulator(auto_id=True)
    empty = (np.zeros((0, 4)), np.zeros(0))
    for frame in range(sequence.num_frames):
        boxes, scores = sequence.detections.get(frame, empty)
        outputs = tracker.update(boxes, scores)
        truth = sequence.truth(frame)
        kept = filter_ignored(outputs, truth)
        hypotheses = np.array([t.box for t in kept]).reshape(-1, 4)
        distances = 1.0 - iou_matrix(truth.care_boxes, hypotheses)
        distances[distances > 1.0 - MATCH_IOU] = np.nan
        accumulator.update(truth.care_ids.tolist(), [t.id for t in kept], distances)
    return accumulator


def summarize(accumulators: dict) -> dict:
    """Return metrics per sequence and overall. Overall pools all frames, as the MOT metrics define."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import motmetrics as mm

        metrics_host = mm.metrics.create()
        names = [str(name) for name in accumulators]
        summary = metrics_host.compute_many(
            list(accumulators.values()), metrics=METRICS, names=names, generate_overall=True,
        )
    result = {}
    for name, row in summary.iterrows():
        values = {}
        for metric in METRICS:
            value = float(row[metric])
            values[metric] = None if np.isnan(value) else round(value, 4)
        result[str(name)] = values
    return result
