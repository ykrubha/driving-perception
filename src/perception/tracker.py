"""Multi-object tracker: predict, match, update, repeat.

Two-stage matching follows the idea in ByteTrack. Confident detections are matched first.
Low-confidence detections then get a chance to continue a track that's still unmatched,
which keeps an object through partial occlusion without letting noise start new tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boxes import iou_matrix


@dataclass(frozen=True)
class TrackerConfig:
    high_score: float = 2.0  # Only detections above this can start a track.
    low_score: float = 0.0  # Detections below this are thrown away.
    match_iou: float = 0.2  # Minimum overlap to accept a match.
    max_age: int = 3  # Frames a track can go unmatched before it's dropped.
    min_hits: int = 2  # Matches needed before a track is reported.
    two_stage: bool = True


@dataclass
class Track:
    id: int
    motion: object
    box: np.ndarray
    score: float
    hits: int = 1
    age: int = 1
    misses: int = 0


@dataclass
class TrackOutput:
    id: int
    box: np.ndarray
    score: float


def match(costs: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Best one-to-one assignment by IoU. Returns matches and unmatched rows and columns."""
    rows, cols = costs.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    row_ids, col_ids = linear_sum_assignment(-costs)
    pairs = [(r, c) for r, c in zip(row_ids, col_ids) if costs[r, c] >= threshold]
    matched_rows = {r for r, _ in pairs}
    matched_cols = {c for _, c in pairs}
    return (
        [(int(r), int(c)) for r, c in pairs],
        [r for r in range(rows) if r not in matched_rows],
        [c for c in range(cols) if c not in matched_cols],
    )


class Tracker:
    def __init__(self, motion_factory: Callable[[np.ndarray], object], config: TrackerConfig | None = None):
        self.motion_factory = motion_factory
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[TrackOutput]:
        """Process one frame's detections and return the tracks matched in this frame."""
        config = self.config
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        keep = scores >= config.low_score
        boxes, scores = boxes[keep], scores[keep]

        high = scores >= config.high_score
        high_ids = np.flatnonzero(high)
        low_ids = np.flatnonzero(~high)
        if not config.two_stage:
            low_ids = np.array([], dtype=int)

        predicted = np.array([t.motion.predicted for t in self.tracks]).reshape(-1, 4)
        pairs, open_tracks, open_high = match(iou_matrix(predicted, boxes[high_ids]), config.match_iou)
        assignments = [(t, int(high_ids[d])) for t, d in pairs]

        if len(low_ids) and open_tracks:
            subset = np.array(open_tracks)
            pairs, still_open, _ = match(iou_matrix(predicted[subset], boxes[low_ids]), config.match_iou)
            assignments += [(int(subset[t]), int(low_ids[d])) for t, d in pairs]
            open_tracks = [int(subset[t]) for t in still_open]

        for track_index, detection in assignments:
            track = self.tracks[track_index]
            track.motion.update(boxes[detection])
            track.box = boxes[detection].copy()
            track.score = float(scores[detection])
            track.hits += 1
            track.age += 1
            track.misses = 0

        for track_index in open_tracks:
            track = self.tracks[track_index]
            track.motion.coast()
            track.age += 1
            track.misses += 1

        self.tracks = [t for t in self.tracks if t.misses <= config.max_age]

        for detection in open_high:
            index = int(high_ids[detection])
            self.tracks.append(Track(self._next_id, self.motion_factory(boxes[index]),
                                     boxes[index].copy(), float(scores[index])))
            self._next_id += 1

        return [
            TrackOutput(t.id, t.box.copy(), t.score)
            for t in self.tracks
            if t.misses == 0 and t.hits >= config.min_hits
        ]
