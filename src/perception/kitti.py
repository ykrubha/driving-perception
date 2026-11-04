"""Load KITTI tracking labels and detections, and define the fixed data splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed before any results were seen. TEST is only used for reported scores.
TRAIN_SEQUENCES = (0, 2, 3, 4, 5, 9, 17, 20)  # fit the motion model
MOTION_VAL_SEQUENCES = (7, 11)  # early stopping for the motion model
TUNE_SEQUENCES = (1, 6, 8, 10, 12)  # choose tracker settings
TEST_SEQUENCES = (13, 14, 15, 16, 18, 19)  # final scores

IMAGE_HEIGHT = 375.0
IMAGE_WIDTH = 1242.0

# Same care rules as the KITTI tracking devkit. Cars outside them are ignored, not penalized.
MAX_TRUNCATION = 0
MAX_OCCLUSION = 2
MIN_HEIGHT = 25.0

LABEL_COLUMNS = [
    "frame", "track_id", "type", "truncated", "occluded", "alpha",
    "x1", "y1", "x2", "y2", "h3d", "w3d", "l3d", "x3d", "y3d", "z3d", "rotation",
]
BOX = ["x1", "y1", "x2", "y2"]


@dataclass
class FrameTruth:
    care_ids: np.ndarray
    care_boxes: np.ndarray
    ignore_boxes: np.ndarray  # Other classes and hard cars. Matching these costs nothing.
    dontcare_boxes: np.ndarray  # Unlabeled regions, matched by area covered.


@dataclass
class Sequence:
    index: int
    num_frames: int
    labels: pd.DataFrame = field(repr=False)
    detections: dict[int, tuple[np.ndarray, np.ndarray]] = field(repr=False, default_factory=dict)

    def truth(self, frame: int) -> FrameTruth:
        rows = self._by_frame.get(frame)
        empty = np.zeros((0, 4))
        if rows is None:
            return FrameTruth(np.zeros(0, dtype=int), empty, empty, empty)
        cars = rows[rows["type"] == "Car"]
        heights = cars["y2"] - cars["y1"]
        easy = (cars["truncated"] <= MAX_TRUNCATION) & (cars["occluded"] <= MAX_OCCLUSION) & (heights >= MIN_HEIGHT)
        care = cars[easy]
        hard = cars[~easy]
        vans = rows[rows["type"] == "Van"]
        dontcare = rows[rows["type"] == "DontCare"]
        return FrameTruth(
            care["track_id"].to_numpy(dtype=int),
            care[BOX].to_numpy(dtype=float),
            np.vstack([hard[BOX].to_numpy(dtype=float), vans[BOX].to_numpy(dtype=float)]),
            dontcare[BOX].to_numpy(dtype=float),
        )

    def __post_init__(self):
        self._by_frame = {frame: rows for frame, rows in self.labels.groupby("frame")}


def load_labels(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=" ", header=None, names=LABEL_COLUMNS, usecols=range(17))


def load_detections(path: str | Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {frame: (boxes, scores)}. File columns: frame, class, x1, y1, x2, y2, score, ..."""
    table = pd.read_csv(path, header=None)
    result = {}
    for frame, rows in table.groupby(0):
        result[int(frame)] = (rows[[2, 3, 4, 5]].to_numpy(dtype=float), rows[6].to_numpy(dtype=float))
    return result


def load_sequence(root: str | Path, index: int, with_detections: bool = True) -> Sequence:
    root = Path(root)
    labels = load_labels(root / "scripts/KITTI/label" / f"{index:04d}.txt")
    detections: dict = {}
    if with_detections:
        detections = load_detections(root / "data/KITTI/detection/pointrcnn_Car_val" / f"{index:04d}.txt")
    last_detection = max(detections, default=-1)
    num_frames = int(max(labels["frame"].max(), last_detection)) + 1
    return Sequence(index, num_frames, labels, detections)


def car_trajectories(labels: pd.DataFrame, min_length: int = 2) -> list[np.ndarray]:
    """Unbroken runs of labeled boxes for each car, as arrays of [x1, y1, x2, y2]."""
    runs = []
    cars = labels[labels["type"].isin(["Car", "Van"])].sort_values(["track_id", "frame"])
    for _, rows in cars.groupby("track_id"):
        frames = rows["frame"].to_numpy()
        boxes = rows[BOX].to_numpy(dtype=float)
        breaks = np.flatnonzero(np.diff(frames) != 1) + 1
        for run in np.split(boxes, breaks):
            if len(run) >= min_length:
                runs.append(run)
    return runs
