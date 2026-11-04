"""Build trackers from named configurations."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from .motion import GruMotion, GruWeights, KalmanMotion, StationaryMotion
from .tracker import Tracker, TrackerConfig

DEFAULT_WEIGHTS = Path("models/motion_gru.npz")

# name: (motion model, two-stage matching)
METHODS = {
    "iou": ("stationary", False),
    "kalman": ("kalman", False),
    "kalman_two_stage": ("kalman", True),
    "gru": ("gru", False),
    "gru_two_stage": ("gru", True),
}


def motion_factory(kind: str, image_height: float, image_width: float,
                   kalman_params: dict | None = None, weights_path: str | Path = DEFAULT_WEIGHTS):
    if kind == "stationary":
        return StationaryMotion
    if kind == "kalman":
        return partial(KalmanMotion, **(kalman_params or {}))
    if kind == "gru":
        weights = GruWeights.load(weights_path)
        return partial(GruMotion, weights=weights, image_height=image_height, image_width=image_width)
    raise ValueError(f"Unknown motion model {kind!r}.")


def make_tracker(method: str, settings: dict, image_height: float, image_width: float,
                 kalman_params: dict | None = None, weights_path: str | Path = DEFAULT_WEIGHTS) -> Tracker:
    kind, two_stage = METHODS[method]
    return Tracker(
        motion_factory(kind, image_height, image_width, kalman_params, weights_path),
        TrackerConfig(two_stage=two_stage, **settings),
    )
