"""Motion models that predict where a tracked box will be in the next frame.

Each model implements ``update(box)``, called with a measured box, ``coast()``, called when
the object wasn't detected, and ``predicted`` for the box expected in the next frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .boxes import cxcywh_to_xyxy, xyxy_to_cxcywh


class StationaryMotion:
    """Baseline: assume the object stays where it was last seen."""

    def __init__(self, box: np.ndarray):
        self.predicted = np.asarray(box, dtype=float).copy()

    def update(self, box: np.ndarray) -> None:
        self.predicted = np.asarray(box, dtype=float).copy()

    def coast(self) -> None:
        pass


class KalmanMotion:
    """Constant-velocity Kalman filter over [cx, cy, w, h] and their velocities.

    Noise scales with box height, because a nearby car moves more pixels per frame than a
    distant one and its box jitters more.
    """

    F = np.eye(8)
    F[0:4, 4:8] = np.eye(4)
    H = np.eye(4, 8)

    def __init__(self, box: np.ndarray, position_noise: float = 0.05, velocity_noise: float = 0.0125,
                 measurement_noise: float = 0.05):
        self.position_noise = position_noise
        self.velocity_noise = velocity_noise
        self.measurement_noise = measurement_noise
        measured = xyxy_to_cxcywh(box)
        self.x = np.concatenate([measured, np.zeros(4)])
        h = max(measured[3], 1.0)
        self.P = np.diag(np.square(np.array(
            [2 * position_noise * h] * 4 + [10 * velocity_noise * h] * 4
        )))
        self._predict()

    def _predict(self) -> None:
        h = max(self.x[3], 1.0)
        Q = np.diag(np.square(np.array([self.position_noise * h] * 4 + [self.velocity_noise * h] * 4)))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.predicted = cxcywh_to_xyxy(self.x[:4])

    def update(self, box: np.ndarray) -> None:
        z = xyxy_to_cxcywh(box)
        h = max(self.x[3], 1.0)
        R = np.eye(4) * (self.measurement_noise * h) ** 2
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self._predict()

    def coast(self) -> None:
        self._predict()


# Features are scale-free, so a model trained at one image size can run at another.
NUM_FEATURES = 9


def box_features(previous: np.ndarray, current: np.ndarray, observed: float, image_height: float,
                 image_width: float) -> np.ndarray:
    """Features of one step, from two [cx, cy, w, h] boxes. Works on arrays of any leading shape."""
    h = np.maximum(previous[..., 3], 1.0)
    return np.stack([
        (current[..., 0] - previous[..., 0]) / h,
        (current[..., 1] - previous[..., 1]) / h,
        np.log(np.maximum(current[..., 2], 1.0) / np.maximum(previous[..., 2], 1.0)),
        np.log(np.maximum(current[..., 3], 1.0) / np.maximum(previous[..., 3], 1.0)),
        (current[..., 0] - image_width / 2) / image_height,
        (current[..., 1] - image_height / 2) / image_height,
        np.log(np.maximum(current[..., 2], 1.0) / image_height),
        np.log(np.maximum(current[..., 3], 1.0) / image_height),
        np.full(current.shape[:-1], float(observed)),
    ], axis=-1)


def apply_delta(box: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a predicted [dx/h, dy/h, dlog w, dlog h] to a [cx, cy, w, h] box."""
    h = np.maximum(box[..., 3], 1.0)
    return np.stack([
        box[..., 0] + delta[..., 0] * h,
        box[..., 1] + delta[..., 1] * h,
        np.maximum(box[..., 2] * np.exp(np.clip(delta[..., 2], -1, 1)), 1.0),
        np.maximum(box[..., 3] * np.exp(np.clip(delta[..., 3], -1, 1)), 1.0),
    ], axis=-1)


@dataclass
class GruWeights:
    """Weights of a one-layer GRU with a linear output, run in NumPy with no torch needed."""

    w_input: np.ndarray
    w_hidden: np.ndarray
    b_input: np.ndarray
    b_hidden: np.ndarray
    w_output: np.ndarray
    b_output: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "GruWeights":
        data = np.load(path)
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        np.savez(path, **{name: getattr(self, name) for name in self.__dataclass_fields__})

    @property
    def hidden_size(self) -> int:
        return self.w_hidden.shape[1]

    def step(self, features: np.ndarray, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One GRU step, using the same equations and weight layout as torch.nn.GRU."""
        x = (features - self.feature_mean) / self.feature_std
        gates = self.w_input @ x + self.b_input + self.w_hidden @ hidden + self.b_hidden
        size = self.hidden_size
        reset = _sigmoid(gates[:size])
        update = _sigmoid(gates[size:2 * size])
        candidate = np.tanh(
            self.w_input[2 * size:] @ x + self.b_input[2 * size:]
            + reset * (self.w_hidden[2 * size:] @ hidden + self.b_hidden[2 * size:])
        )
        new_hidden = (1 - update) * candidate + update * hidden
        return self.w_output @ new_hidden + self.b_output, new_hidden


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


class GruMotion:
    """Learned motion model. It sees recent motion and image position, so unlike a
    constant-velocity filter it can learn perspective: boxes grow and move faster as they near."""

    def __init__(self, box: np.ndarray, weights: GruWeights, image_height: float, image_width: float):
        self.weights = weights
        self.image_height = image_height
        self.image_width = image_width
        self.hidden = np.zeros(weights.hidden_size)
        self.current = xyxy_to_cxcywh(box)
        self._advance(self.current, observed=1.0)

    def _advance(self, box: np.ndarray, observed: float) -> None:
        features = box_features(self.current, box, observed, self.image_height, self.image_width)
        delta, self.hidden = self.weights.step(features, self.hidden)
        self.current = box
        self._next = apply_delta(box, delta)
        self.predicted = cxcywh_to_xyxy(self._next)

    def update(self, box: np.ndarray) -> None:
        self._advance(xyxy_to_cxcywh(box), observed=1.0)

    def coast(self) -> None:
        self._advance(self._next, observed=0.0)
