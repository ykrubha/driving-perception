"""Box helpers. Boxes are float arrays of [x1, y1, x2, y2]."""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between N and M boxes, returned as an N x M array."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def intersection_over_second(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of each box in ``a`` covered by each box in ``b``, for ignore regions."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 1e-9, None) * np.clip(a[:, 3] - a[:, 1], 1e-9, None)
    return intersection / area_a[:, None]


def xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=float)
    return np.stack([
        (box[..., 0] + box[..., 2]) / 2,
        (box[..., 1] + box[..., 3]) / 2,
        box[..., 2] - box[..., 0],
        box[..., 3] - box[..., 1],
    ], axis=-1)


def cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=float)
    return np.stack([
        box[..., 0] - box[..., 2] / 2,
        box[..., 1] - box[..., 3] / 2,
        box[..., 0] + box[..., 2] / 2,
        box[..., 1] + box[..., 3] / 2,
    ], axis=-1)
