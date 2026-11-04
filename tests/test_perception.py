"""Tests that need no downloaded data. They use synthetic scenes with known answers."""

import numpy as np
import pytest

from perception.boxes import cxcywh_to_xyxy, intersection_over_second, iou_matrix, xyxy_to_cxcywh
from perception.kitti import FrameTruth, car_trajectories, load_labels
from perception.metrics import filter_ignored, summarize
from perception.motion import (
    GruMotion, GruWeights, KalmanMotion, NUM_FEATURES, StationaryMotion, apply_delta, box_features,
)
from perception.tracker import Tracker, TrackerConfig, TrackOutput, match

CONFIG = TrackerConfig(high_score=0.5, low_score=0.1, match_iou=0.2, max_age=3, min_hits=2, two_stage=True)


def box_at(x, y=100.0, w=50.0, h=40.0):
    return np.array([x, y, x + w, y + h], dtype=float)


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert iou_matrix(a, b) == pytest.approx(np.array([[1.0, 50 / 150, 0.0]]))
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)
    assert iou_matrix(np.zeros((1, 4)), np.zeros((1, 4)))[0, 0] == 0


def test_intersection_over_second_measures_share_covered():
    small = np.array([[2, 2, 4, 4]], dtype=float)
    large = np.array([[0, 0, 10, 10]], dtype=float)
    assert intersection_over_second(small, large)[0, 0] == pytest.approx(1.0)
    assert intersection_over_second(large, small)[0, 0] == pytest.approx(0.04)


def test_box_conversion_round_trips():
    box = np.array([[10.0, 20.0, 50.0, 80.0]])
    assert cxcywh_to_xyxy(xyxy_to_cxcywh(box)) == pytest.approx(box)


def test_match_respects_threshold_and_is_one_to_one():
    costs = np.array([[0.9, 0.8], [0.85, 0.05]])
    pairs, rows, cols = match(costs, 0.3)
    assert sorted(pairs) == [(0, 1), (1, 0)]
    assert rows == [] and cols == []
    pairs, rows, cols = match(np.array([[0.1]]), 0.3)
    assert pairs == [] and rows == [0] and cols == [0]


def test_kalman_learns_constant_velocity():
    motion = KalmanMotion(box_at(0))
    for step in range(1, 20):
        motion.update(box_at(step * 10.0))
    assert motion.predicted == pytest.approx(box_at(200.0), abs=1.0)


def test_kalman_keeps_moving_while_coasting():
    motion = KalmanMotion(box_at(0))
    for step in range(1, 20):
        motion.update(box_at(step * 10.0))
    motion.coast()
    motion.coast()
    assert motion.predicted[0] == pytest.approx(220.0, abs=2.0)


def test_stationary_stays_put():
    motion = StationaryMotion(box_at(5))
    motion.coast()
    assert motion.predicted == pytest.approx(box_at(5))


def test_numpy_gru_matches_torch():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    gru = torch.nn.GRU(NUM_FEATURES, 8, batch_first=True)
    head = torch.nn.Linear(8, 4)
    weights = GruWeights(
        gru.weight_ih_l0.detach().numpy(), gru.weight_hh_l0.detach().numpy(),
        gru.bias_ih_l0.detach().numpy(), gru.bias_hh_l0.detach().numpy(),
        head.weight.detach().numpy(), head.bias.detach().numpy(),
        np.zeros(NUM_FEATURES), np.ones(NUM_FEATURES),
    )
    inputs = torch.randn(1, 6, NUM_FEATURES)
    with torch.no_grad():
        expected, _ = gru(inputs)
        expected = head(expected)[0].numpy()
    hidden = np.zeros(8)
    for step in range(6):
        output, hidden = weights.step(inputs[0, step].numpy().astype(float), hidden)
        assert output == pytest.approx(expected[step].astype(float), abs=1e-5)


def test_features_and_delta_are_consistent():
    previous = np.array([100.0, 100.0, 50.0, 40.0])
    current = np.array([110.0, 96.0, 55.0, 44.0])
    features = box_features(previous, current, 1.0, 375.0, 1242.0)
    assert features.shape == (NUM_FEATURES,)
    rebuilt = apply_delta(previous, features[:4])
    assert rebuilt == pytest.approx(current)


def test_gru_motion_runs_and_saves(tmp_path):
    rng = np.random.default_rng(0)
    size = 4
    weights = GruWeights(
        rng.normal(size=(3 * size, NUM_FEATURES)), rng.normal(size=(3 * size, size)),
        np.zeros(3 * size), np.zeros(3 * size), rng.normal(size=(4, size)) * 0.01, np.zeros(4),
        np.zeros(NUM_FEATURES), np.ones(NUM_FEATURES),
    )
    path = tmp_path / "weights.npz"
    weights.save(path)
    loaded = GruWeights.load(path)
    motion = GruMotion(box_at(0), loaded, 375.0, 1242.0)
    motion.update(box_at(5))
    motion.coast()
    assert motion.predicted.shape == (4,) and np.isfinite(motion.predicted).all()


def run_scene(tracker, frames):
    return [tracker.update(boxes, scores) for boxes, scores in frames]


def test_track_needs_min_hits_before_it_is_reported():
    tracker = Tracker(KalmanMotion, CONFIG)
    first = tracker.update(box_at(0)[None], np.array([0.9]))
    second = tracker.update(box_at(5)[None], np.array([0.9]))
    assert first == []
    assert [t.id for t in second] == [1]


def test_identity_survives_a_short_gap_and_dies_after_a_long_one():
    tracker = Tracker(KalmanMotion, CONFIG)
    x = 0.0
    for _ in range(6):
        outputs = tracker.update(box_at(x)[None], np.array([0.9]))
        x += 10
    assert outputs[0].id == 1
    for _ in range(2):
        tracker.update(np.zeros((0, 4)), np.zeros(0))
        x += 10
    assert tracker.update(box_at(x)[None], np.array([0.9]))[0].id == 1

    for _ in range(CONFIG.max_age + 2):
        tracker.update(np.zeros((0, 4)), np.zeros(0))
    assert tracker.tracks == []
    tracker.update(box_at(x)[None], np.array([0.9]))
    assert tracker.tracks[0].id != 1


def test_motion_model_keeps_identity_when_an_object_speeds_up():
    """At 45 px per frame a 50 px box barely overlaps its last position, so only prediction helps."""
    positions = [i * 30.0 for i in range(6)]
    for _ in range(8):
        positions.append(positions[-1] + 45.0)
    frames = [(box_at(x)[None], np.array([0.9])) for x in positions]
    config = TrackerConfig(0.5, 0.1, 0.2, 1, 1, False)
    still = Tracker(StationaryMotion, config)
    moving = Tracker(KalmanMotion, config)
    run_scene(still, frames)
    run_scene(moving, frames)
    assert moving._next_id == 2
    assert still._next_id > 5


def test_low_score_detection_continues_a_track_but_cannot_start_one():
    tracker = Tracker(KalmanMotion, CONFIG)
    for i in range(4):
        tracker.update(box_at(i * 5.0)[None], np.array([0.9]))
    outputs = tracker.update(box_at(20.0)[None], np.array([0.2]))
    assert [t.id for t in outputs] == [1]

    fresh = Tracker(KalmanMotion, CONFIG)
    for i in range(4):
        assert fresh.update(box_at(500.0)[None], np.array([0.2])) == []
    assert fresh.tracks == []


def test_single_stage_ignores_low_score_detections():
    config = TrackerConfig(0.5, 0.1, 0.2, 3, 1, False)
    tracker = Tracker(KalmanMotion, config)
    tracker.update(box_at(0)[None], np.array([0.9]))
    assert tracker.update(box_at(1)[None], np.array([0.2])) == []
    assert tracker.tracks[0].misses == 1


def test_two_crossing_objects_keep_their_identities():
    tracker = Tracker(KalmanMotion, TrackerConfig(0.5, 0.1, 0.1, 3, 1, True))
    identities = {}
    for i in range(30):
        boxes = np.array([box_at(i * 8.0, y=100), box_at(232 - i * 8.0, y=112)])
        for output in tracker.update(boxes, np.array([0.9, 0.9])):
            side = "right" if iou_matrix(output.box[None], boxes[None, 0])[0, 0] > 0.9 else "left"
            identities.setdefault(side, set()).add(output.id)
    assert len(identities["right"]) == 1 and len(identities["left"]) == 1
    assert identities["right"] != identities["left"]


def test_filter_ignored_keeps_scored_cars_and_drops_ignored_regions():
    truth = FrameTruth(
        np.array([1]), box_at(0)[None], box_at(300)[None], np.array([[600.0, 0.0, 800.0, 300.0]]),
    )
    tracks = [
        TrackOutput(1, box_at(2), 1.0),      # on a scored car
        TrackOutput(2, box_at(302), 1.0),    # on an ignored van
        TrackOutput(3, box_at(650), 1.0),    # inside a DontCare region
        TrackOutput(4, box_at(1000), 1.0),   # a real false positive
    ]
    assert [t.id for t in filter_ignored(tracks, truth)] == [1, 4]


def test_metrics_on_a_perfect_and_a_broken_tracker():
    mm = pytest.importorskip("motmetrics")

    def accumulate(ids):
        accumulator = mm.MOTAccumulator(auto_id=True)
        for hypothesis in ids:
            accumulator.update([1], [hypothesis], np.array([[0.1]]))
        return accumulator

    summary = summarize({"perfect": accumulate([7] * 10), "switching": accumulate([7] * 5 + [8] * 5)})
    assert summary["perfect"]["mota"] == 1.0
    assert summary["perfect"]["idf1"] == 1.0
    assert summary["perfect"]["num_switches"] == 0
    assert summary["switching"]["num_switches"] == 1
    assert summary["switching"]["idf1"] == pytest.approx(0.5)


def test_trajectories_split_at_gaps(tmp_path):
    rows = []
    for frame in [0, 1, 2, 5, 6, 7, 8]:
        rows.append(f"{frame} 3 Car 0 0 0 {frame} 0 {frame + 10} 10 1 1 1 0 0 0 0")
    rows.append("0 -1 DontCare -1 -1 -10 0 0 5 5 -1000 -1000 -1000 -10 -1 -1 -1")
    path = tmp_path / "labels.txt"
    path.write_text("\n".join(rows) + "\n")
    runs = car_trajectories(load_labels(path), min_length=2)
    assert [len(run) for run in runs] == [3, 4]
