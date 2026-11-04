"""Run detection and tracking on a driving video, and draw the result.

    python scripts/run_video.py --video data/project_video.mp4 --start 18 --seconds 20 --gif assets/tracking.gif

The tracker is the same code scored on KITTI. This video has no ground truth, so the output
shows behavior and can't give an accuracy score.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from perception.factory import make_tracker

VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck
# Hand-set for YOLO's 0-1 confidence scale. The KITTI settings use a different detector's scale.
VIDEO_SETTINGS = {"high_score": 0.4, "low_score": 0.1, "match_iou": 0.2, "max_age": 5, "min_hits": 3}


def color_for(track_id: int) -> tuple[int, int, int]:
    hue = int((track_id * 47) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw(frame: np.ndarray, outputs: list, trails: dict, caption: str) -> np.ndarray:
    image = frame.copy()
    for output in outputs:
        x1, y1, x2, y2 = output.box.astype(int)
        color = color_for(output.id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"ID {output.id}"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(y1 - height - 8, 0)
        cv2.rectangle(image, (x1, top), (x1 + width + 8, top + height + 8), color, -1)
        cv2.putText(image, label, (x1 + 4, top + height + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        points = list(trails[output.id])
        for start, end in zip(points, points[1:]):
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(image, caption, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/project_video.mp4")
    parser.add_argument("--weights", default="data/yolo11n.pt")
    parser.add_argument("--method", default="kalman_two_stage")
    parser.add_argument("--settings", default="results/tracker_settings.json")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="outputs/tracking.mp4")
    parser.add_argument("--gif", default=None)
    parser.add_argument("--gif-width", type=int, default=640)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--stats", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    limit = None if args.seconds is None else int(args.seconds * fps)

    kalman_params = None
    if Path(args.settings).exists():
        kalman_params = json.loads(Path(args.settings).read_text()).get("kalman_params")
    detector = YOLO(args.weights)
    tracker = make_tracker(args.method, VIDEO_SETTINGS, float(height), float(width), kalman_params)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_fps = fps / args.stride
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    trails: dict = defaultdict(lambda: deque(maxlen=20))
    lengths: dict = defaultdict(int)
    gif_frames = []
    gif_every = max(1, round(out_fps / args.gif_fps))
    index = processed = detections_seen = 0
    detect_seconds = track_seconds = 0.0

    while True:
        ok, frame = capture.read()
        if not ok or (limit is not None and index >= limit):
            break
        if index % args.stride:
            index += 1
            continue
        tic = time.perf_counter()
        result = detector.predict(frame, imgsz=args.imgsz, conf=VIDEO_SETTINGS["low_score"],
                                  classes=VEHICLE_CLASSES, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros(0)
        toc = time.perf_counter()
        outputs = tracker.update(boxes, scores)
        track_seconds += time.perf_counter() - toc
        detect_seconds += toc - tic
        detections_seen += int((scores >= VIDEO_SETTINGS["high_score"]).sum())

        for output in outputs:
            center_x = (output.box[0] + output.box[2]) / 2
            trails[output.id].append((int(center_x), int(output.box[3])))
            lengths[output.id] += 1

        caption = f"YOLO11n + {args.method}   tracks: {len(outputs)}   frame {index}"
        drawn = draw(frame, outputs, trails, caption)
        writer.write(drawn)
        if args.gif and processed % gif_every == 0:
            scale = args.gif_width / width
            small = cv2.resize(drawn, (args.gif_width, int(height * scale)), interpolation=cv2.INTER_AREA)
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        processed += 1
        index += 1

    capture.release()
    writer.release()

    if args.gif:
        import imageio.v2 as imageio

        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.gif, gif_frames, duration=1000 / args.gif_fps, loop=0)

    stats = {
        "video": Path(args.video).name, "method": args.method, "start_s": args.start,
        "frames_processed": processed, "processed_fps": round(out_fps, 2),
        "tracks_reported": len(lengths),
        "median_track_frames": float(np.median(list(lengths.values()))) if lengths else 0,
        "longest_track_frames": max(lengths.values(), default=0),
        "high_confidence_detections": detections_seen,
        "detector_ms_per_frame": round(1000 * detect_seconds / max(processed, 1), 1),
        "tracker_ms_per_frame": round(1000 * track_seconds / max(processed, 1), 2),
        "settings": VIDEO_SETTINGS,
    }
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
