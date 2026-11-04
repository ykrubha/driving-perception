"""Download the data and detector weights at pinned versions. Nothing here is redistributed.

    python scripts/download_data.py

Gets KITTI tracking labels and PointRCNN detections from the AB3DMOT repository, a highway
dash-cam video from Udacity's vehicle detection project, and YOLO11n weights.
"""

from __future__ import annotations

import hashlib
import subprocess
import urllib.request
from pathlib import Path

DATA = Path("data")
AB3DMOT_REPO = "https://github.com/xinshuoweng/AB3DMOT.git"
AB3DMOT_COMMIT = "61f3bd72574093e367916c757b4747ca445f978c"
FILES = {
    "project_video.mp4": (
        "https://raw.githubusercontent.com/udacity/CarND-Vehicle-Detection/"
        "66d678b186f4e70bbb283b52e1ae36d4187f2ff9/project_video.mp4",
        "651949922042c223375d40bf8929def97dbb62c6ade651d1f5b08eea47127d5f",
    ),
    "yolo11n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    ),
}


def fetch_kitti() -> None:
    target = DATA / "ab3dmot"
    if (target / "scripts/KITTI/label/0000.txt").exists():
        print("KITTI data already present.")
        return
    target.mkdir(parents=True, exist_ok=True)
    for command in (
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", AB3DMOT_REPO],
        ["git", "config", "remote.origin.promisor", "true"],
        ["git", "config", "remote.origin.partialclonefilter", "blob:none"],
        ["git", "sparse-checkout", "set", "--no-cone", "/scripts/KITTI/label/*",
         "/data/KITTI/detection/pointrcnn_Car_val/*", "/LICENSE"],
        ["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", AB3DMOT_COMMIT],
        ["git", "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"],
    ):
        subprocess.run(command, cwd=target, check=True)
    print("Fetched KITTI labels and detections.")


def fetch_file(name: str, url: str, sha256: str) -> None:
    path = DATA / name
    if not path.exists():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        path.unlink()
        raise SystemExit(f"Checksum mismatch for {name}: got {digest}")
    print(f"{name} verified.")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    fetch_kitti()
    for name, (url, sha256) in FILES.items():
        fetch_file(name, url, sha256)


if __name__ == "__main__":
    main()
