"""
track.py
---------
Track every player in a football clip using YOLOv8 + ByteTrack.

Usage:
    python -m src.detection.track data/clips/play01.mp4 \
           --model yolov8n.pt --imgsz 960 --conf 0.25 --classes 0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def run(
    video_path: str | Path,
    model: str = "yolov8n.pt",
    imgsz: int = 960,
    conf: float = 0.25,
    classes: list[int] | None = None,
) -> None:
    video_path = Path(video_path)
    model = YOLO(model)
    model.track(
        source=str(video_path),
        imgsz=imgsz,
        conf=conf,
        classes=classes,
        tracker="bytetrack.yaml",
        save=True,
        save_txt=True,
        save_conf=True,
        project="output/runs",
        name="track",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video_path")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", type=int, nargs="*", default=[0])
    run(**vars(ap.parse_args()))