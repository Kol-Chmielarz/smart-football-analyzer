"""
visualize_tracks.py
-------------------
Read a clip, run YOLO+ByteTrack, and save an annotated video with
colour-coded boxes, ID text, and short motion trails.

Usage:
    python -m src.analysis.visualize_tracks data/clips/play01.mp4 \
           --out output/runs/track/track_fancy.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def fancy_track(
    video_path: str | Path,
    out_path: str | Path,
    model_path: str = "yolov8n.pt",
    imgsz: int = 960,
    trail_len: int = 10,
) -> None:
    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    history: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for res in model.track(
        source=str(video_path),
        stream=True,
        classes=[0],
        imgsz=imgsz,
        tracker="bytetrack.yaml",
    ):
        frame = res.orig_img.copy()

        # zip boxes to track IDs
        for box, tid in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.id.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            history[tid].append((cx, cy))
            history[tid] = history[tid][-trail_len:]

            # deterministic colour per ID
            colour = tuple(int(c) for c in np.random.default_rng(tid).integers(40, 255, 3))

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame,
                f"ID {tid}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )

            pts = np.array(history[tid], dtype=np.int32)
            cv2.polylines(frame, [pts], False, colour, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved overlay video ➜ {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Input clip (e.g. data/clips/play01.mp4)")
    ap.add_argument(
        "--out",
        default="output/runs/track/track_fancy.mp4",
        help="Path for annotated video",
    )
    args = ap.parse_args()
    fancy_track(args.video, args.out)