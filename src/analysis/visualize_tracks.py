"""
visualize_tracks.py
-------------------
Run tracking and create a prettier overlay: coloured boxes, ID text,
and short motion trails.

Example:
    python -m src.analysis.visualize_tracks data/clips/play01.mp4 \
           --out output/runs/track/track_fancy.mp4 \
           --model yolov8m.pt --imgsz 1280 --trail_len 15
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
    model_path: str,
    imgsz: int,
    trail_len: int,
) -> None:
    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            history[tid].append((cx, cy))
            history[tid] = history[tid][-trail_len:]
            colour = tuple(int(c) for c in np.random.default_rng(tid).integers(40, 255, 3))
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if len(history[tid]) > 1:
                cv2.polylines(frame, [np.array(history[tid], dtype=np.int32)], False, colour, 2)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved overlay ➜ {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="output/runs/track/track_fancy.mp4")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--trail_len", type=int, default=10)
    args = ap.parse_args()
    fancy_track(args.video, args.out, args.model, args.imgsz, args.trail_len)