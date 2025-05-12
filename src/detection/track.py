"""
Track every player in a clip using YOLOv8 + ByteTrack.

Usage:
    python -m src.detection.track data/clips/play01.mp4 \
           --model yolov8n.pt --imgsz 960 --conf 0.25
Outputs:
    - output/runs/track/track/play01.mp4      (annotated video)
    - output/runs/track/track/labels/*.txt    (per-frame detections)
"""
import argparse, pathlib
from ultralytics import YOLO

def run(video_path: str,
        model_path="yolov8n.pt",
        imgsz: int = 960,
        conf: float = 0.25):
    video_path = pathlib.Path(video_path)
    model = YOLO(model_path)

    model.track(
        source=str(video_path),
        imgsz=imgsz,
        conf=conf,
        tracker="bytetrack.yaml",
        save=True,                 # save annotated MP4
        save_txt=True,             # save per-frame *.txt labels
        project="output/runs",
        name="track"
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video_path")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf",  type=float, default=0.25)
    run(**vars(ap.parse_args()))