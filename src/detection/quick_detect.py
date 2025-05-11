"""
quick_detect.py
----------------
Run YOLOv8 on a video and show the detections live.

Usage:
    python -m src.detection.quick_detect data/clips/play01.mp4
Press Esc to quit the preview window.
"""
import cv2, sys, pathlib
from ultralytics import YOLO

def main(video_path: str, model_path="yolov8n.pt", imgsz=960, conf=0.3):
    model = YOLO(model_path)                       # tiny, fast model
    cap   = cv2.VideoCapture(str(video_path))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model(frame, imgsz=imgsz, conf=conf)[0]
        cv2.imshow("players", res.plot())          # draw boxes
        if cv2.waitKey(1) & 0xFF == 27:            # Esc exits
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.detection.quick_detect <video>")
        sys.exit(1)
    main(sys.argv[1])