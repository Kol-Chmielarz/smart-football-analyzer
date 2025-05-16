"""
track_parser.py
----------------
Utility to load YOLO-ByteTrack label files into a pandas DataFrame.

Each label TXT line may have 6 columns (id x y w h conf) or 7 columns
(id x y w h conf cls). Coordinates are normalized (0–1).

Usage
-----
from src.utils.track_parser import load_tracks
df = load_tracks("output/runs/trackX/labels")
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_tracks(label_dir: str | Path = "output/runs/track/track/labels") -> pd.DataFrame:
    """
    Load all *.txt label files in `label_dir` into a tidy DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: frame, id, x, y, w, h, conf, cls
    """
    label_dir = Path(label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    rows: list[tuple] = []
    for txt_path in sorted(label_dir.glob("*.txt")):
        frame_num = int(txt_path.stem.split("_")[-1])
        with txt_path.open() as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) == 7:
                    tid, x, y, w, h, conf, cls = map(float, parts)
                elif len(parts) == 6:
                    tid, x, y, w, h, conf = map(float, parts)
                    cls = -1.0
                else:
                    raise ValueError(f"Unexpected label format in {txt_path}: {parts}")
                rows.append((frame_num, int(tid), x, y, w, h, conf, int(cls)))

    df = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h", "conf", "cls"])
    return df.sort_values(["id", "frame"]).reset_index(drop=True)