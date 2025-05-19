"""
track_parser.py
---------------
Load YOLO-ByteTrack TXT labels into a DataFrame.

Label formats supported
-----------------------
7-col TRACK  : cls x y w h conf track_id
6-col TRACK  : track_id cls x y w h          (no conf)
6-col DETECT : cls x y w h conf             (no track IDs)

All coordinates are normalised (0–1).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_tracks(label_dir: str | Path) -> pd.DataFrame:
    label_dir = Path(label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(label_dir)

    rows: list[tuple] = []
    for txt in sorted(label_dir.glob("*.txt")):
        frame = int(txt.stem.split("_")[-1])
        with txt.open() as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue

                if len(parts) == 7:          # cls x y w h conf tid
                    cls, x, y, w, h, conf, tid = map(float, parts)
                elif len(parts) == 6:
                    if float(parts[0]).is_integer():  # assume TRACK: tid cls ...
                        tid, cls, x, y, w, h = map(float, parts)
                        conf = 1.0
                    else:                              # DETECT: cls x y w h conf
                        cls, x, y, w, h, conf = map(float, parts)
                        tid = -1.0
                else:
                    continue

                rows.append((frame, int(tid), x, y, w, h, conf, int(cls)))

    cols = ["frame", "id", "x", "y", "w", "h", "conf", "cls"]
    return pd.DataFrame(rows, columns=cols).sort_values(["frame", "id"]).reset_index(drop=True)