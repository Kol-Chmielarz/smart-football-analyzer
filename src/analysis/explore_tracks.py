"""
Explore the tracking labels produced by src.detection.track.

Examples
--------
Auto‑detect the newest run:
    python -m src.analysis.explore_tracks

Specify a label directory explicitly:
    python -m src.analysis.explore_tracks output/runs/track8/labels
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.utils.track_parser import load_tracks


def _latest_labels(root: str = "output/runs") -> str | None:
    """Return the most recently modified 'track*/labels' folder, or None."""
    paths = sorted(Path(root).glob("track*/labels"), key=lambda p: p.stat().st_mtime)
    return str(paths[-1]) if paths else None


def main(label_dir: str | None = None) -> None:
    # Auto‑pick newest run if none supplied
    if label_dir is None:
        label_dir = _latest_labels()
        if label_dir is None:
            sys.exit("❌ No label folders found. Run the tracker first.")
        print(f"Using latest label dir: {label_dir}")

    df = load_tracks(label_dir)
    print(df.head())
    print("Unique player IDs:", df['id'].nunique())
    print("Total frames:", df['frame'].max() + 1)


if __name__ == "__main__":
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_arg)