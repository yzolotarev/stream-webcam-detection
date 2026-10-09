"""Score live_cams against a labels file.

A moment is correct when every labelled webcam lies inside an output panel and
no labelled non-webcam face (static art, people in the watched video) does.

Frames are not distributed with this repository. To re-run, supply your own
corpus laid out as <corpus>/<source>/<moment>/f0.jpg..f4.jpg, with optional
whole-stream samples in <corpus>/<source>/stream/s*.jpg, and a labels file in
the format of labels-dev.json:

    python live/evaluate.py <corpus> live/labels-dev.json
    python live/evaluate.py --results live/results-dev.json live/labels-dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def inside(panel, point) -> bool:
    return panel[0] <= point[0] <= panel[0] + panel[2] and panel[1] <= point[1] <= panel[1] + panel[3]


def score(label: dict, panels: list) -> str:
    covered = [any(inside(p, c) for p in panels) for c in label["cams"]]
    leaked = any(inside(p, n) for p in panels for n in label["not_cams"])
    if leaked:
        return "WRONG"
    if all(covered):
        return "ok"
    return "partial" if any(covered) else "fallback"


def detect(corpus: Path, key: str) -> dict:
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import live_cams
    folder = corpus / key
    images = [cv2.imread(str(folder / f"f{i}.jpg")) for i in range(5)]
    stream = [cv2.imread(str(p)) for p in sorted((folder.parent / "stream").glob("s*.jpg"))] or None
    return live_cams.detect(images, stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="?", type=Path, help="folder with frames (omit with --results)")
    parser.add_argument("labels", type=Path)
    parser.add_argument("--results", type=Path, help="re-score saved detector output instead of running it")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text())["moments"]
    saved = json.loads(args.results.read_text()) if args.results else None
    if saved is None and args.corpus is None:
        parser.error("give a corpus folder or --results")
    tally: dict[str, int] = {}
    for key, label in sorted(labels.items()):
        result = saved[key] if saved is not None else detect(args.corpus, key)
        verdict = score(label, result["panels"])
        tally[verdict] = tally.get(verdict, 0) + 1
        print(f"{key:12} {verdict:8} {result['mode']}")
    print(tally)
    return 0


if __name__ == "__main__":
    sys.exit(main())
