"""Find every live webcam in a few sampled frames; never a static picture.

Rules, in order (all on ~5 sampled frames of a clip):

1. Faces are detected at FACE_WIDTH (not the 640 px work width), so a small
   in-call webcam is still seen.
2. A face counts only if it stays in one place in >= MIN_FRAMES samples.
3. Liveness: the face pixels must change between samples taken tens of seconds
   apart. Overlay art, avatars and thumbnails are pixel-identical -> rejected.
4. The webcam window is the moving region around the face, bounded by the
   static stream design (change mask). Too large/merged -> face-framed box.
5. Webcam vs a person inside the watched video: the primary webcam sits in the
   same place across the whole stream (stream samples). Other live faces are
   added only when they share the primary webcam's moving window (a call app
   showing both), or when there is no primary webcam and two or more live faces
   of similar size fill separate windows (a conference grid). Anything else is a
   person in the watched video and is never cropped.

Output mode: ``webcam`` (one), ``multi`` (several), ``full_frame`` (none, or a
camera that already fills the screen).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).with_name("yunet.onnx")
FACE_WIDTH = 1280
FACE_SCORE = 0.75
MIN_FRAMES = 3
LIVE_DIFF = 8.0            # mean abs grey-level change of the face crop between samples
MASK_SIZE = (480, 270)
MASK_DIFF = 18             # per-pixel max-min across samples that counts as "moving"
WINDOW_MAX_AREA = 0.42     # a moving region bigger than this is content, not a webcam window
FULL_SCREEN_AREA = 0.60    # one live window this large -> show the whole frame
MIN_WINDOW_FACES = 3.0     # a real webcam window is at least this many face-widths wide
CALL_WINDOW_MAX_AREA = 0.40  # one call-app window holding several webcams is at most this big
WINDOW_ASPECT = (0.9, 2.1)   # plausible webcam window width/height in pixels
STREAM_MASK_DIFF = 4        # p90-p10 grey spread over hours: any live camera pixel exceeds it
INSET = 0.015               # shave this fraction of each panel side (border slivers)
SELF_VIEW_MAX_SHARE = 0.4    # a call face under this share of the biggest face width is a self-view
OVERLAY_MAX_AREA = 0.15      # the streamer's own webcam window is at most this share of the frame
PIP_SCALE = 2.0            # a face this many times bigger in the same region makes yours an inset
NESTED_MARGIN = 0.012       # trim margin around a webcam nested in another (fraction of frame)
SECONDARY_MIN_SCALE = 0.6  # extra recurring webcams must be at least this fraction of the largest
RECURRENCE_PRIMARY = 0.60  # fraction of stream samples with a face at the same place -> overlay webcam


def _detector(width: int, height: int):
    detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (width, height), FACE_SCORE, 0.3, 5000)
    return detector


def detect_faces(images: list[np.ndarray]) -> list[list[list[float]]]:
    """Normalised [x, y, w, h, score] faces per image at FACE_WIDTH."""
    out = []
    detector = None
    for image in images:
        if image is None:
            out.append([])
            continue
        height = round(image.shape[0] * FACE_WIDTH / image.shape[1])
        scaled = cv2.resize(image, (FACE_WIDTH, height))
        if detector is None:
            detector = _detector(FACE_WIDTH, height)
        _, faces = detector.detect(scaled)
        rows = []
        for face in faces if faces is not None else []:
            x, y, w, h = face[:4]
            if w < 8 or h < 8:
                continue
            rows.append([x / FACE_WIDTH, y / height, w / FACE_WIDTH, h / height, float(face[-1])])
        out.append(rows)
    return out


def _centre(box) -> np.ndarray:
    return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])


def _same_place(a, b) -> bool:
    ratio = b[2] / max(a[2], 1e-6)
    return 0.6 < ratio < 1.65 and float(np.linalg.norm(_centre(a) - _centre(b))) < 0.6 * max(a[2], b[2])


def cluster_faces(frames: list[list[list[float]]]) -> list[dict]:
    """Faces that recur in the same place across samples."""
    clusters: list[dict] = []
    for index, faces in enumerate(frames):
        for face in sorted(faces, key=lambda f: -f[2] * f[3]):
            match = next((c for c in clusters if index not in c["frames"]
                          and _same_place(c["box"], face)), None)
            if match is None:
                match = {"frames": {}, "box": face[:4]}
                clusters.append(match)
            match["frames"][index] = face[:4]
            match["box"] = np.median(np.array(list(match["frames"].values())), axis=0).tolist()
    return [c for c in clusters if len(c["frames"]) >= MIN_FRAMES]


def face_change(images: list[np.ndarray], box) -> float:
    """Median grey-level change of the face crop between consecutive samples."""
    crops = []
    for image in images:
        if image is None:
            continue
        height, width = image.shape[:2]
        x, y, w, h = box
        x0, y0 = max(0, int((x - 0.1 * w) * width)), max(0, int((y - 0.1 * h) * height))
        x1, y1 = min(width, int((x + 1.1 * w) * width)), min(height, int((y + 1.1 * h) * height))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        grey = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        crops.append(cv2.resize(grey, (48, 48)).astype(np.float32))
    diffs = [float(np.mean(np.abs(a - b))) for a, b in zip(crops, crops[1:])]
    return float(np.median(diffs)) if diffs else 0.0


def change_mask(images: list[np.ndarray]) -> np.ndarray:
    greys = [cv2.GaussianBlur(cv2.cvtColor(cv2.resize(image, MASK_SIZE), cv2.COLOR_BGR2GRAY), (5, 5), 0)
             for image in images if image is not None]
    stack = np.stack(greys).astype(np.int16)
    moving = (stack.max(axis=0) - stack.min(axis=0)) >= MASK_DIFF
    moving = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return moving


def stream_mask(images: list[np.ndarray]) -> np.ndarray:
    """Pixels that vary over hours of stream: every pixel of a live camera window
    does (light, sensor noise, movement); static overlay design only flickers on its
    edges, which the opening removes. Gives solid, sharp-edged window rectangles."""
    greys = np.stack([cv2.cvtColor(cv2.resize(image, MASK_SIZE), cv2.COLOR_BGR2GRAY).astype(np.float32)
                      for image in images if image is not None])
    spread = np.percentile(greys, 90, axis=0) - np.percentile(greys, 10, axis=0)
    mask = (spread >= STREAM_MASK_DIFF).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def moving_window(mask: np.ndarray, box) -> tuple[list[float] | None, float]:
    """Bounding box (normalised) and area of the moving region holding the face."""
    _count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    width, height = MASK_SIZE
    # The face itself may hold still; look for moving pixels in and around it.
    x0, y0 = int(max(0, (box[0] - box[2]) * width)), int(max(0, box[1] * height))
    x1, y1 = int(min(width, (box[0] + 2 * box[2]) * width)), int(min(height, (box[1] + 2.5 * box[3]) * height))
    patch = labels[y0:y1, x0:x1]
    ids, sizes = np.unique(patch[patch > 0], return_counts=True)
    if not len(ids):
        return None, 0.0
    label = int(ids[np.argmax(sizes)])
    x, y, w, h, _area = stats[label]
    window = [x / width, y / height, w / width, h / height]
    return window, window[2] * window[3]


def _intersect(a, b) -> list[float] | None:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return [x1, y1, x2 - x1, y2 - y1] if x2 > x1 and y2 > y1 else None


def face_framed_box(box) -> list[float]:
    """Head-and-shoulders 16:9 box around a face, clamped to the frame."""
    # In normalised coordinates of a 16:9 frame a 16:9 box has h == w.
    w = min(1.0, box[2] * 5.0)
    h = w
    cx = box[0] + box[2] / 2
    top = box[1] - box[3] * 0.9
    x = min(max(0.0, cx - w / 2), 1.0 - w)
    y = min(max(0.0, top), 1.0 - h)
    return [x, y, w, h]


def recurrence(box, stream_faces: list[list[list[float]]] | None, window=None) -> float | None:
    """Share of stream samples with a face at this place, or inside this webcam window.

    A streamer leans and shifts inside a fixed webcam window, so a small window
    counts any face inside it; a large (merged/content) window counts only the
    same face position.
    """
    if not stream_faces:
        return None
    small = window is not None and window[2] * window[3] <= CALL_WINDOW_MAX_AREA

    def hit(faces) -> bool:
        return any(_same_place(box, face) or (small and _contains(window, _centre(face))
                                              and 0.6 < face[2] / max(box[2], 1e-6) < 1.65)
                   for face in faces)

    return sum(hit(faces) for faces in stream_faces) / len(stream_faces)


def _contains(window, point) -> bool:
    return window[0] <= point[0] <= window[0] + window[2] and window[1] <= point[1] <= window[1] + window[3]


def _exclude_nested(cams: list[dict]) -> None:
    """Trim a big webcam panel so a smaller webcam inside it (picture-in-picture in
    a call app) is not shown twice. Cut the side that loses the least area."""
    for big in cams:
        for small in cams:
            if small is big or small["panel"][2] * small["panel"][3] >= big["panel"][2] * big["panel"][3]:
                continue
            if not _contains(big["panel"], _centre(small["panel"])):
                continue
            x, y, w, h = big["panel"]
            pad = NESTED_MARGIN  # the inner window's frame/border reaches a bit past its panel
            sx, sy, sw, sh = small["panel"][0] - pad, small["panel"][1] - pad, \
                small["panel"][2] + 2 * pad, small["panel"][3] + 2 * pad
            options = [
                [x, y, w, sy - y],                  # keep above
                [x, sy + sh, w, y + h - sy - sh],   # keep below
                [x, y, sx - x, h],                  # keep left
                [sx + sw, y, x + w - sx - sw, h],   # keep right
            ]
            fc = _centre(big["face"])
            keep = [o for o in options if o[2] > 0 and o[3] > 0 and _contains(o, fc)]
            if keep:
                big["panel"] = max(keep, key=lambda o: o[2] * o[3])


def _split_shared_windows(live: list[dict], mask: np.ndarray, stream_faces) -> None:
    """Two call tiles side by side often merge into one moving region. Two live
    faces of similar size in the same region are split at the stillest column
    between them (the tile gap), so each participant gets its own window
    (2026-10-09: a partner who looked away failed the recurrence bar)."""
    width, height = MASK_SIZE
    for a in live:
        for b in live:
            if a is b or a["window"] is None or b["window"] is None or a["face"][0] >= b["face"][0]:
                continue
            if any(abs(u - v) > 1e-3 for u, v in zip(a["window"], b["window"])):
                continue
            if not 0.5 < b["face"][2] / max(a["face"][2], 1e-6) < 2.0:
                continue
            x, y, w, h = a["window"]
            lo, hi = a["face"][0] + a["face"][2], b["face"][0]
            c0, c1 = int(lo * width), int(hi * width)
            r0, r1 = int(y * height), max(int(y * height) + 1, int((y + h) * height))
            if c1 - c0 < 2:
                continue
            density = mask[r0:r1, c0:c1].mean(axis=0)
            stillest = np.flatnonzero(density <= density.min() + 1e-6)
            split = (c0 + int(np.median(stillest))) / width
            a["window"] = [round(v, 4) for v in (x, y, split - x, h)]
            b["window"] = [round(v, 4) for v in (split, y, x + w - split, h)]
            for c in (a, b):
                c["window_area"] = round(c["window"][2] * c["window"][3], 4)
                c["recurrence"] = recurrence(c["face"], stream_faces, c["window"])
                c["split"] = True


def detect(images: list[np.ndarray], stream_images: list[np.ndarray] | None = None) -> dict:
    faces = detect_faces(images)
    clusters = cluster_faces(faces)
    mask = change_mask(images)
    # Over hours of stream everything inside a fixed webcam window changes while
    # the overlay design stays put, so the stream-wide mask shows the whole window;
    # a few samples of one moment often show only the moving head.
    wide_mask = stream_mask(stream_images) if stream_images and len(stream_images) >= 6 else None
    stream_faces = detect_faces(stream_images) if stream_images else None
    candidates = []
    for cluster in clusters:
        box = cluster["box"]
        window, area = moving_window(mask, box)
        if wide_mask is not None:
            wide, wide_area = moving_window(wide_mask, box)
            if wide is not None and wide_area <= WINDOW_MAX_AREA:
                window, area = wide, wide_area
        candidates.append({
            "face": [round(v, 4) for v in box], "frames": len(cluster["frames"]),
            "change": round(face_change(images, box), 2),
            "window": window and [round(v, 4) for v in window], "window_area": round(area, 4),
            "recurrence": recurrence(box, stream_faces, window), "verdict": "content"})
    live = [c for c in candidates if c["change"] >= LIVE_DIFF]
    _split_shared_windows(live, wide_mask if wide_mask is not None else mask, stream_faces)
    # Picture-in-picture: over hours a call window and the small inset camera in
    # its corner merge into one moving region. The inset (a live face at most half
    # the size of another live face in the same region) takes its own window from
    # this moment's samples; the big face keeps the region, minus the inset later.
    for c in live:
        if c["window"] is None:
            continue
        bigger = [o for o in live if o is not c and _contains(c["window"], _centre(o["face"]))
                  and o["face"][2] >= PIP_SCALE * c["face"][2]]
        if bigger:
            inset, inset_area = moving_window(mask, c["face"])
            if inset is None or inset_area >= 0.5 * c["window_area"]:
                inset = face_framed_box(c["face"])
                inset_area = inset[2] * inset[3]
            c["window"], c["window_area"] = [round(v, 4) for v in inset], round(inset_area, 4)
            c["pip"] = True
    for c in candidates:
        if c["change"] < LIVE_DIFF:
            c["verdict"] = "static"
    for c in live:
        c["shared"] = [o for o in live if o is not c and c["window"] is not None
                       and c["window_area"] <= CALL_WINDOW_MAX_AREA
                       and _contains(c["window"], _centre(o["face"]))]
    primary = [c for c in live if c["recurrence"] is not None and c["recurrence"] >= RECURRENCE_PRIMARY]
    if stream_faces is None:
        primary = [c for c in live if c["window"] is not None and c["window_area"] <= WINDOW_MAX_AREA]
    # A webcam inside the watched video (another streamer's overlay) can also sit
    # still for hours, but it is shown much smaller than this stream's own camera.
    if primary:
        largest = max(c["face"][2] for c in primary)
        primary = [c for c in primary if c["face"][2] >= SECONDARY_MIN_SCALE * largest]
    for c in primary:
        c["verdict"] = "webcam"
    # The streamer's own overlay camera: a recurring live face alone in a small
    # window. When it exists, faces inside other windows are watched content even
    # if that content stays on screen for the whole stream (2026-10-08: a stream
    # spent watching a video about the streamer, with cameras inside it).
    overlay = [c for c in primary if not c.get("pip") and c["window"] is not None
               and c["window_area"] <= OVERLAY_MAX_AREA and not c.get("shared")
               # Two recurring live windows of about the same size are a call
               # grid side by side, not an overlay over watched content (2026-10-09).
               and not any(o is not c and o["window"] is not None and o["window_area"] <= 2 * c["window_area"]
                           and c["window_area"] <= 2 * o["window_area"] for o in primary)]
    if overlay:
        for c in primary:
            if c not in overlay:
                c["verdict"] = "content"
        primary = overlay
    for c in live:
        if c["verdict"] == "webcam":
            continue
        with_primary = any(c in p.get("shared", []) or p in c.get("shared", []) for p in primary)
        sizes = [o["face"][2] for o in live]
        grid = (not primary and len(live) >= 2 and max(sizes) / max(min(sizes), 1e-6) < 2.5)
        if with_primary or grid:
            c["verdict"] = "webcam"
    cams = [c for c in candidates if c["verdict"] == "webcam"]
    for c in cams:
        nested_only = all(o["face"][2] < c["face"][2] for o in c.get("shared", []))
        alone = (c["window"] is not None and c["window_area"] <= WINDOW_MAX_AREA and nested_only
                 and c["window"][2] >= MIN_WINDOW_FACES * c["face"][2])
        if alone:
            # Pixel aspect of the window (normalised w/h times 16/9): a webcam is
            # 4:3..16:9-ish; a long strip means the mask leaked into overlay text.
            aspect = c["window"][2] / max(c["window"][3], 1e-6) * 16 / 9
            alone = WINDOW_ASPECT[0] <= aspect <= WINDOW_ASPECT[1]
        c["panel"] = c["window"] if alone else face_framed_box(c["face"])
        if not alone and c["window"] is not None and c["window_area"] <= WINDOW_MAX_AREA \
                and _contains(c["window"], _centre(c["face"])):
            # A close-up camera (face big in its window) gets a face-framed box;
            # it must not spill over into the neighbouring window (2026-10-09).
            c["panel"] = _intersect(c["panel"], c["window"]) or c["panel"]
        c.pop("shared", None)
    for c in candidates:
        c.pop("shared", None)
    _exclude_nested(cams)
    # A call app shows the local user's own preview as a tiny tile (a duplicate of a
    # big participant). With two or more big participants a tile this much smaller
    # is that self-view, not a separate person (observed 2026-10-09 on a two-person call).
    if len(cams) >= 3:
        # Compare faces, not found windows: a real grid tile can get a small window
        # estimate, but its face is as big as the others' (4-tile call, corpus m0).
        biggest = max(c["face"][2] for c in cams)
        big = [c for c in cams if c["face"][2] >= SELF_VIEW_MAX_SHARE * biggest]
        if len(big) >= 2:
            for c in cams:
                if c not in big:
                    c["verdict"] = "self_view"
            cams = big
    for c in cams:
        # Window borders/frames reach a few pixels into the estimate; shave them.
        x, y, w, h = c["panel"]
        c["panel"] = [x + INSET * w, y + INSET * h, w * (1 - 2 * INSET), h * (1 - 2 * INSET)]
    full_screen = len(cams) == 1 and cams[0]["window_area"] >= FULL_SCREEN_AREA
    mode = "full_frame" if not cams or full_screen else "multi" if len(cams) > 1 else "webcam"
    return {"mode": mode, "panels": [c["panel"] for c in cams], "candidates": candidates}


SAMPLE_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
STREAM_SAMPLES = 12
CHUNKED_HTTP = ["-request_size", "4000000", "-multiple_requests", "1"]


def read_samples(video: Path, fractions=SAMPLE_FRACTIONS) -> list[np.ndarray | None]:
    """Full-resolution frames at the given fractions of the clip."""
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = total / fps if fps > 0 else 0.0
        images = []
        for fraction in fractions:
            cap.set(cv2.CAP_PROP_POS_MSEC, fraction * duration * 1000)
            ok, image = cap.read()
            images.append(image if ok else None)
        return images
    finally:
        cap.release()


def fetch_stream_samples(url: str, folder: Path, count: int = STREAM_SAMPLES) -> list[np.ndarray]:
    """~12 frames spread over the whole source, fetched once and cached in *folder*."""
    import subprocess

    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"s{index:02d}.jpg" for index in range(count)]
    if not all(path.is_file() for path in paths):
        result = subprocess.run(
            ["yt-dlp", "--no-playlist", "--print", "%(duration)s", "--get-url",
             "--extractor-args", "youtube:player_client=web_safari",
             "--format", "bv*[height<=1080]", url],
            text=True, capture_output=True, check=True, timeout=120)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        duration = float(next(line for line in lines if not line.startswith("http")))
        stream = next(line for line in lines if line.startswith("http"))
        for index, path in enumerate(paths):
            if path.is_file():
                continue
            at = duration * (0.04 + 0.92 * index / max(1, count - 1))
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *CHUNKED_HTTP,
                            "-ss", f"{at:.3f}", "-i", stream, "-frames:v", "1", "-q:v", "2", str(path)],
                           check=False, timeout=90)
    return [image for image in (cv2.imread(str(path)) for path in paths if path.is_file()) if image is not None]
