"""Exploratory face recurrence, NOT a webcam-panel detector.
Run from project root: .venv/bin/python experiment/probe.py
"""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'experiment'
CLIPS = json.loads((OUT / 'clips.json').read_text())['clips']
COLORS = [(40, 230, 40), (0, 170, 255), (230, 100, 230), (255, 210, 0)]


def cluster_frames(frames, detector):
    # Global spatial recurrence, not identity tracking or scene segmentation.
    clusters = []
    for frame in frames:
        used = set()
        for box in sorted(frame[detector], key=lambda b: b[2] * b[3], reverse=True):
            a = np.array(box[:4])
            center = a[:2] + a[2:] / 2
            matches = []
            for i, cluster in enumerate(clusters):
                if i in used:
                    continue
                median = np.median(cluster['boxes'], axis=0)
                distance = np.linalg.norm(center - median[:2] - median[2:] / 2)
                ratio = a[2:] / np.maximum(median[2:], 1e-8)
                if distance < 0.075 and np.all((ratio > 0.4) & (ratio < 2.5)):
                    matches.append((distance, i))
            if matches:
                index = min(matches)[1]
            else:
                index = len(clusters)
                clusters.append({'boxes': [], 'frame_indices': []})
            used.add(index)
            clusters[index]['boxes'].append(a.tolist())
            clusters[index]['frame_indices'].append(frame['index'])
    clusters.sort(key=lambda c: len(c['boxes']), reverse=True)
    for i, c in enumerate(clusters):
        c['id'] = i + 1
        c['count'] = len(c['boxes'])
        c['coverage'] = c['count'] / len(frames)
        c['median_face_bbox'] = np.median(c.pop('boxes'), axis=0).tolist()
    return clusters


def sheet(tiles, path, cols=3):
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * 390, cols * 640, 3), np.uint8)
    for i, tile in enumerate(tiles):
        y, x = (i // cols) * 390, (i % cols) * 640
        canvas[y:y + 390, x:x + 640] = tile
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f'Cannot write {path}')


def annotate(image, frame):
    h, w = image.shape[:2]
    rendered = image.copy()
    for detector, color in [('haar', (0, 180, 255)), ('yunet', (60, 230, 60))]:
        for b in frame[detector]:
            x, y, bw, bh = np.rint(np.array(b[:4]) * [w, h, w, h]).astype(int)
            cv2.rectangle(rendered, (x, y), (x + bw, y + bh), color, 2)
    scale = min(640 / w, 360 / h)
    resized = cv2.resize(rendered, (round(w * scale), round(h * scale)))
    tile = np.zeros((390, 640, 3), np.uint8)
    tile[:resized.shape[0], :resized.shape[1]] = resized
    label = f"#{frame['index']:02d} {frame['actual_s']:.1f}s  Y:{len(frame['yunet'])} H:{len(frame['haar'])}"
    cv2.putText(tile, label, (8, 381), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return tile


def main():
    cv2.setNumThreads(2)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if haar.empty():
        raise RuntimeError('Haar cascade unavailable')
    yunet = cv2.FaceDetectorYN.create(str(OUT / 'yunet.onnx'), '', (320, 320), 0.8, 0.3, 5000)
    report = {'opencv': cv2.__version__, 'model_sha256': hashlib.sha256((OUT / 'yunet.onnx').read_bytes()).hexdigest(),
              'method': '48 midpoint samples; Haar scaleFactor=1.1 minNeighbors=5 minSize=20; YuNet score>=0.8; spatial clustering center distance<0.075 and per-axis size ratio 0.4..2.5; no panel detection', 'videos': []}
    for clip in CLIPS:
        video_id = clip['id']
        started = time.monotonic()
        cap = cv2.VideoCapture(str(ROOT / clip['file']))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {video_id}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        frames, tiles = [], []
        decode_s = haar_s = yunet_s = 0.0
        for index, timestamp in enumerate((np.arange(48) + 0.5) * duration / 48):
            t = time.monotonic()
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
            ok, image = cap.read()
            decode_s += time.monotonic() - t
            if not ok:
                raise RuntimeError(f'Decode failed: {video_id} @ {timestamp}')
            h, w = image.shape[:2]
            frame = {'index': index, 'requested_s': float(timestamp), 'actual_s': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000}
            t = time.monotonic()
            boxes = haar.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            frame['haar'] = [(np.array(b, dtype=float) / [w, h, w, h]).tolist() for b in boxes]
            haar_s += time.monotonic() - t
            t = time.monotonic()
            yunet.setInputSize((w, h))
            _, faces = yunet.detect(image)
            frame['yunet'] = [] if faces is None else [(f[:4] / [w, h, w, h]).tolist() + [float(f[-1])] for f in faces]
            yunet_s += time.monotonic() - t
            frames.append(frame)
            tiles.append(annotate(image, frame))
            if index % 4 == 0:
                cv2.imwrite(str(OUT / f'{video_id}-{index:02d}.jpg'), image)
        cap.release()
        for page in range(4):
            sheet(tiles[page * 12:(page + 1) * 12], OUT / f'{video_id}-page{page + 1}.jpg')
        sheet(tiles[::4], OUT / f'{video_id}-overview.jpg')
        entry = {'id': video_id, 'file': clip['file'], 'duration_s': duration, 'width': w, 'height': h,
                 'frames': frames, 'clusters': {d: cluster_frames(frames, d) for d in ['haar', 'yunet']},
                 'timing_s': {'decode': decode_s, 'haar': haar_s, 'yunet': yunet_s, 'total': time.monotonic() - started}}
        report['videos'].append(entry)
        (OUT / 'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: entry[k] for k in ['id', 'file', 'timing_s']}, ensure_ascii=False), flush=True)
        for d in ['haar', 'yunet']:
            print(d, [(c['count'], np.round(c['median_face_bbox'], 3).tolist()) for c in entry['clusters'][d][:5]], flush=True)


if __name__ == '__main__':
    main()
