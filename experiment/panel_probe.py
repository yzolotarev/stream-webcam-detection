"""Offline experiment: corner-attached rectangular webcam panels.

Run from project root: .venv/bin/python experiment/panel_probe.py
Inference never reads panel-reference.json. Borders and appearance are checked
on every sampled frame; there is no blind carry-forward or continuous tracking.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'experiment'
WORK_WIDTH = 640
CONFIG = {
    'stable_radius_px': 3,
    'resize_fraction': 0.25,
    'appearance_rmse_max': 0.22,
    'stable_score_min': 0.24,
    'stable_side_min': 0.10,
    'changed_score_min': 0.55,
    'changed_side_min': 0.35,
    'change_gain_min': 1.15,
    'seed_score_min': 0.35,
    'seed_margin_min': 0.06,
    'bank_score_min': 0.40,
}


def orient(image, corner):
    if corner not in ('tl', 'tr', 'bl', 'br'):
        raise ValueError(f'Invalid corner: {corner}')
    if corner[0] == 'b':
        image = np.flip(image, axis=0)
    if corner[1] == 'r':
        image = np.flip(image, axis=1)
    return np.ascontiguousarray(image)


def face_corner(box):
    x, y, w, h = box[:4]
    return ('t' if y + h / 2 < 0.5 else 'b') + ('l' if x + w / 2 < 0.5 else 'r')


def canonical_face(box, corner, width, height):
    x, y, w, h = box[:4]
    return np.array([1-x-w if corner[1] == 'r' else x,
                     1-y-h if corner[0] == 'b' else y, w, h]) * [width, height, width, height]


def panel_box(width, height, corner, image_width, image_height):
    return [0 if corner[1] == 'l' else image_width-width,
            0 if corner[0] == 't' else image_height-height, width, height]


def border_maps(image):
    """Colour discontinuity at x/y, with local texture baseline removed."""
    f = image.astype(np.float32)
    dx = np.zeros(f.shape[:2], np.float32)
    dy = dx.copy()
    dx[:, 2:-2] = np.mean(np.abs(f[:, 3:-1] - f[:, 1:-3]), axis=2)
    dy[2:-2] = np.mean(np.abs(f[3:-1] - f[1:-3]), axis=2)
    # Saturation prevents isolated high-contrast objects dominating a long border.
    dx = np.clip(dx / 24, 0, 1)
    dy = np.clip(dy / 24, 0, 1)
    dx = np.maximum(0, dx - 0.5 * cv2.blur(dx, (15, 1)))
    dy = np.maximum(0, dy - 0.5 * cv2.blur(dy, (1, 15)))
    vertical = np.cumsum(dx, axis=0) / np.arange(1, dx.shape[0]+1)[:, None]
    horizontal = np.cumsum(dy, axis=1) / np.arange(1, dy.shape[1]+1)[None, :]
    # Map[y,x] describes half-open rectangle [0,0,x,y].
    vertical[1:] = vertical[:-1].copy()
    horizontal[:, 1:] = horizontal[:, :-1].copy()
    score = np.sqrt(vertical * horizontal)
    return score.astype(np.float32), vertical.astype(np.float32), horizontal.astype(np.float32)


def allowed_map(shape, face):
    h, w = shape
    x, y, fw, fh = face
    yy, xx = np.indices(shape)
    area = xx * yy / (w*h)
    aspect = xx / np.maximum(yy, 1)
    return ((xx >= max(x+fw+1, 0.06*w)) & (yy >= max(y+fh+1, 0.08*h)) &
            (xx <= 0.55*w) & (yy <= 0.6*h) & (area >= 0.015) & (area <= 0.25) &
            (aspect >= 0.35) & (aspect <= 3.5))


def peaks(scores, mask, count=8, radius=6):
    scores = np.where(mask, scores, -1).copy()
    found = []
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(scores), scores.shape)
        value = float(scores[y, x])
        if value <= 0:
            break
        found.append({'w': int(x), 'h': int(y), 'score': value})
        scores[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1] = -1
    return found


def local_candidates(maps, candidate, mask):
    score, vertical, horizontal = maps
    yy, xx = np.indices(score.shape)
    cw, ch = candidate['w'], candidate['h']
    nearby = mask & (np.abs(xx-cw) <= CONFIG['resize_fraction']*cw) & (np.abs(yy-ch) <= CONFIG['resize_fraction']*ch)
    stable = mask & (np.abs(xx-cw) <= CONFIG['stable_radius_px']) & (np.abs(yy-ch) <= CONFIG['stable_radius_px'])
    out = {}
    for name, region in [('stable', stable), ('changed', nearby)]:
        found = peaks(score, region, count=1)
        if found:
            p = found[0]
            p['vertical'] = float(vertical[p['h'], p['w']])
            p['horizontal'] = float(horizontal[p['h'], p['w']])
            out[name] = p
    return out


def appearance(image, proposal):
    crop = image[:proposal['h'], :proposal['w']]
    return cv2.resize(crop, (16,16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def appearance_distance(feature, bank, exclude_index):
    distances = [float(np.sqrt(np.mean((feature-ref)**2))) for index, ref in bank if index != exclude_index]
    return min(distances) if distances else 1.0


def select_proposal(proposals):
    """An explicit abstention is preferable to keeping a stale panel rectangle."""
    stable = proposals.get('stable')
    changed = proposals.get('changed')
    appearance_ok = lambda p: p['appearance_distance'] <= CONFIG['appearance_rmse_max']
    if (changed and appearance_ok(changed) and changed['score'] >= CONFIG['changed_score_min']
            and min(changed['vertical'], changed['horizontal']) >= CONFIG['changed_side_min']
            and changed['score'] > CONFIG['change_gain_min'] * (stable['score'] if stable else 0)):
        return changed, 'detected', 'relocalized_borders_and_appearance'
    if (stable and appearance_ok(stable) and stable['score'] >= CONFIG['stable_score_min']
            and min(stable['vertical'], stable['horizontal']) >= CONFIG['stable_side_min']):
        return stable, 'detected', 'borders_and_appearance'
    if proposals and max(p['score'] for p in proposals.values()) >= CONFIG['stable_score_min']:
        reason = 'appearance_mismatch' if not any(appearance_ok(p) for p in proposals.values()) else 'weak_or_conflicting_borders'
        return None, 'ambiguous', reason
    return None, 'not_detected', 'insufficient_border_evidence'


def read_frame(cap, timestamp):
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
    ok, image = cap.read()
    if not ok:
        raise RuntimeError(f'Decode failed at {timestamp}')
    actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if abs(actual-timestamp) > 0.1:
        raise RuntimeError(f'Inaccurate seek: requested {timestamp}, got {actual}')
    height = round(image.shape[0] * WORK_WIDTH / image.shape[1])
    return cv2.resize(image, (WORK_WIDTH, height), interpolation=cv2.INTER_AREA), actual


def build_model(video, images):
    clusters = video['clusters']['yunet']
    if not clusters or clusters[0]['count'] < max(3, 0.1*len(images)):
        return None, 'insufficient_recurring_faces'
    if len(clusters) > 1 and clusters[1]['count'] >= 0.8*clusters[0]['count']:
        return None, 'competing_face_clusters'
    seed = clusters[0]
    corner = face_corner(seed['median_face_bbox'])
    h, w = images[0].shape[:2]
    face = canonical_face(seed['median_face_bbox'], corner, w, h)
    mask = allowed_map((h,w), face)
    maps = [border_maps(orient(im, corner)) for im in images]
    aggregate = np.median(np.stack([maps[i][0] for i in seed['frame_indices']]), axis=0)
    candidates = peaks(aggregate, mask)
    if not candidates or candidates[0]['score'] < CONFIG['seed_score_min']:
        return None, 'no_supported_corner_rectangle'
    if len(candidates) > 1 and candidates[0]['score']-candidates[1]['score'] < CONFIG['seed_margin_min']:
        return None, 'competing_panel_geometries'
    locals_ = [local_candidates(m, candidates[0], mask) for m in maps]
    # Freeze an offline bank; exclude the evaluated frame from its own match.
    bank = [(i, appearance(orient(images[i],corner), p['stable'])) for i,p in enumerate(locals_)
            if i in seed['frame_indices'] and p.get('stable',{}).get('score',0) >= CONFIG['bank_score_min']]
    if len(bank) < 3:
        return None, 'insufficient_appearance_references'
    return {'corner':corner, 'size':(w,h), 'mask':mask, 'candidates':candidates,
            'bank':bank, 'locals':locals_}, None


def face_inside(faces, bbox):
    x, y, w, h = bbox
    return any(x <= b[0]+b[2]/2 <= x+w and y <= b[1]+b[3]/2 <= y+h for b in faces)


def infer(image, sample, model, original_size, model_error=None, proposals=None, exclude_index=None):
    result = {'index':sample.get('index'), 'sample_id':sample['sample_id'],
              'requested_s':sample['requested_s'], 'actual_s':sample['actual_s'],
              'status':'ambiguous', 'webcam':None, 'bbox_px':None, 'face_in_candidate':False}
    if model is None:
        result['reason'] = model_error
        return result
    canonical = orient(image, model['corner'])
    if proposals is None:
        proposals = local_candidates(border_maps(canonical), model['candidates'][0], model['mask'])
    # Do not mutate cached proposals; distance depends on excluded reference index.
    proposals = {k: dict(p, appearance_distance=appearance_distance(appearance(canonical,p), model['bank'],exclude_index))
                 for k,p in proposals.items()}
    selected, status, reason = select_proposal(proposals)
    result.update(status=status, reason=reason, evidence=proposals)
    candidate = selected or proposals.get('changed') or model['candidates'][0]
    work_w, work_h = model['size']
    work_box = panel_box(candidate['w'], candidate['h'], model['corner'], work_w, work_h)
    normalized = (np.array(work_box) / [work_w, work_h, work_w, work_h]).tolist()
    result['face_in_candidate'] = face_inside(sample.get('yunet',[]), normalized)
    result['candidate_bbox_normalized'] = normalized
    if selected:
        ow, oh = original_size
        pw = round(selected['w'] * ow / work_w)
        ph = round(selected['h'] * oh / work_h)
        bbox = panel_box(pw, ph, model['corner'], ow, oh)
        result['bbox_px'] = bbox
        result['webcam'] = dict(zip(('x','y','w','h'), (np.array(bbox)/[ow,oh,ow,oh]).tolist()))
    return result


def dense_timestamps(frames, duration, maximum_centers=6):
    suspects = [f for f in frames if not f['face_in_candidate'] or f['status'] != 'detected']
    if len(suspects) > maximum_centers:
        indices = np.linspace(0,len(suspects)-1,maximum_centers).round().astype(int)
        suspects = [suspects[int(i)] for i in indices]
    existing = [f['actual_s'] for f in frames]
    times = {round(f['actual_s']+offset,3) for f in suspects for offset in (-2,-1,1,2)}
    # Validate the rounded seek target, not its pre-rounding value near EOF.
    return sorted(t for t in times if 0 <= t < duration and all(abs(t-old)>0.1 for old in existing))


def save_sheet(images, rows, path):
    canvas = np.zeros((390*((len(rows)+2)//3), 640*3, 3), np.uint8)
    for i,(image,row) in enumerate(zip(images,rows)):
        tile = np.zeros((390,640,3),np.uint8)
        tile[:image.shape[0],:image.shape[1]] = image
        if row['webcam']:
            b = row['webcam']
            x,y,w,h = np.rint(np.array([b['x'],b['y'],b['w'],b['h']])*[640,image.shape[0],640,image.shape[0]]).astype(int)
            color = (0,210,255) if not row['face_in_candidate'] else (60,230,60)
            cv2.rectangle(tile,(x,y),(x+w-1,y+h-1),color,2)
        cv2.putText(tile,f"{row['sample_id']} {row['actual_s']:.1f}s {row['status']} face:{int(row['face_in_candidate'])}",
                    (5,379),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
        yy,xx = (i//3)*390,(i%3)*640
        canvas[yy:yy+390,xx:xx+640] = tile
    if not cv2.imwrite(str(path),canvas):
        raise RuntimeError(f'Cannot write {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-dense',action='store_true',help='Only the original 144 sparse samples')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    source = json.loads((OUT/'results.json').read_text())
    output = {'schema_version':1, 'scope':'offline corner-attached rectangles; no exact temporal segments',
              'config':CONFIG, 'work_width':WORK_WIDTH, 'opencv':cv2.__version__,
              'model_sha256':source['model_sha256'], 'videos':[]}
    detector = cv2.FaceDetectorYN.create(str(OUT/'yunet.onnx'),'',(640,360),0.8,0.3,5000)
    for video in source['videos']:
        start = time.monotonic()
        cap = cv2.VideoCapture(str(ROOT/video['file']))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video['id']}")
        try:
            decoded = [read_frame(cap,f['requested_s']) for f in video['frames']]
            images = [im for im,_ in decoded]
            model, error = build_model(video,images)
            rows = []
            for i,(im,actual) in enumerate(decoded):
                sample = dict(video['frames'][i], actual_s=actual, sample_id=f'#{i:02d}')
                rows.append(infer(im,sample,model,(video['width'],video['height']),error,
                                  model['locals'][i] if model else None, i))
            dense_rows, dense_images = [], []
            if not args.skip_dense:
                for i,timestamp in enumerate(dense_timestamps(rows,video['duration_s'])):
                    im,actual = read_frame(cap,timestamp)
                    detector.setInputSize((im.shape[1],im.shape[0]))
                    _,faces = detector.detect(im)
                    boxes = [] if faces is None else [(f[:4]/[im.shape[1],im.shape[0],im.shape[1],im.shape[0]]).tolist() for f in faces]
                    sample = {'sample_id':f'd{i:02d}','requested_s':timestamp,'actual_s':actual,'yunet':boxes}
                    dense_rows.append(infer(im,sample,model,(video['width'],video['height']),error))
                    dense_images.append(im)
        finally:
            cap.release()
        for page in range(4):
            save_sheet(images[page*12:(page+1)*12],rows[page*12:(page+1)*12],OUT/f'{video["id"]}-panel-page{page+1}.jpg')
        save_sheet(images[::4],rows[::4],OUT/f'{video["id"]}-panel-overview.jpg')
        for page in range((len(dense_rows)+11)//12):
            save_sheet(dense_images[page*12:(page+1)*12],dense_rows[page*12:(page+1)*12],OUT/f'{video["id"]}-panel-dense{page+1}.jpg')
        entry = {'id':video['id'],'file':video['file'],'width':video['width'],'height':video['height'],
                 'duration_s':video['duration_s'],'model_error':error,
                 'corner':model['corner'] if model else None,'candidates':model['candidates'] if model else [],
                 'appearance_reference_indices':[i for i,_ in model['bank']] if model else [],
                 'frames':rows,'dense_frames':dense_rows,'timing_s':time.monotonic()-start}
        output['videos'].append(entry)
        print(video['id'],dict(Counter(f['status'] for f in rows)),
              'without face',sum(f['status']=='detected' and not f['face_in_candidate'] for f in rows),
              'dense',dict(Counter(f['status'] for f in dense_rows)),flush=True)
    (OUT/'panel-results.json').write_text(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
