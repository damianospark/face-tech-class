"""
Educational demo pipeline (classic CV) for teaching stepwise face processing.
Saves two-digit prefixed artifacts for enroll and probe images.
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Utilities
def _get_script_dir():
    return Path(__file__).resolve().parent

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# LBP helper
def lbp_image(img, radius=1, points=8):
    # simple circular LBP (neighbors sampled on a grid) but ensure uint8 math
    h, w = img.shape[:2]
    img = img.astype(np.uint8)
    out = np.zeros((h, w), dtype=np.uint8)
    bit = 0
    # sample neighbors in a fixed order to create bits [0..points-1]
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dx == 0 and dy == 0:
                continue
            if bit >= points:
                break
            shifted = np.roll(np.roll(img, -dy, axis=0), -dx, axis=1)
            mask = (shifted >= img).astype(np.uint8)
            out |= (mask << bit)
            bit += 1
    return out

# Simple LBP histogram (uniform not implemented, keep small bins)
def lbp_hist(img, bins=64):
    h = cv2.calcHist([img], [0], None, [bins], [0, 256])
    h = h.flatten().astype(np.float32)
    if h.sum() > 0:
        h /= h.sum()
    return h

# Save helpers
def save_img(path, img):
    cv2.imwrite(str(path), img)

def save_txt(path, text):
    with open(path, 'w') as f:
        f.write(str(text))

# Demo pipeline: enroll/probe pair
def process_pair(enroll_path, probe_path, out_dir):
    """Process one enroll/probe pair and save artifacts into output-enroll and output-chk.

    Filenames always start with a numeric prefix so ordering is obvious.
    """
    script_dir = _get_script_dir()
    out_dir = Path(out_dir)
    out_enroll = out_dir / 'output-enroll'
    out_chk = out_dir / 'output-chk'
    ensure_dir(out_enroll)
    ensure_dir(out_chk)

    enroll = cv2.imread(str(enroll_path))
    probe = cv2.imread(str(probe_path))
    if enroll is None or probe is None:
        raise FileNotFoundError('Enroll or probe image missing')

    # detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def detect_and_crop(img, is_enroll=True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = []
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        dest = out_enroll if is_enroll else out_chk
        if len(faces) == 0:
            h, w = gray.shape
            s = min(h, w)
            y0 = (h-s)//2
            x0 = (w-s)//2
            crop = img[y0:y0+s, x0:x0+s].copy()
            # save original image (no bbox) to indicate detection attempted
            save_img(dest / '01_detection.jpg', img.copy())
            return crop, None
        x, y, w, h = faces[0]
        cx = x + w//2
        cy = y + h//2
        s = max(w, h)
        x0 = max(0, cx - s//2)
        y0 = max(0, cy - s//2)
        x1 = min(img.shape[1], cx + s//2)
        y1 = min(img.shape[0], cy + s//2)
        face = img[y0:y1, x0:x1].copy()
        # draw bbox on original image (un-cropped) to show detection
        vis = img.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0,255,0), 2)
        save_img(dest / '01_detection.jpg', vis)
        return face, (x0, y0, x1-x0, y1-y0)

    enroll_face, enroll_box = detect_and_crop(enroll, is_enroll=True)
    probe_face, probe_box = detect_and_crop(probe, is_enroll=False)

    def approx_align(face, is_enroll=True):
        aligned = face.copy()
        dest = out_enroll if is_enroll else out_chk
        save_img(dest / '02_aligned.jpg', aligned)
        save_txt(dest / '02_aligned_angle.txt', 0)
        return aligned, 0

    enroll_aligned, _ = approx_align(enroll_face, is_enroll=True)
    probe_aligned, _ = approx_align(probe_face, is_enroll=False)

    def preprocess(face, is_enroll=True):
        target = 160
        h, w = face.shape[:2]
        scale = max(target / h, target / w)
        resized = cv2.resize(face, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        h2, w2 = resized.shape[:2]
        y0 = (h2 - target)//2
        x0 = (w2 - target)//2
        crop = resized[y0:y0+target, x0:x0+target].copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        dest = out_enroll if is_enroll else out_chk
        save_img(dest / '03_preprocessed.jpg', eq)
        return eq

    enroll_proc = preprocess(enroll_aligned, is_enroll=True)
    probe_proc = preprocess(probe_aligned, is_enroll=False)

    def part_crops(gray, is_enroll=True):
        h, w = gray.shape
        dest = out_enroll if is_enroll else out_chk
        # eyes
        left_eye = None
        right_eye = None
        if not eyes_cascade.empty():
            eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20))
            eyes_sorted = sorted(eyes, key=lambda e: e[0]) if len(eyes) else []
            if len(eyes_sorted) >= 2:
                ex0, ey0, ew0, eh0 = eyes_sorted[0]
                ex1, ey1, ew1, eh1 = eyes_sorted[1]
                # use lower portion of detected eye boxes to avoid eyebrows
        left_eye = gray[ey0 + eh0//4:ey0 + eh0, ex0:ex0 + ew0]
        right_eye = gray[ey1 + eh1//4:ey1 + eh1, ex1:ex1 + ew1]
        left_box = (ex0, ey0 + eh0//4, ew0, eh0 - eh0//4)
        right_box = (ex1, ey1 + eh1//4, ew1, eh1 - eh1//4)
        if left_eye is None:
            left_eye = gray[h//8:h//3, w//8:w//2]
            left_box = (w//8, h//8, w//2 - w//8, h//3 - h//8)
        if right_eye is None:
            right_eye = gray[h//8:h//3, w//2:w*7//8]
            right_box = (w//2, h//8, w*7//8 - w//2, h//3 - h//8)

        # nose
        nose_crop = None
        try:
            nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
        except Exception:
            nose_cascade = cv2.CascadeClassifier()
        if not nose_cascade.empty():
            noses = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20))
            if len(noses) > 0:
                nx, ny, nw, nh = sorted(noses, key=lambda r: r[2]*r[3], reverse=True)[0]
                nose_crop = gray[ny:ny+nh, nx:nx+nw]
        if nose_crop is None:
            nose_crop = gray[h//3:h*2//3, w//3:w*2//3]
            nose_box = (w//3, h//3, w//3, h//3)

        # mouth
        mouth_crop = None
        try:
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')
        except Exception:
            mouth_cascade = cv2.CascadeClassifier()
        if not mouth_cascade.empty():
            mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            if len(mouths) > 0:
                mx, my, mw, mh = sorted(mouths, key=lambda r: r[2]*r[3], reverse=True)[0]
                mouth_crop = gray[my:my+mh, mx:mx+mw]
        if mouth_crop is None:
            mouth_crop = gray[h*2//3:h*7//8, w//4:w*3//4]
            mouth_box = (w//4, h*2//3, w*3//4 - w//4, h*7//8 - h*2//3)

        parts = {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'nose': nose_crop,
            'mouth': mouth_crop,
        }
        for i, (k, v) in enumerate(parts.items(), start=1):
            save_img(dest / f'05.{i:02d}_{k}.jpg', v)

        # draw actual detected boxes on overlay (convert gray coords to same image scale)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        boxes = []
        if left_box is not None:
            lx, ly, lw, lh = left_box
            boxes.append((lx, ly, lx+lw, ly+lh))
        if right_box is not None:
            rx, ry, rw, rh = right_box
            boxes.append((rx, ry, rx+rw, ry+rh))
        if 'nose_box' in locals():
            nx, ny, nw, nh = nose_box
            boxes.append((nx, ny, nx+nw, ny+nh))
        if 'mouth_box' in locals():
            mx, my, mw, mh = mouth_box
            boxes.append((mx, my, mx+mw, my+mh))
        for (x0, y0, x1, y1) in boxes:
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 2)
        save_img(dest / '04_aligned_parts_overlay.jpg', vis)
        return parts, {'left_box': left_box, 'right_box': right_box, 'nose_box': locals().get('nose_box'), 'mouth_box': locals().get('mouth_box')}

    enroll_parts, enroll_boxes = part_crops(enroll_proc, is_enroll=True)
    probe_parts, probe_boxes = part_crops(probe_proc, is_enroll=False)

    def parts_embedding(parts, is_enroll=True):
        hists = []
        dest = out_enroll if is_enroll else out_chk
        for k in ['left_eye','right_eye','nose','mouth']:
            p = parts[k]
            if p is None or p.size == 0:
                p = np.zeros((16,16), dtype=np.uint8)
            lbp = lbp_image(p)
            # numeric prefix at start
            save_img(dest / f'05.10_embedding_{k}_lbp.jpg', lbp)
            h = lbp_hist(lbp, bins=64)
            hists.append(h)
        emb = np.concatenate(hists)
        if np.linalg.norm(emb) > 0:
            emb = emb / np.linalg.norm(emb)
        return emb

    enroll_emb = parts_embedding(enroll_parts, is_enroll=True)
    probe_emb = parts_embedding(probe_parts, is_enroll=False)

    np.save(out_enroll / '05.20_enroll_embedding.npy', enroll_emb)
    np.save(out_chk / '05.20_probe_embedding.npy', probe_emb)
    save_txt(out_enroll / '05.20_enroll_embedding.txt', ' '.join(map(str, enroll_emb.tolist())))
    save_txt(out_chk / '05.20_probe_embedding.txt', ' '.join(map(str, probe_emb.tolist())))

    # per-part comparisons (cosine on hist blocks)
    part_scores = {}
    for idx, k in enumerate(['left_eye','right_eye','nose','mouth']):
        a = enroll_emb[idx*64:(idx+1)*64]
        b = probe_emb[idx*64:(idx+1)*64]
        sim = float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
        part_scores[k] = sim

    overall = float(np.dot(enroll_emb, probe_emb) / (np.linalg.norm(enroll_emb)*np.linalg.norm(probe_emb) + 1e-12))
    lines = [f'overall_cosine:{overall:.6f}']
    for k, v in part_scores.items():
        lines.append(f'{k}_cosine:{v:.6f}')
    save_txt(out_dir / '07_result.txt', '\n'.join(lines) + '\n')

    # side-by-side compare image (numeric prefix start) and add bottom strip for metrics
    e_vis = cv2.cvtColor(enroll_proc, cv2.COLOR_GRAY2BGR)
    p_vis = cv2.cvtColor(probe_proc, cv2.COLOR_GRAY2BGR)
    comp_body = np.hstack([e_vis, p_vis])

    # prepare text lines (shorter format) and small green font
    lines = [f'overall: {overall:.4f}'] + [f'{k}: {v:.4f}' for k, v in part_scores.items()]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5  # smaller than before
    thickness = 1
    padding_h = 6
    # compute required strip height
    max_w = 0
    line_h = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        max_w = max(max_w, tw)
        line_h = max(line_h, th)
    strip_h = (line_h + padding_h) * len(lines) + padding_h

    # create strip and compose
    strip = np.full((strip_h, comp_body.shape[1], 3), 30, dtype=np.uint8)  # dark grey background
    # write green text left-aligned with small margin
    y = padding_h + line_h
    x = 10
    green = (0, 200, 0)
    for line in lines:
        cv2.putText(strip, line, (x, y), font, scale, green, thickness, cv2.LINE_AA)
        y += line_h + padding_h

    comp = np.vstack([comp_body, strip])
    save_img(out_chk / '06_compare.jpg', comp)
    return overall

if __name__ == '__main__':
    base = _get_script_dir()
    # Prefer local files in the project root: face.png/face-chk.png (or .jpg)
    local_candidates_enroll = [
        base / 'face.png',
        base / 'face.jpg',
    ]
    local_candidates_probe = [
        base / 'face-chk.png',
        base / 'face-chk.jpg',
    ]
    def _pick_existing(cands):
        for p in cands:
            if p.exists():
                return p
        return None
    enroll_src = _pick_existing(local_candidates_enroll)
    probe_src = _pick_existing(local_candidates_probe)
    # Fallback to fi-classic defaults if local files are missing
    if enroll_src is None or probe_src is None:
        src = base.parent / 'fi-classic'
        enroll_src = enroll_src or (src / 'face.jpg')
        probe_src = probe_src or (src / 'face-chk.jpg')
    out = base
    ensure_dir(out)
    cos = process_pair(enroll_src, probe_src, out)
    print('Done. cosine=', cos)
