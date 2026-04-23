from ultralytics import YOLO
from pyzbar.pyzbar import decode as decode_qr
from qreader import QReader
from pylibdmtx.pylibdmtx import decode as decode_dm
import cv2
import numpy as np
import easyocr
from datetime import datetime
import re
import zxingcpp
from PIL import ImageFont, ImageDraw, Image
import urllib.request
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')


def _ensure_font(size=22):
    font_path = "Sarabun-Regular.ttf"
    if not os.path.exists(font_path):
        print("[INFO] Downloading Thai font (one time)...")
        urllib.request.urlretrieve(
            "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf",
            font_path
        )
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

def put_text(img, text, position, font_size=22, color=(0, 200, 100)):
    """Drop-in replacement for cv2.putText — supports Thai and Unicode."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = _ensure_font(font_size)
    r, g, b = color[2], color[1], color[0]
    draw.text(position, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


LAYER1_MODEL   = "best.pt"
QR_CLASS_ID    = 1
LABEL_CLASS_ID = 0

reader = easyocr.Reader(['en','th'])
qreader = QReader()

def order_points(pts):
    pts = pts.astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")

    # Top-left  = smallest (x+y)
    # Bottom-right = largest (x+y)
    # Top-right = smallest (y-x)
    # Bottom-left = largest (y-x)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # TL
    ordered[2] = pts[np.argmax(s)]   # BR

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # TR
    ordered[3] = pts[np.argmax(diff)]  # BL

    return ordered

def text_quality_score(results):
    score = 0.0
    for _, text, conf in results:
        if conf < 0.3:
            continue
        text = text.strip()
        clean = re.sub(r'[^a-zA-Z0-9]', '', text)
        clean_thai  = re.sub(r'[^\u0e00-\u0e7f]', '', text)
        clean_all   = re.sub(r'[^a-zA-Z0-9\u0e00-\u0e7f]', '', text)
        if len(clean_all) == 0:
            continue

        base = len(clean_all) * conf

    
        vowels = len(re.findall(r'[aeiouAEIOU]', clean))
        consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', clean))
        total_alpha = vowels + consonants
        if total_alpha > 0:
            vowel_ratio = vowels / total_alpha
            # Real words: ~35-45% vowels. Penalize extremes (all consonants or all vowels)
            if 0.2 <= vowel_ratio <= 0.6:
                base *= 1.4   
            else:
                base *= 0.5  


        if len(clean) > 3:
            is_upper = clean.isupper()
            is_title = clean[0].isupper() and clean[1:].islower()
            if is_upper or is_title:
                base *= 1.3   # looks intentional
            elif any(c.islower() for c in clean) and any(c.isupper() for c in clean[1:]):
                base *= 0.5  

        if len(clean) >= 4 and conf > 0.6:
            base *= 1.2

        if len(clean_thai) > 0:
            base *= 1.3

        if len(clean_all) >= 4 and conf > 0.6:
            base *= 1.2
        score += base
    return score

def get_high_contrast_variants(img):
    """Generate contrast variants — called inside best_score_for()"""
    variants = []
    h, w = img.shape[:2] if len(img.shape) == 2 else img.shape[:2]

    # Trim 5% border to remove color strips
    pad_y = max(2, int(h * 0.05))
    pad_x = max(2, int(w * 0.05))
    trimmed_img  = img[pad_y:h-pad_y, pad_x:w-pad_x]
    gray = cv2.cvtColor(trimmed_img, cv2.COLOR_BGR2GRAY) if len(trimmed_img.shape) == 3 else trimmed_img

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    lut   = np.array([((i / 255.0) ** 0.4) * 255
                      for i in range(256)], dtype=np.uint8)
    brightened = cv2.LUT(gray, lut)
    _, otsu = cv2.threshold(brightened, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variants.append(trimmed_img)
    variants.append(gray)
    variants.append(clahe.apply(gray))
    variants.append(brightened)
    variants.append(cv2.bitwise_not(gray))
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))
    variants.append(cv2.equalizeHist(gray))

    if len(trimmed_img.shape) == 3:
        for ch in cv2.split(trimmed_img):
            variants.append(ch)

    return variants


def best_score_for(candidate):
    """Try all contrast variants for a candidate rotation — called inside correct_orientation()"""
    best       = 0.0
    best_texts = []

    for v in get_high_contrast_variants(candidate):
        try:
            results = reader.readtext(v, detail=1, paragraph=False,
                                      width_ths=0.9, text_threshold=0.4,
                                      low_text=0.25)
           
            if not results:
                results = reader.readtext(v, detail=1, paragraph=False,
                                          width_ths=1.0,
                                          text_threshold=0.2,
                                          low_text=0.1,
                                          mag_ratio=2.0)
            s = text_quality_score(results)
            if s > best:
                best       = s
                best_texts = [t for _, t, c in results if c > 0.2]
        except Exception:
            continue

    return best, best_texts


def correct_orientation(crop):
    """Force landscape, upscale, then pick best rotation — called inside get_persepctive()"""
    h, w = crop.shape[:2]

    
    if h > w:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        h, w = crop.shape[:2]

   
    target_h = 80
    if h < target_h:
        scale = target_h / h
        crop  = cv2.resize(crop, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    candidates = {
        0:   crop,
        180: cv2.rotate(crop, cv2.ROTATE_180),
    }

    best_crop  = crop
    best_score = -1.0

    for angle, candidate in candidates.items():
        score, texts = best_score_for(candidate)
        print(f"  [orient] {angle}°  score={score:.2f}  texts={texts}")
        if score > best_score:
            best_score = score
            best_crop  = candidate

    return best_crop

def get_persepctive(image, points, pad=10,debug_idx=None):
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    tl = [tl[0] - pad, tl[1] - pad]
    tr = [tr[0] + pad, tr[1] - pad]
    br = [br[0] + pad, br[1] + pad]
    bl = [bl[0] - pad, bl[1] + pad]

    rect = np.array([tl, tr, br, bl], dtype="float32")
    width_a   = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b   = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a   = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b   = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    if max_width == 0 or max_height == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M      = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

   

    warped = correct_orientation(warped)

    
    if debug_idx is not None:
        cv2.imwrite(f"debug_crop_{debug_idx}.jpg", warped)
        print(f"  [DEBUG] Saved debug_crop_{debug_idx}.jpg  shape={warped.shape}")

    return warped


def layer1_detect(frame, model, conf=0.3):
    results = list(model(frame, conf=conf, task='obb'))
    detections = []
    for r in results:
        if r.obb is None:
            continue

        obb_points = r.obb.xyxyxyxy.cpu().numpy()
        classes    = r.obb.cls.cpu().numpy()
        scores     = r.obb.conf.cpu().numpy()

        for i in range(len(obb_points)):
            pts   = obb_points[i]
            cls   = int(classes[i])
            score = float(scores[i])
            label = "qr_code" if cls == QR_CLASS_ID else "medicine_label"

            # Show confidence for every detection including filtered ones
            print(f"  [Box {i}] {label} ({cls})  conf={score:.4f}")

            # Per-class confidence threshold
            # QR codes are consistent — lower threshold fine
            # Medicine labels need higher threshold to avoid false positives
            min_conf = 0.5 if cls == QR_CLASS_ID else 0.75
            if score < min_conf:
                print(f"    → SKIPPED (below {min_conf} threshold)")
                continue

            pad = 15 if cls == QR_CLASS_ID else 10
            try:
                crop = get_persepctive(frame, pts, pad=pad, debug_idx=i)
            except Exception as e:
                print(f"  [WARN] Perspective warp failed box {i}: {e}")
                continue

            if crop.size == 0:
                continue

            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            bbox = (int(min(x_coords)), int(min(y_coords)),
                    int(max(x_coords)), int(max(y_coords)))

            detections.append({
                "cls":   cls,
                "label": "qr_code" if cls == QR_CLASS_ID else "text_label",
                "bbox":  bbox,
                "crop":  crop,
                "conf":  score,
            })

    annotated = results[0].plot() if results else frame
    return annotated, detections

#layer 2
def preprocess_qr(crop):
    h, w = crop.shape[:2]
    scale = max(1, 600 // max(min(h, w), 1))
    if scale > 1:
        crop = cv2.resize(crop, (w * scale, h * scale),
                          interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"debug_crop_QR.jpg", crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    
    blurred   = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)

    
    _, otsu_sharp = cv2.threshold(sharpened, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    _, otsu_gray  = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu_sharp, sharpened, otsu_gray, gray, crop
def layer2_read_qr(crop):
    if crop.size == 0:
        return None

    variants = preprocess_qr(crop)
    for img in variants:
       result = zxingcpp.read_barcodes(img)
       if result:
           return result[0].text
    return None

def score_result(lines):
    """
    Score a set of OCR results.
    Penalize quantity — reward clean high-confidence reads.
    """
    if not lines:
        return 0.0
    
    confs = [conf for _, _, conf in lines]
    
    avg_conf    = sum(confs) / len(confs) 
    high_conf   = sum(1 for c in confs if c > 0.7) 
    line_penalty = max(1, len(lines) * 0.5) 
    
    return (avg_conf + high_conf) / line_penalty


def deduplicate_lines(lines):
    """
    Remove duplicate or near-duplicate text lines.
    Fixes '100 Prenolol 25 Prenolol 25' type results.
    """
    seen  = []
    clean = []
    
    for bbox, text, conf in lines:
        text_stripped = text.strip().lower()
        
        # Skip if very similar to something already kept
        is_duplicate = False
        for s in seen:
            # Check if one is a substring of the other
            if text_stripped in s or s in text_stripped:
                is_duplicate = True
                break
            # Check character overlap ratio
            shorter = min(len(text_stripped), len(s))
            if shorter > 0:
                matches = sum(a == b for a, b in 
                              zip(text_stripped, s))
                if matches / shorter > 0.7:
                    is_duplicate = True
                    break
        
        if not is_duplicate and len(text_stripped) > 1:
            seen.append(text_stripped)
            clean.append((bbox, text, conf))
    
    return clean

def merge_lines(lines):
    if not lines:
        return lines
    
    lines.sort(key=lambda r: r[0][0][0])

    merged = []
    used = set()

    for i, (box_i, text_i,conf_i) in enumerate(lines):
        if i in used:
            continue

        y_center_i = (box_i[0][1] + box_i[2][1]) /2
        height_i = abs(box_i[2][1] - box_i[0][1])

        group_text = text_i
        group_conf = conf_i
        group_box = box_i
        for j, (bbox_j, text_j, conf_j) in enumerate(lines):
            if j <= i or j in used:
                continue

            y_center_j = (bbox_j[0][1] + bbox_j[2][1]) / 2

            # Same row = vertical centers within one line-height of each other
            if abs(y_center_i - y_center_j) < height_i * 0.8:
                group_text += text_j
                group_conf  = (group_conf + conf_j) / 2
                used.add(j)

        used.add(i)
        merged.append((group_box, group_text, group_conf))

    return merged

def layer3_read_label(crop):
    """Final OCR — receives already-oriented, already-upscaled crop"""
    if crop.size == 0:
        return None

    # No upscaling here — correct_orientation already handled it
    kernel = np.array([[0, -1, 0],
                       [-1,  5,-1],
                       [0, -1, 0]])
    crop = cv2.filter2D(crop, -1, kernel)

    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    best_lines = []
    best_score = -1.0

    for img in get_high_contrast_variants(crop):
        try:
            results = reader.readtext(img, detail=1, paragraph=False,
                                      width_ths=0.9, text_threshold=0.5,
                                      low_text=0.3)
            lines = [(bbox, text, conf)
                     for bbox, text, conf in results if conf > 0.25]
            s = score_result(lines)
            if s > best_score:
                best_score = s
                best_lines = lines
        except Exception:
            continue

    if not best_lines:
        return None

    best_lines = merge_lines(best_lines)
    best_lines.sort(key=lambda r: r[0][0][1])
    best_lines = deduplicate_lines(best_lines)
    return "\n".join(text for _, text, _ in best_lines)



def run_snapshot_pipeline(image_path=None):
    model = YOLO(LAYER1_MODEL)

  
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return
        print(f"[INFO] Loaded image: {image_path}")
    else:
        print("[INFO] Opening camera — press SPACE to snapshot, Q to quit.")
        cap   = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps    = cap.get(cv2.CAP_PROP_FPS)

        print(f"[INFO] Camera resolution: {int(width)}x{int(height)} @ {int(fps)}fps")
        frame = None
        while True:
            ret, live = cap.read()
            if not ret:
                break
            display = put_text(live.copy(), "SPACE = snapshot  |  Q = quit",
                               (10, 30), font_size=22, color=(255, 255, 0))
            cv2.imshow("Camera Preview", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                frame = live.copy()  # save clean frame without UI text
                print("[INFO] Snapshot taken.")
                break
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    if frame is None:
        print("[WARN] No frame captured.")
        return

    # ── Layer 1 ──
    print("\n" + "═"*50)
    print("  LAYER 1 — Shape Detection")
    print("═"*50)
    annotated, detections = layer1_detect(frame, model)
    print(f"  Found {len(detections)} region(s):")
    for i, d in enumerate(detections):
        print(f"  [{i+1}] {d['label'].upper()}  conf={d['conf']:.2f}  bbox={d['bbox']}")

    results_log = []
    for i, d in enumerate(detections):
        entry = {
            "index": i + 1,
            "type":  d["label"],
            "bbox":  d["bbox"],
            "conf":  d["conf"],
            "data":  None,
        }

        if d["cls"] == QR_CLASS_ID:
            print(f"\n  LAYER 2 — Reading QR/Barcode [{i+1}]")
            text        = layer2_read_qr(d["crop"])
            entry["data"] = text or "[decode failed]"
            x1, y1, _, _ = d["bbox"]
            label = f"QR: {text[:20]}..." if text and len(text) > 20 else f"QR: {text or 'FAIL'}"
            annotated = put_text(annotated, label, (x1, max(20, y1 - 10)),font_size=22, color=(0, 255, 255))
        else:
            print(f"\n  LAYER 3 — Reading Text Label [{i+1}]")
            text          = layer3_read_label(d["crop"])
            entry["data"] = text or "[no text found]"
            x1, y1, _, _  = d["bbox"]
            first_line    = (text or "").split("\n")[0][:20]
            annotated = put_text(annotated, f"LBL: {first_line}",
                                  (x1, max(20, y1 - 10)),
                                  font_size=22, color=(0, 200, 100))

        results_log.append(entry)

    # ── Print Report ──
    print("\n" + "═"*50)
    print("  SCAN REPORT")
    print("═"*50)
    for entry in results_log:
        print(f"\n  [{entry['index']}] Type : {entry['type'].upper()}")
        print(f"       Conf : {entry['conf']:.2f}")
        print(f"       BBox : {entry['bbox']}")
        print(f"       Data :")
        for line in str(entry['data']).splitlines():
            print(f"              {line}")
    print("\n" + "═"*50)

    
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"scan_result_{ts}.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"  Saved → {out_path}")

    cv2.imshow("Scan Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results_log


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_snapshot_pipeline(image_path=sys.argv[1])
    else:
        run_snapshot_pipeline()