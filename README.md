# AI Medicine Vision

A computer vision system for identifying medicine labels and QR codes from a live camera feed. The system uses a 3-layer pipeline to detect, locate, and read information from medicine packaging — supporting both English and Thai text.

---

## Overview

The system takes a snapshot from a connected webcam and processes it through three layers:

- **Layer 1** — A custom-trained YOLO OBB (Oriented Bounding Box) model detects and locates regions of interest in the frame, classifying each as either a `medicine_label` or a `qr_code`. The OBB model handles medicine boxes held at any angle.
- **Layer 2** — For QR code regions, the crop is preprocessed and decoded using ZXing to extract the encoded data.
- **Layer 3** — For medicine label regions, EasyOCR reads the text from the crop. The system handles any orientation by trying multiple rotations and picking the highest quality result.

---

## Features

- Detects medicine labels and QR codes at any angle using OBB detection
- Perspective correction — de-tilts angled crops before reading
- Automatic orientation correction — handles upside down and sideways labels
- Reads both English and Thai text
- Multiple contrast preprocessing variants for difficult labels (colored text, low contrast)
- Deduplication and line merging for clean OCR output
- Unicode-safe overlay rendering using PIL/Pillow with Sarabun font
- Continuous snapshot mode — take multiple scans without restarting
- Saves annotated result image with timestamp after each scan
- Per-class confidence thresholds to reduce false positives

---

## Requirements

### Hardware
- Webcam minimum 1080p resolution recommended for reliable QR decoding

### Python dependencies

```
ultralytics
opencv-python
easyocr
zxingcpp
pyzbar
pylibdmtx
qreader
Pillow
numpy
```

Install all dependencies:

```bash
pip install ultralytics opencv-python easyocr zxingcpp pyzbar pylibdmtx qreader Pillow numpy
```

---

## Project Structure

```
project/
├── v1.py               # Main pipeline script
├── best.pt             # Trained YOLO OBB model weights
├── Sarabun-Regular.ttf # Thai font (auto-downloaded on first run)
├── scan_result_*.jpg   # Saved annotated results
└── debug_crop_*.jpg    # Debug crops saved during processing
```

---

## Usage

### Live camera mode
Press `SPACE` to take a snapshot. Press `SPACE` again after results are shown to take another. Press `Q` to quit.

```bash
python v1.py
```

### Static image mode
Pass an image path as an argument to process a saved photo instead of using the camera.

```bash
python v1.py path/to/image.jpg
```

---

## Pipeline Detail

### Layer 1 — Shape Detection
The YOLO OBB model (`best.pt`) processes the full frame and returns oriented bounding boxes for every detected region. Each detection is classified as either `medicine_label` (class 0) or `qr_code` (class 1).

Each detected region goes through perspective correction — the four OBB corner points are used to warp the tilted crop into a flat, upright rectangle before being passed downstream.

Confidence thresholds are applied per class:
- `qr_code` — minimum 0.50
- `medicine_label` — minimum 0.75

### Layer 2 — QR Code Reading
QR crops are preprocessed with unsharp masking and Otsu thresholding before being passed to ZXing. ZXing handles rotation natively so no manual rotation loop is needed.

### Layer 3 — Label OCR
Label crops are passed through multiple contrast preprocessing variants (grayscale, CLAHE, Otsu threshold, channel splits, histogram equalization) and read with EasyOCR. The orientation correction step tries 0° and 180° rotations and picks the one that scores highest on text quality. Results are merged, sorted top-to-bottom, and deduplicated before output.

---

## Model Training

The YOLO model was trained using Roboflow for dataset management and annotation.

**Training configuration:**
```python
model = YOLO("yolo11n-obb.pt")
model.train(
    data="data.yaml",
    epochs=200,
    patience=50,
    imgsz=960,
    batch=8,
    device=0,
    task='obb',
    degrees=180.0,
    flipud=0.5
)
```

**Dataset:**
- 2 classes: `medicine_label`, `qr_code`
- ~300-400 images covering multiple medicine brands
- Images taken at varied angles, distances, and lighting conditions
- Includes combo images where QR codes appear on medicine label faces, annotated with both classes separately

---

## Output

After each snapshot the terminal prints a scan report:

```
══════════════════════════════════════════════════
  SCAN REPORT
══════════════════════════════════════════════════

  [1] Type : TEXT_LABEL
       Conf : 0.91
       BBox : (312, 145, 598, 223)
       Data :
              Pencor
              Doxazosin mesylate eq. to
              Doxazosin 1 mg

  [2] Type : QR_CODE
       Conf : 0.87
       BBox : (601, 310, 720, 428)
       Data :
              DOXA-T-@10,25090103070031

══════════════════════════════════════════════════
```

An annotated image is also saved as `scan_result_YYYYMMDD_HHMMSS.jpg`.

---

## Known Limitations

- QR codes embedded directly on medicine label faces may be misclassified as `medicine_label` if the training dataset does not include enough combo-annotated examples
- Very blurry or low-resolution QR crops may fail to decode regardless of preprocessing
- OCR accuracy on highly stylised fonts or very low contrast labels may vary
- Training and inference resolution should match for best detection accuracy
