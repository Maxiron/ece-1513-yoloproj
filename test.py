"""
ALPR Test Script — Automatic License Plate Recognition

Runs the full ALPR pipeline (YOLO detection + preprocessing + OCR + post-processing)
on a given image or video and prints detected license plate numbers.

Usage:
    python test.py                                # uses default image (data/car_lp_image.png)
    python test.py data/car_lp_image.png          # specific image
    python test.py data/sample_traffic_video.mp4  # video file
    python test.py --ocr paddle                   # use PaddleOCR instead of EasyOCR
"""

import argparse
import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Constants ──
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = str(SCRIPT_DIR / "data" / "car_lp_image.png")
MODEL_PATH = str(SCRIPT_DIR / "models" / "best.pt")

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

PLATE_PATTERNS = [
    r"^[A-Z]{4}[0-9]{3}$",
    r"^[A-Z]{3}[0-9]{4}$",
    r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$",
    r"^[0-9]{1,4}[A-Z]{1,3}[0-9]{0,4}$",
    r"^[A-Z0-9]{5,8}$",
]


# ── Preprocessing ──
def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
    TARGET_HEIGHT = 64
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = TARGET_HEIGHT / h
    resized = cv2.resize(gray, (max(1, int(w * scale)), TARGET_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(resized)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    return binary


# ── OCR ──
def run_ocr_easyocr(plate_binary: np.ndarray, reader) -> tuple:
    results = reader.readtext(
        plate_binary,
        allowlist=OCR_ALLOWLIST,
        detail=1,
        paragraph=False,
        min_size=5,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.4,
        width_ths=0.7,
    )
    if not results:
        return "", 0.0
    results_sorted = sorted(results, key=lambda r: r[0][0][0])
    texts = [r[1] for r in results_sorted]
    confs = [r[2] for r in results_sorted]
    raw_text = " ".join(texts).strip()
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return raw_text, mean_conf


def run_ocr_paddle(plate_binary: np.ndarray, reader) -> tuple:
    result = reader.ocr(plate_binary, cls=False)
    if not result or not result[0]:
        return "", 0.0
    detections = result[0]
    detections_sorted = sorted(detections, key=lambda d: d[0][0][0])
    texts = [d[1][0] for d in detections_sorted]
    confs = [d[1][1] for d in detections_sorted]
    raw_text = " ".join(texts).strip()
    raw_text = re.sub(r"[^A-Za-z0-9 ]", "", raw_text)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return raw_text, mean_conf


# ── Post-processing ──
def postprocess_plate(raw_text: str) -> dict:
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    matched_pattern = None
    for pattern in PLATE_PATTERNS:
        if re.match(pattern, cleaned):
            matched_pattern = pattern
            break
    return {
        "raw": raw_text,
        "cleaned": cleaned,
        "valid": matched_pattern is not None,
        "pattern_matched": matched_pattern,
    }


# ── Pipeline ──
class ALPRPipeline:
    def __init__(self, yolo_weights: str, ocr_reader, ocr_engine: str = "easyocr",
                 conf_thresh: float = 0.4, iou_thresh: float = 0.45):
        self.detector = YOLO(yolo_weights)
        self.ocr_reader = ocr_reader
        self.ocr_engine = ocr_engine
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self._run_ocr = run_ocr_paddle if ocr_engine == "paddle" else run_ocr_easyocr

    def process_image(self, img_path: str) -> list:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        det_results = self.detector(
            img_path,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )[0]

        output = []
        for box_obj in det_results.boxes:
            x1, y1, x2, y2 = box_obj.xyxy[0].cpu().numpy().astype(int)
            det_conf = float(box_obj.conf[0])
            if x2 <= x1 or y2 <= y1:
                continue
            plate_crop = img_bgr[y1:y2, x1:x2]
            plate_bin = preprocess_plate(plate_crop)
            raw_text, ocr_conf = self._run_ocr(plate_bin, self.ocr_reader)
            post = postprocess_plate(raw_text)
            output.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": round(det_conf, 3),
                "raw_ocr": raw_text,
                "ocr_conf": round(ocr_conf, 3),
                "plate": post["cleaned"],
                "valid": post["valid"],
            })
        return output


def process_video(pipeline: ALPRPipeline, video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_idx = 0
    all_plates = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, frame)

        try:
            results = pipeline.process_image(tmp_path)
        finally:
            os.unlink(tmp_path)

        for r in results:
            all_plates.append({"frame": frame_idx, **r})
        frame_idx += 1

    cap.release()
    return all_plates


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def main():
    parser = argparse.ArgumentParser(description="ALPR — detect license plates in an image or video")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                        help="Path to an image or video file (default: data/car_lp_image.png)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Path to YOLO weights (default: models/best.pt)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detection confidence threshold (default: 0.4)")
    parser.add_argument("--ocr", choices=["easyocr", "paddle"], default="easyocr",
                        help="OCR engine to use (default: easyocr)")
    args = parser.parse_args()

    input_file = str(Path(args.input).resolve())
    model_file = str(Path(args.model).resolve())

    if not os.path.isfile(input_file):
        print(f"Error: file not found: {args.input}")
        return
    if not os.path.isfile(model_file):
        print(f"Error: model not found: {args.model}")
        return

    print(f"Loading model: {model_file}")
    print(f"OCR engine: {args.ocr}")
    print(f"GPU available: {torch.cuda.is_available()}")

    if args.ocr == "paddle":
        from paddleocr import PaddleOCR
        reader = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=torch.cuda.is_available(),
            show_log=False,
        )
    else:
        import easyocr
        reader = easyocr.Reader(
            lang_list=["en"],
            gpu=torch.cuda.is_available(),
            verbose=False,
        )

    pipeline = ALPRPipeline(
        yolo_weights=model_file,
        ocr_reader=reader,
        ocr_engine=args.ocr,
        conf_thresh=args.conf,
    )

    input_path = input_file
    print(f"Processing: {input_path}")
    print("-" * 60)

    if is_video(input_path):
        plates = process_video(pipeline, input_path)
        if not plates:
            print("No license plates detected in video.")
            return

        seen = {}
        for p in plates:
            text = p["plate"]
            if text and text not in seen:
                seen[text] = p

        print(f"Detected {len(plates)} plate(s) across video frames.")
        print(f"Unique plates: {len(seen)}")
        print("-" * 60)
        for text, info in seen.items():
            status = "VALID" if info["valid"] else "unverified"
            print(f"  Plate: {text:<12}  det_conf={info['det_conf']:.2f}  "
                  f"ocr_conf={info['ocr_conf']:.2f}  [{status}]  "
                  f"(first seen frame {info['frame']})")
    else:
        results = pipeline.process_image(input_path)
        if not results:
            print("No license plates detected.")
            return

        print(f"Detected {len(results)} plate(s):")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            status = "VALID" if r["valid"] else "unverified"
            print(f"  [{i}] Plate: {r['plate']:<12}  det_conf={r['det_conf']:.2f}  "
                  f"ocr_conf={r['ocr_conf']:.2f}  [{status}]")
            print(f"       Raw OCR: \"{r['raw_ocr']}\"  Box: {r['box']}")


if __name__ == "__main__":
    main()
