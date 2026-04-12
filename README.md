# Automatic License Plate Recognition (ALPR)

> **ECE 1513: Intro to Machine Learning — Course Project | University of Toronto, Winter 2026**

An end-to-end machine learning pipeline that detects and reads vehicle license plates from images and video frames. Built with **YOLO26** for detection and **EasyOCR** for character recognition.

---

## Overview

Traditional rule-based image processing struggles with real-world variability — lighting changes, motion blur, camera angles, and plate occlusions. This project tackles those challenges using a four-stage deep learning pipeline:

1. **License plate localization** — fine-tuned YOLO26 object detector
2. **Image preprocessing** — CLAHE, Gaussian denoising, adaptive thresholding
3. **Character recognition** — OCR model (EasyOCR)
4. **Post-processing & validation** — regex-based format validation and character correction

**Applications:** Traffic monitoring · Automated toll collection · Parking management · Law enforcement

---

## Pipeline Architecture

```
Input Image / Video Frame
         │
         ▼
┌─────────────────────────┐
│   YOLO26 Detection      │  ← Fine-tuned on Roboflow LP dataset
│   (Plate Localization)  │
└─────────────┬───────────┘
              │  Bounding box crop
              ▼
┌─────────────────────────┐
│   Image Preprocessing   │  ← CLAHE · Gaussian blur · Adaptive threshold
│   (OpenCV)              │
└─────────────┬───────────┘
              │  Enhanced plate image
              ▼
┌─────────────────────────┐
│   OCR Model             │  ← EasyOCR (alphanumeric)
│   (Character Reading)   │
└─────────────┬───────────┘
              │  Raw text string
              ▼
┌─────────────────────────┐
│   Post-Processing       │  ← Regex validation · Character correction
│   (Validation)          │
└─────────────┬───────────┘
              │
              ▼
     Plate Number Output
```

---

## Project Structure

```
├── data/
│   ├── car_lp_image.png           # Sample test image
│   ├── sample_traffic_video.mp4   # Sample test video
│   └── labelled_plates.csv        # Ground truth for OCR experiments
├── models/
│   └── best.pt                    # Fine-tuned YOLO26 weights
├── ECE1513_ALPR_Project.ipynb     # Full training & evaluation notebook
├── test.py                        # Inference script (image & video)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the ALPR Pipeline

```bash
# Run on the default sample image (data/car_lp_image.png)
python test.py

# Run on a specific image
python test.py data/car_lp_image.png

# Run on a video file
python test.py data/sample_traffic_video.mp4

# Use PaddleOCR instead of EasyOCR
python test.py --ocr paddle
python test.py data/sample_traffic_video.mp4 --ocr paddle

# Adjust detection confidence threshold
python test.py data/car_lp_image.png --conf 0.3

# Use a different model weights file
python test.py data/car_lp_image.png --model models/best.pt
```

The script prints each detected plate number along with detection confidence, OCR confidence, and bounding box coordinates. For video inputs, it deduplicates plates across frames and reports unique detections.

---

## References

1. Ultralytics. *Using Ultralytics YOLO11 for Automatic Number Plate Recognition.* https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-automatic-number-plate-recognition

2. Redmon, J. et al. *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. https://arxiv.org/pdf/1506.02640

3. Roboflow. *License Plate Recognition Dataset (v12).* https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/12

4. [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR) / [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)

---

<p align="center">
  Made for ECE 1513 — University of Toronto &nbsp;|&nbsp; Winter 2026
</p>
