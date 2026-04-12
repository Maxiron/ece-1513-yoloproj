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

## References

1. Ultralytics. *Using Ultralytics YOLO11 for Automatic Number Plate Recognition.* https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-automatic-number-plate-recognition

2. Redmon, J. et al. *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. https://arxiv.org/pdf/1506.02640

3. Roboflow. *License Plate Recognition Dataset (v12).* https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/12

4. [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR) / [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)

---

<p align="center">
  Made for ECE 1513 — University of Toronto &nbsp;|&nbsp; Winter 2026
</p>
