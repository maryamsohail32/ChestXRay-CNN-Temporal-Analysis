# 🫁 Chest X-Ray Pneumonia Classification — CNN vs Transfer Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> Benchmarking three CNN architectures for pneumonia detection on the **real RSNA Pneumonia Detection Challenge dataset** — with a fully transparent, reproducible evaluation pipeline.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Results](#-results)
- [Key Finding](#-key-finding)
- [Limitations](#-limitations)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🔬 Overview
This project compares a **Simple CNN**, a **Deeper CNN**, and a **MobileNetV2 transfer-learning model** for binary pneumonia classification (Normal vs. Pneumonia) on real, patient-level chest X-ray data.

An earlier version of this project used a synthetic dataset and reported inflated, internally inconsistent metrics (AUC ≈ 1.0 alongside confusion matrices showing the model predicting a single class every time). This version replaces that entirely: retrained and evaluated on real data, with every number below coming from a genuine held-out test set the models never saw during training.

**Why that matters:** a lot of student ML projects report numbers that look impressive but don't hold up under scrutiny. This one is built the other way around — the methodology is the point, not just the score.

---

## 📊 Dataset
| | |
|---|---|
| **Source** | RSNA Pneumonia Detection Challenge (Kaggle) |
| **Split** | 70% train / 15% val / 15% test — stratified by patient, `random_state=42` |
| **Class balance** | ~22.5% pneumonia-positive across all splits |
| **Preprocessing** | DICOM → grayscale → resized to 150×150 → normalized to [0,1] |

Splitting by **patient ID** (not by image) avoids the classic leakage bug where the same patient shows up in both train and test sets.

---

## 🧠 Results

| Model | Test Accuracy | Test AUC |
|---|:---:|:---:|
| Simple CNN | 0.682 | 0.805 |
| Deep CNN | 0.791 | 0.781 |
| **MobileNetV2 (transfer learning)** | 0.700 | **0.838** 🏆 |

All metrics: `.evaluate()` on a held-out test set. Class weighting applied during training to handle the ~1:3.4 class imbalance.

---

## 🔎 Key Finding

> **Accuracy can lie. AUC doesn't — not here, anyway.**

The Deep CNN has the *highest* accuracy (0.79) but the *lowest* AUC (0.78). MobileNetV2 has middling accuracy but the best AUC. Under class imbalance, a model can rack up accuracy just by leaning toward the majority class — AUC is what actually reveals which model separates the two classes well. That gap between the two metrics is itself a finding, not noise.

---

## ⚠️ Limitations
- Only 5 training epochs — likely underfit; longer training could shift these numbers
- No confusion matrix / precision-recall breakdown yet for the real-data models
- Images downsampled to 150×150 from the original 1024×1024 — may lose fine-grained detail
- Single split, no cross-validation — results are from one run, not averaged

---

## 🛠️ Tech Stack
`Python` · `TensorFlow / Keras` · `scikit-learn` · `pydicom` · `OpenCV`

---

## 👩‍💻 Author
**Maryam Sohail Ahmed**
BS Artificial Intelligence, Dawood University of Engineering & Technology
[GitHub](https://github.com/maryamsohail32) · [Portfolio](https://maryamsohailahmed.vercel.app)
