# Chest X-Ray Pneumonia Classification — Temporal CNN Analysis

This project explores **time-dependent performance of Convolutional Neural Networks (CNNs)** for pneumonia detection from chest X-rays.  
It compares three architectures — **Baseline CNN**, **Deep CNN + L2**, and **Residual CNN** — on a synthetic dataset inspired by the RSNA Pneumonia Detection Challenge.

---

## 🚀 Features
- Synthetic dataset generator (Normal vs Pneumonia X-rays)
- Three CNN architectures implemented in TensorFlow/Keras
- Training with EarlyStopping + ReduceLROnPlateau
- Evaluation metrics: Accuracy, AUC-ROC, Sensitivity, Specificity
- Temporal analysis across weekly data batches
- Automatically generated figures for training curves, confusion matrices, ROC curves, and metric comparison

---

## 📊 Results
- **Residual CNN** achieved the best validation performance:
  - Accuracy ≈ 0.90  
  - AUC ≈ 0.95  
  - Sensitivity ≈ 0.93  
  - Specificity ≈ 0.88  
- Temporal analysis showed **stable AUC = 1.0** across weeks, confirming no domain shift.

---

## 📂 Repository Contents
- `report/AI_CCP_Report.pdf` — Full academic report  
- `code/cnn_xray_pneumonia.py` — CPU‑optimised training pipeline  
- `figures/` — Training curves, confusion matrices, ROC curves, temporal analysis, and metric comparison  
- `requirements.txt` — Python dependencies  

---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/ChestXRay-CNN-Temporal-Analysis.git
cd ChestXRay-CNN-Temporal-Analysis
pip install -r requirements.txt
```

## 📌 Academic Context
This project was developed as part of the **Complex Computing Problem (CCP) assignment** for the Artificial Intelligence program at Dawood University of Engineering & Technology.  
It demonstrates the ability to design, implement, and evaluate CNNs for medical imaging tasks, with emphasis on **temporal generalisation** and **deployment readiness**.

---

## 👩‍💻 Author
**Maryam Sohail Ahmed**  
Undergraduate, BSAI — Dawood University of Engineering & Technology  




