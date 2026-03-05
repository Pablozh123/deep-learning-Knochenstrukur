# Skeletal Age & Ethnicity Estimation from Hand X-Rays

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pablozh123/deep-learning-Knochenstrukur/blob/main/2_0.ipynb)

A multi-task deep learning system that predicts a patient's **age** and **ethnicity** from a single hand X-ray image — with no other clinical data required.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-1.x-CC0000?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?style=flat-square)
![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

---

## STAR Summary

### Situation
Determining skeletal age from X-ray images is a core task in paediatric radiology and forensic medicine. Manual assessment by a radiologist is time-consuming, subjective, and costly. Simultaneously, understanding demographic patterns in bone development has clinical value — yet most existing models treat age estimation and demographic classification as entirely separate problems.

### Task
Build a **single neural network** that solves both problems at once:
1. **Age Regression** — predict the patient's age (in years) from a hand X-ray image.
2. **Ethnicity Classification** — classify the patient into one of four ethnic groups (Hispanic, African-American, Asian, Caucasian).

The model must be **fair across all demographic groups** and generalise well despite the dataset containing only ~1,400 images.

### Action

**Data & Preprocessing**
- Loaded 1,390 hand X-ray images with associated metadata (age, ethnicity).
- Removed 1 broken record, leaving a clean dataset of **1,389 images**.
- Performed exploratory analysis: age range 0–20 years; balanced ethnicity distribution (23–26% per class).
- Created fine-grained age bins and a combined `strata` column (`race + age_bin`) for **stratified splitting** (64 % train / 16 % val / 20 % test), ensuring no demographic group was lost in any split.

**Augmentation Strategy**
- Light affine transforms (rotation ±3°, shift/scale ±3%) to simulate real-world positioning variance.
- Brightness/contrast jitter (±15%) to account for differences between X-ray devices.
- Sensor noise simulation.
- **No horizontal flip** — all images show the left hand; flipping would introduce anatomically incorrect data.

**Model Architecture — Multi-Task ResNet18**
- Pretrained ResNet18 backbone (transfer learning from ImageNet).
- Two task-specific heads branching from the shared backbone:
  - **Age head**: linear regression with `SmoothL1Loss` (robust to outliers).
  - **Ethnicity head**: 4-class softmax with `CrossEntropyLoss` + class weights (ensures fairness for under-represented groups).
- Optimizer: AdamW (lr = 3e-4), Mixed Precision training, batch size 32, 10 epochs.

**Interpretability**
- Grad-CAM visualisations confirm the model attends to the **epiphyseal growth plates** — the clinically correct bone regions for age assessment.

**Baseline Comparison**
- A custom CNN trained from scratch served as a baseline to quantify the value of transfer learning.

### Result

| Model | Age MAE (yrs) | Age RMSE (yrs) | Ethnicity Accuracy | Ethnicity F1-macro |
|---|---|---|---|---|
| **ResNet18** (transfer learning) | **1.22** | **1.75** | **71.6 %** | **72.3 %** |
| Custom CNN (from scratch) | 3.62 | 4.24 | 32.0 % | 25.1 % |

The ResNet18 model estimates age with an **average error of just 1.22 years** on unseen test images, and correctly classifies ethnicity in **~72 % of cases** — a 3x improvement over the from-scratch baseline. Stratified splitting and class-weighted loss ensured consistent performance across all four demographic groups.

---

## How to Run

### Prerequisites
- Google account with Google Drive access
- Dataset CSV (`datastats.csv`) and images placed in `MyDrive/Data/`

### Steps

1. **Open the notebook in Google Colab**
   Click the badge at the top of this README, or open [`2_0.ipynb`](2_0.ipynb) manually.

2. **Mount Google Drive**
   The first cell mounts your Drive and sets all paths automatically.

3. **Run all cells in order** (`Runtime > Run all`)
   The notebook will:
   - Load and clean the dataset
   - Perform exploratory data analysis and visualisations
   - Split data (stratified, reproducible with `random_state=42`)
   - Train ResNet18 and the Custom CNN baseline
   - Evaluate both models on the held-out test set
   - Generate Grad-CAM heatmaps for interpretability

4. **Results** are printed as a comparison table and saved plots in your Drive.

> **Note:** A GPU runtime is strongly recommended (`Runtime > Change runtime type > T4 GPU`). Training takes ~5–10 minutes on a free Colab GPU.

---

## Project Structure

```
.
└── 2_0.ipynb          # Main notebook (EDA + training + evaluation)
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Stratified split by `race + age_bin` | Prevents any demographic from being absent in val/test sets |
| SmoothL1 for age regression | Robust to outliers; more stable than pure MAE or MSE |
| Class-weighted CrossEntropy | Penalises errors on minority groups equally — model fairness |
| No horizontal flip augmentation | Left-hand X-rays only; flipping creates anatomically wrong inputs |
| Transfer learning (ResNet18) | Only ~1,400 images available; pretrained weights prevent overfitting |
