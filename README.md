# 🔬 Breast Cancer Prediction App

A Streamlit-based web application that predicts whether a breast tumor is **Benign (non-cancerous)** or **Malignant (cancerous)** using machine learning.
The app also supports batch prediction, model retraining, evaluation metrics visualization, feature range analysi.

---

## 📌 Features

### ✅ Prediction (Manual + Batch)
- Input six tumor features manually or upload a CSV file.
- Predict tumor type: `Benign` or `Malignant`.
- Get confidence scores and explanations.
- Download detailed CSV prediction reports.

### 🔄 Model Retraining
- Upload new data with `target` labels to retrain the model.
- Automatically evaluates:
  - Accuracy
  - Classification report
  - Confusion matrix (heatmap)
- Saves the updated model.

### 📈 Feature Range Validation
- Automatically calculates and stores feature ranges during retraining.
- Highlights any out-of-range values during prediction.
- Optional upload to visualize feature min/max values.

---

## 🧠 Machine Learning Model

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Dataset**: Wisconsin Breast Cancer Dataset (or user-uploaded)
- **Features Used**:
  - Mean Radius
  - Mean Texture
  - Mean Perimeter
  - Mean Area
  - Mean Smoothness
  - Mean Compactness

---

## 🛠️ Tech Stack

| Layer           | Tools/Packages                        |
|----------------|----------------------------------------|
| Frontend       | `Streamlit`                            |
| ML & Data      | `Scikit-learn`, `Pandas`, `NumPy`      |
| Visualization  | `Matplotlib`, `Seaborn`                |
| Persistence    | `pickle` (model), `JSON` (feature ranges) |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/nilesh5911/breast-cancer-streamlit-app.git
cd breast-cancer-streamlit-app
