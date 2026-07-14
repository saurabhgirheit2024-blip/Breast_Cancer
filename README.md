# Breast Cancer Classification Pipeline

This project implements an end-to-end Machine Learning classification pipeline to predict breast cancer diagnoses (Malignant vs. Benign) using the **Wisconsin Breast Cancer Dataset**. The pipeline features a dynamic, interactive console menu (switch-case) that allows users to train, evaluate, and compare multiple machine learning algorithms on the fly.

---

## 📊 Dataset Overview

The model uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

### Target Variable
*   **`diagnosis`**: Categorized as `'M'` (Malignant) or `'B'` (Benign).
*   *Preprocessing:* Cleanly mapped to `1` (Malignant) and `0` (Benign) to maintain compatibility with gradient boosters and default classification metrics.

### Features (30 Numeric Features)
*   **Mean Attributes:** `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, etc.
*   **Standard Error Attributes:** `radius_se`, `texture_se`, `perimeter_se`, etc.
*   **Worst (Largest) Attributes:** `radius_worst`, `texture_worst`, `perimeter_worst`, etc.

---

## 🛠️ Data Preprocessing & Scaling

1.  **Identifier Removal:** The unique ID column (`id`) is dropped since it has no predictive power.
2.  **Target Encoding:** Mapped categorical string labels `'M'` and `'B'` to numeric values `1` and `0` respectively.
3.  **Feature Scaling:** Standardized features using `StandardScaler` to ensure zero mean and unit variance. This is critical for distance-based models (KNN), optimization-based models (SVC, Neural Networks), and stable training.
4.  **Train-Test Split:** Partitioned the data into training (80%) and testing (20%) sets.

---

## ⚙️ Interactive Model Selection (Switch-Case Menu)

You can select from the following state-of-the-art algorithms:

| Option | Algorithm | Description |
| :--- | :--- | :--- |
| **`1`** | **Random Forest Classifier** | An ensemble of decision trees using bagging to reduce variance. |
| **`2`** | **K-Nearest Neighbors (KNN)** | A distance-based instance classifier that categorizes samples based on proximity. |
| **`3`** | **XGBoost Classifier** | An optimized, highly efficient Gradient Boosting decision tree framework. |
| **`4`** | **Neural Network (MLPClassifier)** | A Multi-layer Perceptron model with backpropagation for non-linear mappings. |
| **`5`** | **Support Vector Classifier (SVC)** | A maximum-margin classifier optimized using kernel tricks. |

---

## 📈 Evaluation Metrics

After running any classifier, the pipeline outputs:
*   **Accuracy** (Overall correctness)
*   **Precision** (Proportion of positive identifications that are correct)
*   **Recall** (Proportion of actual positives identified correctly)
*   **F1 Score** (Harmonic mean of precision and recall)
*   **Classification Report** (Class-wise metrics breakdown)
*   **Confusion Matrix** (Visual representation of True/False Positives and Negatives)

---

## 🚀 Getting Started

### 1. Installation
Install the necessary python dependencies:
```bash
pip install numpy pandas scikit-learn xgboost jupyter
```

### 2. Dataset Setup
Ensure you have the `breast-cancer.csv` file saved locally. Update the file path in cell 2 of the notebook:
```python
df = pd.read_csv("path/to/your/breast-cancer.csv")
```

### 3. Execution
Launch Jupyter and run the notebook:
```bash
jupyter notebook "Breast Cancer.ipynb"
```
Run the cells sequentially. In **Cell 12**, type the number corresponding to your chosen classifier to train and evaluate the model interactively.
