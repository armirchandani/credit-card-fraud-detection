# 🕵️‍♀️ Credit Card Fraud Detection Using Random Forest

A machine learning project that detects fraudulent credit card transactions using a **Random Forest Classifier**. The model is trained on anonymized transaction data to distinguish fraudulent and legitimate transactions with high accuracy while addressing the challenges of highly imbalanced data.

This project demonstrates the complete machine learning workflow, including data preprocessing, model training, evaluation, and feature importance analysis.

---

## Table of Contents

1. Features
2. How It Works
3. Why It Matters
4. Tech Stack
5. Installation
6. Usage
7. Project Structure
8. Results
9. Future Improvements
10. Contributors

---

# Features

- **Data Preprocessing**
  - Handled missing values using **SimpleImputer**
  - Standardized numerical features with **StandardScaler**
  - Split the dataset into **80% training** and **20% testing**

- **Fraud Detection Model**
  - Built a **Random Forest Classifier**
  - Trained on anonymized credit card transaction data
  - Optimized for highly imbalanced classification problems

- **Model Evaluation**
  - Achieved **99.6% accuracy**
  - Generated confusion matrix and ROC curve
  - Evaluated using precision, recall, F1-score, and AUC

- **Feature Importance**
  - Identified the most influential features contributing to fraud detection
  - Visualized feature importance using bar charts
  - Explored feature correlations with heatmaps

---

# How It Works

1. **Load Dataset**
   - Import anonymized credit card transaction data.

2. **Preprocess Data**
   - Handle missing values
   - Scale numerical features
   - Split into training and testing datasets

3. **Train Model**
   - Build a Random Forest Classifier
   - Learn patterns that distinguish fraudulent transactions

4. **Evaluate Performance**
   - Measure accuracy, precision, recall, F1-score, and ROC-AUC
   - Visualize results with confusion matrices and ROC curves

5. **Analyze Feature Importance**
   - Identify which transaction features contribute most to fraud detection.

---

# Why It Matters

Credit card fraud results in significant financial losses each year.

This project demonstrates how machine learning can:

- Detect fraudulent transactions in real time
- Minimize financial risk
- Improve fraud prevention systems
- Reduce false positives while maintaining high fraud detection rates
- Support data-driven financial security solutions

---

# Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### Machine Learning

- Random Forest Classifier
- StandardScaler
- SimpleImputer

### Development Tools

- Jupyter Notebook
- Git
- GitHub

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook fraud_detection_rf.ipynb
```

---

# Usage

### Load the Dataset

```python
import pandas as pd

df = pd.read_csv("creditcard_.csv")
```

### Train the Random Forest Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)
```

### Evaluate Performance

```python
from sklearn.metrics import classification_report

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
```

---

# Project Structure

```
credit-card-fraud-detection/
│
├── creditcard_.csv
├── fraud_detection_rf.ipynb
├── requirements.txt
└── README.md
```

---

# Results

📊 **99.6% Accuracy**

🎯 **100% Precision** for fraudulent transactions

🔍 **85% Recall**, successfully identifying the majority of fraud cases

📈 **0.99 ROC-AUC Score**, demonstrating excellent model discrimination

### Top Predictive Features

- V12
- V11
- V10
- V14
- V4

---

# Future Improvements

- Address class imbalance using SMOTE
- Compare performance with XGBoost and LightGBM
- Perform hyperparameter optimization using GridSearchCV
- Deploy the model as a web application
- Implement real-time fraud detection using streaming transaction data

---

# Contributors

**Aastha Mirchandani**

Business Analytics Student | University of San Francisco

Interested in Data Science, Machine Learning, FinTech, and Cybersecurity

---

⭐ If you found this project helpful, consider giving it a star!
