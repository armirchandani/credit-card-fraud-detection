# credit-card-fraud-detection
🕵️‍♀️ Credit Card Fraud Detection Using Random Forest
This project applies machine learning to detect fraudulent credit card transactions using a Random Forest Classifier. The dataset contains anonymized transaction features and a binary label indicating fraud or non-fraud.

📊 Dataset
Source: Synthetic credit card transaction data

Size: 14,720 transactions, 30+ features (PCA-transformed V1 to V28, Amount)

Target: Class (0 = Non-Fraud, 1 = Fraud)

Challenge: Highly imbalanced data (~0.4% fraud)

🔧 Technologies Used
Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn (RandomForestClassifier, StandardScaler, metrics)

Jupyter Notebook

🛠️ Preprocessing Steps
Handled missing values using SimpleImputer

Scaled features using StandardScaler

Split data into training (80%) and testing (20%) sets

🤖 Model: Random Forest Classifier
n_estimators = 100

max_depth = 10

min_samples_split = 5

random_state = 42

Classifier trained on scaled data

📈 Evaluation Metrics
Metric	Value
Accuracy	99.6%
Precision (1)	100%
Recall (1)	85%
Macro F1-Score	96%
AUC Score	0.99

✅ Excellent performance on imbalanced data
📉 Confusion matrix and ROC curve visualized

🔍 Feature Importance
Top predictors of fraud:

V12

V11

V10

V14

V4

Visualized using bar plots and heatmaps for correlation insights.

📌 Key Insights
Fraud detection models must prioritize recall to avoid missing fraud cases.

Tree-based models like Random Forests handle feature interactions and non-linearity well.

Feature scaling and handling imbalance are critical preprocessing steps.

📁 Project Structure
Copy
Edit
credit-card-fraud-detection/
├── creditcard_.csv
├── fraud_detection_rf.ipynb
├── README.md
└── requirements.txt
🚀 Getting Started
Clone this repo

Install dependencies
pip install -r requirements.txt

Open the notebook:
jupyter notebook fraud_detection_rf.ipynb

🙋‍♀️ Author
Aastha Mirchandani
Aspiring finance + data professional | USFCA
Connect with me on LinkedIn
