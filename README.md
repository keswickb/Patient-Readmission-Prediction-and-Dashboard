🏥 Patient Readmission Prediction and Dashboard
 
 Overview

This project predicts the likelihood of patient readmission within 30 days of hospital discharge using machine learning. It combines data science workflows (EDA, preprocessing, modeling, and inference) with a Streamlit dashboard for interactive predictions.

The project demonstrates a full end-to-end ML pipeline:

* Data exploration & cleaning

* Feature engineering & preprocessing

* Model training, evaluation, and persistence

* Interactive dashboard for real-time predictions

Tech Stack:

* Languages & Tools: Python, Pandas, NumPy, Matplotlib, Seaborn

* Machine Learning: scikit-learn (Logistic Regression, Random Forest, etc.)

* Model Persistence: Joblib

* Web App / Dashboard: Streamlit

* Version Control: Git + GitHub

Workflow:

1. Exploratory Data Analysis (EDA)

* Inspected patient demographics, diagnoses, and discharge patterns

* Identified missing values and class imbalance in readmission rates

* Visualized correlations with Seaborn heatmaps and countplots


2. Preprocessing

Encoded categorical variables using One-Hot Encoding

Scaled numerical features with StandardScaler

Saved preprocessing artifacts (scaler.pkl, feature_cols.pkl)


3. Modeling

Split data into training/validation sets

Trained multiple models:

Logistic Regression

Random Forest

(extendable to XGBoost/LightGBM)

Evaluated with Accuracy, Precision, Recall, F1, ROC-AUC

Saved best model (pipeline_model.pkl)


4. Inference

* Aligned test set columns with training columns

* Applied scaler and preprocessing consistently

* Generated predictions and probabilities

* Saved final results as submission.csv


5. Interactive Dashboard

Built with Streamlit:

* Sidebar inputs for patient details (age, gender, diagnosis, etc.)

* Backend loads trained model + scaler

* Prediction displayed with probability score

* Show engineered features used by the model


Run with: streamlit run dashboards/app.py


📊 Example Dashboard

The dashboard allows clinicians and stakeholders to input patient data and receive real-time readmission risk predictions.

https://github.com/user-attachments/assets/22a13b67-9d8d-4718-9d38-492d5a52e006

<img width="1440" height="900" alt="Screenshot of Streamlit" src="https://github.com/user-attachments/assets/9c4169da-2cc8-488c-9d23-f2e56f84324b" />


