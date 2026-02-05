# Credit Risk Scoring Model

A machine learning project to predict loan default probability using real-world lending data

#Project Overview

This project builds an end-to-end credit risk scoring system that:
- Predicts loan default probability with 75%+ AUC
- Processes 2M+ loan applications with SQL
- Deploys an interactive web app for risk assessment

# Tech Stack

- **Languages:** Python, SQL
- **Database:** MySQL
- **ML Libraries:** scikit-learn, XGBoost, SHAP
- **Web Framework:** Streamlit
- **Data Analysis:** Pandas, Numpy

## Quick Start
```bash
#Clone repository
git clone https://github.com/YOUR_USERNAME/credit-risk-model.git
cd credit-risk-model

# Install dependencies
pip install -r requirements.txt

# Run StreamLit app
streamlit run app.py
```

## Dataset
Using [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) 
containing 2.2M loan applications.

## Key Features

- **Feature Engineering:** DTI ratio, credit utilization, payment history
- **Model Comparison:** Logistic Regression, Random Forest, XGBoost
- **Explainability:** SHAP values for model interpretability
- **SQL Analytics:** Complex queries for cohort analysis and risk segmentation

## Results

- **AUC Score:** N/A
- **Precision:** N/A
- **Recall:** N/A
