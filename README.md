# Credit Risk Scoring Model

A machine learning project to predict loan default probability using real-world lending data

## Project Overview

This project builds an end-to-end credit risk scoring system that:
- Processes 2.2M+ loan applications using SQL (MySQL)
- Engineers 15+ risk features (DTI, credit utilization, FICO bins)
- Achieves 72% AUC with Random Forest classifier

## Tech Stack
- **Database:** MySQL 8.0
- **Languages:** Python 3.10, SQL
- **ML Libraries:** scikit-learn, XGBoost, SHAP
- **Data Processing:** Pandas, NumPy
- **Web Framework:** Streamlit
- **Visualization:** Plotly, Matplotlib, Seaborn

## Dataset
- **Source:** [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Size:** 2.2M loan applications
- **Features:** 150+ variables including FICO scores, income, DTI, employment history
- **Target:** Binary default indicator (Charged Off vs Fully Paid)

## Project Structure
```
credit-risk-model/
├── data/
│   ├── raw/                  # Original Kaggle dataset
│   └── processed/            # Cleaned and featured data
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── load_to_mysql.py
├── sql/
│   ├── schema.sql            # MySQL database schema
│   └── queries.sql           # Analytical SQL queries
├── models/
│   ├── xgboost_model.pkl
│   └── feature_importance.csv
├── app.py                    # Streamlit web application
└── requirements.txt
```
## Key Features

### Feature Engineering
- **Credit Utilization:** Revolving balance / credit limit
- **Debt Burden Score:** Weighted combination of DTI and utilization
- **Delinquency Flags:** Binary indicators for payment history
- **FICO Bins:** Categorical risk tiers (Poor, Fair, Good, Excellent)
- **Loan-to-Income Ratio:** Application amount relative to income

### SQL Analytics
- Cohort analysis by origination date
- Risk segmentation by FICO/DTI
- Portfolio performance tracking
- State-level default rates

## Quick Start

### Prerequisites
- Python 3.8+
- MySQL 8.0+
- 4GB+ RAM

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/credit-risk-model.git
cd credit-risk-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup MySQL database
mysql -u root -p < sql/schema.sql
```

### Run Locally
```bash
# 1. Download Kaggle dataset to data/raw/

# 2. Clean and process data
python src/data_preprocessing.py

# 3. Load data to MySQL
python src/load_to_mysql.py

# 4. Engineer features
python src/feature_engineering.py

# 5. Train models
python src/model_training.py

# 6. Launch web app
streamlit run app.py
```