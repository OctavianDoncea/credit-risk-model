import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def create_advanced_features(df):
    # Risk fetures for credit modelling
    print("Creating advanced features..")


    df['credit_utilization'] = df['revol_util'] / 100.0
    df['credit_utilization'] = df['credit_utilization'].clip(0, 1)

    df['debt_burden_score'] = ((df['dti'] / 100) * 0.6 + (df['credit_utilization']) * 0.4)

    df['has_delinquencies'] = (df['delinq_2yrs'] > 0).astype(int)
    df['high_delinquency'] = (df['delinq_2yrs'] >= 2).astype(int)

    df['has_bankruptcy'] = (df['pub_rec_bankruptcies'] > 0).astype(int)

    df['high_inquiries'] = (df['inq_last_6mths'] >= 2).astype(int)

    df['credit_history_depth'] = np.log1p(df['total_acc'])

    df['account_mix_score'] = (np.log1p(df['open_acc']) * 0.5 + np.log1p(df['total_acc']) * 0.5)

    df['loan_income_category'] = pd.cut(
        df['loan_to_income'],
        bins=[0, 0.1, 0.2, 0.3, 0.5, 10],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )

    df['fico_bin'] = pd.cut(
        df['fico_score'],
        bins = [0, 600, 650, 700, 750, 850],
        labels = ['poor', 'fair', 'good', 'very_good', 'excellent']
    )

    df['dti_bins'] = pd.cut(
        df['dti'],
        bins=[0, 10, 20, 30, 40, 100],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )

    grade_median_rate = df.groupby('grade')['int_rate'].transform('median')
    df['rate_premium'] = df['int_rate'] - grade_median_rate

    df['simple_risk_score'] = (
        (df['fico_score'] < 650).astype(int) * 3 +
        (df['dti'] > 30).astype(int) * 2 +
        (df['delinq_2yrs'] > 0).astype(int) * 2 +
        (df['pub_rec_bankruptcies'] > 0).astype(int) * 4 +
        (df['credit_utilization'] > 0.8).astype(int) * 2
    )

    df['verified'] = (df['verification_status'] == 'Verified').astype(int)
    df['owns_home'] = (df['home_ownership'].isin(['OWN', 'MORTGAGE'])).astype(int)
    df['long_term'] = (df['term'] == 60).astype(int)

    print(f"Created {15} new feature groups")
    print(f"Total columns now: {len(df.columns)}")

    return df

def prepare_modelling_data(df):
    feature_columns = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_score", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
                        "revol_util", "total_acc", "mort_acc", "loan_to_income", "credit_utilization", "debt_burden_score", "credit_history_depth", 
                        "account_mix_score", "rate_premium", "simple_risk_score", "has_delinquencies", "high_delinquency", "has_bankruptcy",
                        "high_inquiries", "verified", "owns_home", "long_term", "defaulted"
    ]

    categorical_features = ["grade", "purpose", "home_ownership"]

    df_model = df[feature_columns + categorical_features].copy()
    df_model = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

    print(f"Final modelling dataset shape: {df_model.shape}")
    print(f"Features: {df_model.columns.tolist()}")

    return df_model

if __name__ == "__main__":
    print("Loading data to MySQL...")

    db_user = os.getenv('MYSQL_USER', 'root')
    db_password = os.getenv('MYSQL_PASSWORD', '')
    db_host = os.getenv('MYSQL_HOST', 'localhost')
    db_name = os.getenv('MYSQL_DATABASE', 'credit_risk_db')

    engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')

    df = pd.read_sql("SELECT * FROM loans LIMIT 50000", engine)
    df_enhanced = create_advanced_features(df)
    df_model = prepare_modelling_data(df)
    df_model.to_csv("../data/processed.modelling_data.csv", index=False)

    print(f"\nFeature engineering complete!")
    print("\n== TOP FEATURES BY CORRELATION WITH DEFAULT ==")
    correlations = df_model.corr()['defaulted'].sort_values(ascending=False)
    print(correlations.head(15))

    engine.dispose()