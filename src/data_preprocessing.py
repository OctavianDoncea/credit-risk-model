import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath, sample_size=None):
    print("Loading data...")

    if sample_size:
        df = pd.read_csv(filepath, nrows=sample_size)
    else:
        chunks=[]
        for chunk in pd.read_csv(filepath, chunksize=100000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

    print(f"Loaded {len(df)} rows")

    print("Creating target variable...")

    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]
    df['defaulted'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

    print(f"Default rate: {df['defaulted'].mean()*100:.2f}%")

    columns_to_keep = [
        'id',
        'loan_amnt',
        'funded_amnt',
        'term',
        'int_rate',
        'grade',
        'sub_grade',
        'emp_length',
        'home_ownership',
        'annual_inc',
        'verification_status',
        'issue_d',
        'loan_status',
        'purpose',
        'addr_state',
        'dti',
        'delinq_2yrs',
        'fico_range_low',
        'fico_range_high',
        'inq_last_6mths',
        'open_acc',
        'pub_rec',
        'revol_bal',
        'revol_util',
        'total_acc',
        'mort_acc',
        'pub_rec_bankruptcies',
        'defaulted'
    ]

    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep]

    print("Cleaning columns...")

    if 'int_rate' in df.columns:
        df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '', regex=False).replace('', np.nan)
        df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')

    if 'revol_util' in df.columns:
        df['revol_util'] = df['revol_util'].astype(str).str.replace('%', '', regex=False).replace('', np.nan)
        df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')

    if 'term' in df.columns:
        df['term'] = df['term'].astype(str).str.extract(r'(\d+)', expand=False)
        df['term'] = pd.to_numeric(df['term'], errors='coerce')

    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].replace({
            '< 1 year': '0',
            '10+ years': '10',
            'n/a': None
        })
        df['emp_length'] = df['emp_length'].astype(str).str.extract(r'(\d+)', expand=False)
        df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')

    print("Handling missing values...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print("Creating derived features...")

    #Average FICO score
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2

    #Loan to income ratio
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']

    print("Removing outliers...")

    #Remove extreme values (income, DTI)
    if 'annual_inc' in df.columns:
        df = df[df['annual_inc'] < df['annual_inc'].quantile(0.99)]
        df = df[df['annual_inc'] > 0]

    if 'dti' in df.columns:
        df = df[df['dti'] < 100]

    if 'issue_d' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')

    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df

def save_to_csv(df, filepath):
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")

if __name__ == "__main__":
    df_clean = load_and_clean_data('../data/raw/lending_club.csv', sample_size=50000)

    save_to_csv(df_clean, '../data/processed/lending_club_clean.csv')

    print("\n=== SUMMARY STATISTICS ===")
    print(df_clean.describe())
    print(f"\nDefault rate: {df_clean['defaulted'].mean()*100:.2f}%")