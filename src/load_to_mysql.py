import pandas as pd
from sqlalchemy import create_engine
import mysql.connector

def load_data_to_mysql(csv_path, db_config, table_name='loans', chunksize=10000):
    print("Creating database connection...")

    engine = create_engine(
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}/{db_config['database']}"
    )

    print(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)
    # Rename columns to match SQL schema
    column_mapping = {
        'id': 'loan_id',
        'issue_d': 'issue_date'
    }
    df.rename(columns=column_mapping, inplace=True)

    schema_columns = [
        'loan_id', 'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'grade', 'sub_grade',
        'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'issue_date',
        'loan_status', 'purpose', 'addr_state', 'dti', 'delinq_2yrs', 'fico_score',
        'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'mort_acc', 'pub_rec_bankruptcies', 'loan_to_income', 'defaulted'
    ]

    df_to_load = df[[col for col in schema_columns if col in df.columns]]
    print(f"Loading {len(df_to_load)} rows to MySQL...")

    df_to_load.to_sql(
        table_name,
        engine,
        if_exists='append',
        index=False,
        chunksize=chunksize,
        method='multi'
    )
    print("Data loaded successfully!")

    verify_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
    result = pd.read_sql(verify_query, engine)
    print(f"Total rows in database: {result['row_count'][0]}")

    engine.dispose()

if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'Subaruforester_1',
        'database': 'credit_risk_db'
    }

    load_data_to_mysql(
        csv_path = '../data/processed/lending_club_clean.csv',
        db_config=db_config,
        table_name='loans'
    )