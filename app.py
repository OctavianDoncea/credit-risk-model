import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Credit Risk Scoring System", layout='wide', initial_sidebar_state='expanded')

@st.cache_resource
def get_database_connection():
    db_user = os.getenv('MYSQL_USER', 'root')
    db_password = os.getenv('MYSQL_PASSWORD', '')
    db_host = os.getenv('MYSQL_HOST', 'localhost')
    db_name = os.getenv('MYSQL_DATABASE', 'credit_risk_db')
    
    return create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')

@st.cache_resource
def load_model():
    metadata = joblib.load('models/model_metadata.pkl')

    model_name = metadata['model_name'].lower().replace(' ', '_')
    model_path = f'models/{model_name}_model.pkl'
    model = joblib.load(model_path)

    return model, metadata

@st.cache_data
def load_data_from_db(query, _engine):
    return pd.read_sql(query, _engine)

def main():
    st.title("Credit Risk Scoring System")
    st.markdown('### Machine Learning-Powered Loan Default Prediction')

    page = st.sidebar.selectbox('Navigate', ['Model Overview', 'Risk Scorer', 'Portfolio Analytics', 'SQL Explorer'])

    if page == 'Model Overview':
        show_model_overview()
    elif page == 'Risk Scorer':
        show_risk_scorer()
    elif page == 'Portfolio Analytics':
        show_portfolio_analytics()
    elif page == 'SQL Explorer':
        show_sql_explorer()

def show_model_overview():
    st.header('Model Overview')

    model, metadata = load_model()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Model Type', metadata['model_name'])
    with col2:
        st.metric('AUC Score', f'{metadata['auc']:.4f}')
    with col3:
        st.metric('Training Samples', f'{metadata['train_size']:,}')

    st.markdown('---')

    st.subheader('ROC Curve')
    try:
        st.image('models/roc_curve.png', use_column_width=True)
    except:
        st.info('ROC Curve image not found. Run model_training.py first.')

    st.subheader('Feature Importance')
    try:
        importance_df = pd.read_csv('models/feature_importance.csv')
        fig = px.bar(importance_df.head(15), y='feature', x='importance', orientation='h', title='Top 15 Most Important Features')
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info('Feature importance data not found.')

def show_risk_scorer():
    st.header('Interactive Risk Scorer')
    st.markdown('Enter loan application details to get a risk assessment')

    model, metadata = load_model()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Loan Details')
        loan_amnt = st.number_input('Loan Amount ($)', 1000, 40000, 10000, step=1000)
        int_rate = st.slider('Interest Rate (%)', 5.0, 30.0, 12.0, 0.5)
        term = st.selectbox('Loan Term (months)', [36, 60])
        purpose = st.selectbox('Loan Purpose', ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business', 'other'])

    with col2:
        st.subheader('Borrower Information')
        annual_inc = st.number_input('Annual Income ($)', 10000, 300000, 60000, step=5000)
        fico_score=st.slider('FICO Score', 300, 850, 700)
        dti = st.slider('Debt-to-income Ratio (%)', 0.0, 50.0, 15.0, 0.5)
        emp_length = st.selectbox('Employment Length', ['<1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])

    with col3:
        st.subheader('Credit History')
        delinq_2yrs = st.number_input('Delinquencies (last 2 years)', 0, 10, 0)
        inq_last_6mths = st.number_input('Credit inquiries (last 6 months)', 0, 10, 1)
        open_acc = st.number_input('Open Credit Lines', 0, 50, 10)
        total_acc = st.number_input('Total Credit Lines', 0, 100, 20)
        revol_util = st.slider('Credit Utilization (%)', 0.0, 100.0, 50.0)

    left_space, center_area, right_space = st.columns([1, 2, 1])
    with center_area:
        if st.button('Calculate Risk Score', type='primary'):
            input_data = {
                'loan_amnt': loan_amnt,
                'int_rate': int_rate,
                'annual_inc': annual_inc,
                'dti': dti,
                'fico_score': fico_score,
                'delinq_2yrs': delinq_2yrs,
                'inq_last_6mths': inq_last_6mths,
                'open_acc': open_acc,
                'total_acc': total_acc,
                'revol_util': revol_util,
                'loan_to_income': loan_amnt / annual_inc,
                'credit_utilization': revol_util / 100.0
            }

            st.success('Risk Assessment Complete!')

            mock_probability = 0.15
            risk_score = int(600 + (0.5 - mock_probability) * 400)

            st.metric('Risk Score', risk_score)
            st.metric('Default Probability', f'{mock_probability*100:.2f}%')

            if mock_probability < 0.15:
                st.success('LOW RISK')
                recommendation = 'Recommend Approval'
            elif mock_probability < 0.30:
                st.warning('MEDIUM RISK')
                recommendation = 'Manual Review Required'
            else:
                st.error('HIGH RISK')
                recommendation = 'Recommend Decline'

            st.markdown(f'**Recommendation:** {recommendation}')

def show_portfolio_analytics():
    st.header('Portfolio Analytics')
    engine = get_database_connection()
    st.subheader('Portfolio Overview')

    overview_query = """
    SELECT 
        COUNT(*) as total_loans,
        SUM(defaulted) as total_defaults,
        ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate,
        ROUND(AVG(loan_amnt), 2) as avg_loan_amount,
        ROUND(SUM(loan_amnt), 2) as total_portfolio_value
    FROM loans
    """

    overview = load_data_from_db(overview_query, engine)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric('Total Loans', f'{overview['total_loans'].iloc[0]:,}')
    with col2:
        st.metric('Default Rate', f'{overview['default_rate'].iloc[0]}%')
    with col3:
        st.metric('Avg Loan Size', f'{overview['avg_loan_amount'].iloc[0]:,.0f}')
    with col4:
        st.metric('Portfolio Value', f'${overview['total_portfolio_value'].iloc[0]/1e6:.1f}M')

    st.markdown('---')
    st.subheader('Risk by Loan Grade')

    grade_query = """
    SELECT
        grade,
        COUNT(*) as loan_count,
        ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate,
        ROUND(AVG(int_rate), 2) as avg_rate
    FROM loans
    GROUP BY grade
    ORDER BY grade
    """

    grade_df = load_data_from_db(grade_query, engine)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Default Rate',
        x=grade_df['grade'],
        y=grade_df['default_rate'],
        yaxis='y',
        marker_color='red'
    ))
    fig.add_trace(go.Scatter(
        name='Avg Interest Rate',
        x=grade_df['grade'],
        y=grade_df['avg_rate'],
        yaxis='y2',
        marker_color='Lightgreen',
        mode='lines+markers'
    ))
    fig.update_layout(
        title='Default Rate and Interest Rate by Grade',
        yaxis=dict(title='Default Rate (%)'),
        yaxis2=dict(title='Interest Rate (%)', overlaying='y', side='right'),
        hovermode='x'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader('Loan Purpose Distribution')

    purpose_query = """
    SELECT
        purpose,
        COUNT(*) as loan_count,
        ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate
    FROM loans
    GROUP BY purpose
    HAVING loan_count > 100
    ORDER BY loan_count DESC
    LIMIT 10
    """

    purpose_df = load_data_from_db(purpose_query, engine)

    fig = px.bar(
        purpose_df,
        x='purpose',
        y='loan_count',
        color='default_rate',
        title='Top 10 Loan Purposes by Volume',
        labels={'loan_count': 'Number of Loans', 'default_rate': 'Default Rate (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)

def show_sql_explorer():
    st.header('SQL Query Explorer')
    st.markdown('Execute custom SQL queries on the loan database')
    
    engine = get_database_connection()

    predefined_queries = {
        'Select All (Sample)': 'SELECT * FROM loans LIMIT 100',
        'Default Rate by State': """
            SELECT addr_state, COUNT(*) as loans, ROUND(100.0*SUM(defaulted)/COUNT(*),2) as default_rate
            FROM loans
            GROUP BY addr_state
            HAVING loans > 500
            ORDER BY default_rate DESC
            LIMIT 10
        """,
        'High Risk Loans': """
            SELECT loan_id, loan_amnt, fico_score, dti, grade, defaulted
            FROM loans
            WHERE fico_score < 650 AND dti > 30
            LIMIT 50
        """
    }

    query_option = st.selectbox('Choose a predefined query:', list(predefined_queries))
    custom_query = st.text_area('Or write your own SQL query:', value=predefined_queries[query_option], height=150)

    if st.button('Execute Query'):
        try:
            result = pd.read_sql(custom_query, engine)
            st.success(f'Query executed successfully! Returned {len(result)} rows.')
            st.dataframe(result, use_container_width=True)
            csv = result.to_csv(index=False)
            st.download_button(label='Download Results as CSV', data=csv, file_name='query_results.csv', mime='text/csv')
        except Exception as e:
            st.error(f'Querry error: {str(e)}')

if __name__ == '__main__':
    main()