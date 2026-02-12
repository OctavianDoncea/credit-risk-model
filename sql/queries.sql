USE credit_risk_db;

-- Overall default rate
SELECT
	COUNT(*) as total_loans,
    SUM(defaulted) as total_default,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(loan_amnt), 2) as avg_loan_amount,
    ROUND(AVG(int_rate), 2) as avg_interest_rate
FROM loans;


-- Default rate by loan grade
SELECT
	grade,
    COUNT(*) as loan_amnt,
    SUM(defaulted) as defaults,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(int_rate), 2) as avg_interest_rate,
    ROUND(AVG(loan_amnt), 2) as avg_loan_amount
FROM loans
GROUP BY grade
ORDER BY grade;


-- Default rate by purpose
SELECT
	purpose,
    COUNT(*) as loan_count,
    SUM(defaulted) as defaults,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(loan_amnt), 2) as avg_loan_amnt
FROM loans
GROUP BY purpose
HAVING loan_count > 100
ORDER BY default_rate_pct DESC;


-- Analysis by origination date
SELECT
	DATE_FORMAT(issue_date, '%Y-%m') as cohort_month,
    COUNT(*) as loan_count,
    SUM(defaulted) as defaults,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(loan_amnt), 2) as avg_loan_size,
    ROUND(SUM(CASE WHEN defaulted = 1 THEN loan_amnt ELSE 0 END), 2) as total_loss_amount
FROM loans
WHERE issue_date IS NOT NULL
GROUP BY cohort_month
ORDER BY cohort_month DESC
LIMIT 24;


-- Risk segmentation by FICO score
SELECT 
    CASE 
        WHEN fico_score >= 750 THEN 'Excellent (750+)'
        WHEN fico_score >= 700 THEN 'Good (700-749)'
        WHEN fico_score >= 650 THEN 'Fair (650-699)'
        WHEN fico_score >= 600 THEN 'Poor (600-649)'
        ELSE 'Very Poor (<600)'
    END as credit_tier,
    COUNT(*) as loan_count,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(int_rate), 2) as avg_interest_rate,
    ROUND(AVG(dti), 2) as avg_dti
FROM loans
GROUP BY credit_tier
ORDER BY 
    FIELD(credit_tier, 
          'Excellent (750+)', 
          'Good (700-749)', 
          'Fair (650-699)', 
          'Poor (600-649)', 
          'Very Poor (<600)');
          
          
-- DTI analysis
SELECT
	CASE
		WHEN dti < 10 THEN '0-10%'
        WHEN dti < 20 THEN '10-20%'
        WHEN dti < 30 THEN '20-30%'
        WHEN dti < 40 THEN '30-40%'
        ELSE '+40%'
	END as dti_bucket,
    COUNT(*) as loan_amnt,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(fico_score), 0) as avg_fico
FROM loans
GROUP BY dti_bucket
ORDER BY FIELD(dti_bucket, '0-10%', '10-20%', '20-30%', '30-40%', '+40%');


-- Home ownership analysis
SELECT
	home_ownership,
    COUNT(*) as loan_count,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(annual_inc), 2) as avg_income,
    ROUND(AVG(loan_amnt), 2) as avg_loan_amount
FROM loans
GROUP BY home_ownership
HAVING loan_count > 100
ORDER BY default_rate_pct DESC;


-- Employment length impact
SELECT
	CASE
		WHEN emp_length IS NULL THEN 'Unknown'
        WHEN emp_length = 0 THEN '<1 year'
        WHEN emp_length BETWEEN 1 AND 2 THEN '1-2 years'
        WHEN emp_length BETWEEN 3 AND 5 THEN '3-5 years'
        WHEN emp_length BETWEEN 6 AND 9 THEN '6-9 years'
        ELSE '10+ years'
	END as employment_bucket,
    COUNT(*) as loan_count,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(annual_inc), 2) as avg_income
FROM loans
GROUP BY employment_bucket
ORDER BY default_rate_pct DESC;


-- State-level analysis
SELECT
	addr_state,
	COUNT(*) as loan_count,
    SUM(defaulted) as defaults,
    ROUND(100.0 * SUM(defaulted) / 	COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(loan_amnt), 2) as avg_loan_amount
FROM loans
WHERE addr_state IS NOT NULL
GROUP BY addr_state
HAVING loan_count > 500
ORDER BY loan_count DESC
LIMIT 10;


-- High-risk loan identification
SELECT 
    loan_id,
    loan_amnt,
    int_rate,
    grade,
    fico_score,
    dti,
    delinq_2yrs,
    pub_rec_bankruptcies,
    defaulted
FROM loans
WHERE 
    (fico_score < 650 AND dti > 30)
    OR (delinq_2yrs > 2)
    OR (pub_rec_bankruptcies > 0)
ORDER BY 
    CASE WHEN defaulted = 1 THEN 0 ELSE 1 END,
    fico_score ASC
LIMIT 100;


-- Feature correlation analysis
SELECT
	'DTI vs Default' as metric,
    ROUND(AVG(CASE WHEN defaulted = 1 THEN dti END), 2) as avg_defaulted,
    ROUND(AVG(CASE WHEN defaulted = 0 THEN dti END), 2) as avg_paid,
    ROUND(AVG(CASE WHEN defaulted = 1 THEN dti END) - AVG(CASE WHEN defaulted = 0 THEN dti END), 2) as difference
FROM loans

UNION ALL

SELECT
	'FICO vs Default',
    ROUND(AVG(CASE WHEN defaulted = 1 THEN fico_score END), 2),
    ROUND(AVG(CASE WHEN defaulted = 0 THEN fico_score END), 2),
    ROUND(AVG(CASE WHEN defaulted = 1 THEN fico_score END) - AVG(CASE WHEN defaulted = 0 THEN fico_score END), 2)
FROM loans

UNION ALL

SELECT
	'Income vs Default',
    ROUND(AVG(CASE WHEN defaulted = 1 THEN annual_inc END), 2),
    ROUND(AVG(CASE WHEN defaulted = 0 THEN annual_inc END), 2),
    ROUND(AVG(CASE WHEN defaulted = 1 THEN annual_inc END) - AVG(CASE WHEN defaulted = 0 THEN annual_inc END), 2)
FROM loans;


-- Monthly portfolio metrics
SELECT 
    DATE_FORMAT(issue_date, '%Y-%m') as month,
    COUNT(*) as loans_originated,
    SUM(loan_amnt) as total_funded,
    ROUND(AVG(int_rate), 2) as avg_rate,
    ROUND(100.0 * SUM(defaulted) / COUNT(*), 2) as default_rate_pct
FROM loans
GROUP BY month
ORDER BY month DESC;