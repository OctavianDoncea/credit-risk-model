CREATE DATABASE IF NOT EXISTS credit_risk_db;
USE credit_risk_db;
SHOW DATABASES;

DROP TABLE IF EXISTS loans;

CREATE TABLE loans (
	-- Primary key
    loan_id BIGINT PRIMARY KEY,
    
    -- Loan details
    loan_amnt DECIMAL(10, 2),
    funded_amnt DECIMAL(10, 2),
    term INT,
    int_rate DECIMAL(5, 2),
    grade VARCHAR(1),
    sub_grade VARCHAR(2),
    
    -- Borrower information
    emp_length INT,
    home_ownership VARCHAR(20),
    annual_inc DECIMAL(12, 2),
    verification_status VARCHAR(20),
    
    -- Loan metadata
    issue_date DATE,
    loan_status VARCHAR(30),
    purpose VARCHAR(50),
    addr_state VARCHAR(2),
    
    -- Credit metrics
    dti DECIMAL(5, 2),
    delinq_2yrs INT,
    fico_score DECIMAL(5, 2),
    inq_last_6mths INT,
    open_acc INT,
    pub_rec INT,
    revol_bal DECIMAL(12, 2),
    revol_util DECIMAL(5, 2),
    total_acc INT,
    mort_acc INT,
    pub_rec_bankruptcies INT,
    
    loan_to_income DECIMAL(5, 3),
    defaulted TINYINT(1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for performance
    INDEX idx_grade (grade),
    INDEX idx_defaulted (defaulted),
    INDEX idx_issue_date (issue_date),
    INDEX idx_purpose (purpose),
    INDEX idx_fico (fico_score)
) ENGINE=InnoDB, DEFAULT CHARSET=utf8mb4;

SHOW TABLES;