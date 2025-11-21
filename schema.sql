-- AML Neural Monitor Database Schema
-- Supports PostgreSQL, MySQL, and SQLite

-- Drop existing tables (for clean setup)
DROP TABLE IF EXISTS model_scores CASCADE;
DROP TABLE IF EXISTS anomalies CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS entities CASCADE;
DROP TABLE IF EXISTS entity_clusters CASCADE;

-- 1. Entities Table
CREATE TABLE entities (
    entity_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) CHECK (type IN ('personal', 'business', 'shell', 'unknown')),
    risk_score DECIMAL(3,2) CHECK (risk_score >= 0 AND risk_score <= 1),
    country_code VARCHAR(3),
    kyc_score DECIMAL(3,2) CHECK (kyc_score >= 0 AND kyc_score <= 1),
    account_age_days INTEGER,
    total_transaction_count INTEGER DEFAULT 0,
    total_transaction_volume DECIMAL(15,2) DEFAULT 0,
    suspicious_flag_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional entity resolution fields
    phone_hash VARCHAR(64),
    email_domain VARCHAR(255),
    address_hash VARCHAR(64),
    tax_id_hash VARCHAR(64),
    device_fingerprint VARCHAR(64),
    cluster_id VARCHAR(50),
    
    -- Graph features
    degree_centrality DECIMAL(5,4),
    betweenness_centrality DECIMAL(5,4),
    pagerank_score DECIMAL(5,4),
    community_id INTEGER
);

-- 2. Transactions Table
CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    from_entity_id VARCHAR(50) NOT NULL,
    to_entity_id VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    transaction_type VARCHAR(50),
    payment_method VARCHAR(50),
    timestamp TIMESTAMP NOT NULL,
    
    -- Location data
    origin_country VARCHAR(3),
    destination_country VARCHAR(3),
    cross_border BOOLEAN DEFAULT FALSE,
    
    -- Risk indicators
    suspicious BOOLEAN DEFAULT FALSE,
    flagged_by_rules BOOLEAN DEFAULT FALSE,
    risk_score DECIMAL(3,2),
    
    -- Additional features
    is_round_amount BOOLEAN DEFAULT FALSE,
    time_since_last_transaction INTEGER, -- in seconds
    velocity_flag BOOLEAN DEFAULT FALSE,
    structuring_flag BOOLEAN DEFAULT FALSE,
    
    -- Graph features
    edge_weight DECIMAL(5,4),
    path_length_from_source INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (from_entity_id) REFERENCES entities(entity_id),
    FOREIGN KEY (to_entity_id) REFERENCES entities(entity_id)
);

-- 3. Entity Clusters Table (for entity resolution results)
CREATE TABLE entity_clusters (
    cluster_id VARCHAR(50) PRIMARY KEY,
    master_entity_id VARCHAR(50),
    cluster_size INTEGER,
    cluster_risk_score DECIMAL(3,2),
    resolution_confidence DECIMAL(3,2),
    resolution_method VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (master_entity_id) REFERENCES entities(entity_id)
);

-- 4. Anomalies Table
CREATE TABLE anomalies (
    anomaly_id VARCHAR(50) PRIMARY KEY,
    anomaly_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    risk_score DECIMAL(3,2) NOT NULL,
    confidence_score DECIMAL(3,2),
    
    -- Detection metadata
    detection_method VARCHAR(100),
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(20),
    
    -- Involved entities (stored as JSON array in string)
    involved_entities TEXT,
    involved_transactions TEXT,
    
    -- Pattern details
    pattern_name VARCHAR(100),
    pattern_description TEXT,
    total_amount DECIMAL(15,2),
    transaction_count INTEGER,
    hop_count INTEGER,
    time_span_hours INTEGER,
    
    -- ML model outputs
    isolation_forest_score DECIMAL(5,4),
    autoencoder_score DECIMAL(5,4),
    graph_anomaly_score DECIMAL(5,4),
    ensemble_score DECIMAL(5,4),
    
    -- SHAP feature importance (stored as JSON)
    feature_importance TEXT,
    
    -- Investigation status
    status VARCHAR(50) DEFAULT 'pending',
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP,
    sar_filed BOOLEAN DEFAULT FALSE,
    sar_reference VARCHAR(100),
    notes TEXT
);

-- 5. Model Scores Table (for ML predictions)
CREATE TABLE model_scores (
    score_id SERIAL PRIMARY KEY,
    entity_id VARCHAR(50),
    transaction_id VARCHAR(50),
    model_name VARCHAR(100),
    model_version VARCHAR(20),
    score_value DECIMAL(5,4),
    score_type VARCHAR(50), -- 'anomaly', 'risk', 'similarity'
    features_used TEXT, -- JSON array of feature names
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id),
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

-- Create indexes for performance
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_transactions_from_entity ON transactions(from_entity_id);
CREATE INDEX idx_transactions_to_entity ON transactions(to_entity_id);
CREATE INDEX idx_transactions_suspicious ON transactions(suspicious);
CREATE INDEX idx_entities_risk_score ON entities(risk_score);
CREATE INDEX idx_entities_cluster ON entities(cluster_id);
CREATE INDEX idx_anomalies_severity ON anomalies(severity);
CREATE INDEX idx_anomalies_detection_timestamp ON anomalies(detection_timestamp);
CREATE INDEX idx_model_scores_entity ON model_scores(entity_id);
CREATE INDEX idx_model_scores_timestamp ON model_scores(prediction_timestamp);

-- Sample data insertion
INSERT INTO entities (entity_id, name, type, risk_score, country_code, kyc_score, account_age_days, cluster_id) VALUES
('E001', 'John Smith Personal', 'personal', 0.15, 'US', 0.92, 730, 'C001'),
('E002', 'SHELLCO_HOLDINGS_LTD', 'shell', 0.89, 'KY', 0.12, 45, 'C002'),
('E003', 'Maria Garcia', 'personal', 0.22, 'MX', 0.78, 365, 'C003'),
('E004', 'OFFSHORE_ALPHA_CORP', 'shell', 0.94, 'PA', 0.08, 30, 'C002'),
('E005', 'TechStart Inc', 'business', 0.35, 'US', 0.88, 1095, 'C004'),
('E006', 'CRYPTO_EXCHANGE_XYZ', 'business', 0.68, 'MT', 0.45, 180, 'C005'),
('E007', 'Anonymous Wallet 7B3X', 'unknown', 0.76, 'RU', 0.00, 90, 'C006');

INSERT INTO transactions (transaction_id, from_entity_id, to_entity_id, amount, timestamp, suspicious, risk_score, cross_border) VALUES
('T001', 'E001', 'E002', 49999.99, '2024-01-15 09:30:00', TRUE, 0.87, TRUE),
('T002', 'E002', 'E004', 48500.00, '2024-01-16 14:22:00', TRUE, 0.91, TRUE),
('T003', 'E004', 'E006', 47000.00, '2024-01-17 11:45:00', TRUE, 0.93, TRUE),
('T004', 'E003', 'E002', 9900.00, '2024-01-18 16:10:00', TRUE, 0.76, TRUE),
('T005', 'E005', 'E006', 15000.00, '2024-01-19 10:05:00', FALSE, 0.23, FALSE),
('T006', 'E006', 'E007', 25000.00, '2024-01-20 13:30:00', TRUE, 0.82, TRUE);

INSERT INTO anomalies (anomaly_id, anomaly_type, severity, risk_score, confidence_score, pattern_name, total_amount, transaction_count, hop_count) VALUES
('A001', 'Complex Layering Scheme', 'critical', 0.93, 0.87, 'Multi-hop Shell Company Layering', 145499.99, 3, 3),
('A002', 'Smurfing Pattern', 'high', 0.78, 0.92, 'Structured Deposits Below Threshold', 34900.00, 2, 2);