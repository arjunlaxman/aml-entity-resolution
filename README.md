# 🧠 AML Neural Monitor: End-to-End Machine Learning System

This is a production-ready Anti-Money Laundering (AML) detection system. It features a complete data pipeline: synthetic data generation, SQL storage, Entity Resolution, ML Anomaly Detection, and a FastAPI backend serving a real-time dashboard.

## 🏗 Architecture

1.  **Data Layer**:
    * `data_generator.py`: Generates synthetic transactions with embedded money laundering patterns (Smurfing, Layering, Round-Tripping).
    * `schema.sql`: SQLite/PostgreSQL schema for storing entities, transactions, and scores.
2.  **Intelligence Layer**:
    * `entity_resolution.py`: Resolves multiple accounts to single identities using fuzzy matching and graph clustering.
    * `ml_model.py`: Detects anomalies using **Isolation Forest**, **Autoencoders**, and Rule-based graph traversals.
3.  **Application Layer**:
    * `main.py`: **FastAPI** server providing REST endpoints.
    * `index.html`: Frontend dashboard consuming live API data.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
