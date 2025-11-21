import os
import sqlite3
import pandas as pd
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict, Optional
from pydantic import BaseModel
import uvicorn
from sqlalchemy import create_engine, text

# Import our custom modules
from data_generator import AMLDataGenerator
from entity_resolution import EntityResolver
from ml_model import AnomalyDetector

app = FastAPI(title="AML Neural Monitor API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Config
DB_PATH = "./aml_database.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Global state for caching models
detector = AnomalyDetector()
resolver = EntityResolver()

class AnalysisResponse(BaseModel):
    status: str
    message: str
    metrics: Dict

def init_db():
    """Initialize database schema"""
    with open("schema.sql", "r") as f:
        schema_sql = f.read()
        # Split by semicolon to execute multiple statements
        # Note: SQLite specific adjustments might be needed for complex SQL
        # For this demo, we'll use pandas to write tables which is safer/easier
        pass

def load_data_pipeline():
    """Full pipeline: Generate Data -> Resolve Entities -> Detect Anomalies -> Save to DB"""
    print("🚀 Starting AML Pipeline...")
    
    # 1. Generate Data
    generator = AMLDataGenerator(num_entities=500, num_transactions=2000)
    entities_df = generator.generate_entities()
    transactions_df = generator.generate_transactions()
    
    # 2. Entity Resolution
    print("🔍 Running Entity Resolution...")
    entities_df = resolver.resolve_entities(entities_df, transactions_df)
    
    # 3. ML Anomaly Detection
    print("🧠 Running ML Anomaly Detection...")
    features_df = detector.prepare_features(entities_df, transactions_df)
    detector.train_models(features_df)
    anomaly_scores = detector.detect_anomalies(features_df)
    
    # Merge scores back to entities
    entities_df = pd.merge(entities_df, anomaly_scores[['entity_id', 'ensemble_score', 'is_anomaly']], on='entity_id', how='left')
    
    # 4. Pattern Detection
    patterns = detector.detect_patterns(entities_df, transactions_df, anomaly_scores)
    
    # 5. Save to Database
    print("💾 Saving to Database...")
    entities_df.to_sql('entities', engine, if_exists='replace', index=False)
    transactions_df.to_sql('transactions', engine, if_exists='replace', index=False)
    anomaly_scores.to_sql('model_scores', engine, if_exists='replace', index=False)
    
    # Save patterns as JSON string in a simpler table or file for this demo
    patterns_df = pd.DataFrame(patterns)
    patterns_df.to_sql('anomalies', engine, if_exists='replace', index=False)
    
    print("✅ Pipeline Complete")

@app.on_event("startup")
async def startup_event():
    """Run pipeline on startup if DB doesn't exist"""
    if not os.path.exists(DB_PATH):
        load_data_pipeline()
    else:
        # Load pre-trained models if available
        try:
            detector.load_models()
        except:
            pass

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"status": "active", "system": "AML Neural Monitor"}

@app.post("/api/run_analysis")
async def trigger_analysis(background_tasks: BackgroundTasks):
    """Trigger a full re-run of the analysis pipeline"""
    background_tasks.add_task(load_data_pipeline)
    return {"status": "processing", "message": "Pipeline started in background"}

@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics from real DB data"""
    try:
        with engine.connect() as conn:
            t_count = pd.read_sql("SELECT COUNT(*) as c FROM transactions", conn).iloc[0]['c']
            e_count = pd.read_sql("SELECT COUNT(*) as c FROM entities", conn).iloc[0]['c']
            a_count = pd.read_sql("SELECT COUNT(*) as c FROM anomalies", conn).iloc[0]['c']
            
            # Calculate risk level
            avg_risk = pd.read_sql("SELECT AVG(ensemble_score) as r FROM model_scores", conn).iloc[0]['r']
            risk_level = "CRITICAL" if avg_risk > 0.7 else "HIGH" if avg_risk > 0.4 else "LOW"
            
            return {
                "transaction_count": int(t_count),
                "entity_count": int(e_count),
                "anomaly_count": int(a_count),
                "risk_level": risk_level,
                "avg_risk_score": float(avg_risk)
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/anomalies")
async def get_anomalies():
    """Get detected anomalies"""
    try:
        query = "SELECT * FROM anomalies ORDER BY risk_score DESC LIMIT 50"
        df = pd.read_sql(query, engine)
        
        # Convert JSON strings back to objects if necessary, or just return dict
        anomalies = []
        for _, row in df.iterrows():
            # Normalize data structure for frontend
            anomalies.append({
                "id": row.get('pattern_id', 'UNK'),
                "type": row.get('pattern_type', 'Unknown').replace('_', ' ').title(),
                "severity": row.get('severity', 'Medium').upper(),
                "riskScore": float(row.get('risk_score', 0)),
                "totalAmount": float(row.get('total_amount', 0)),
                "hops": int(row.get('hop_count', 0)),
                "description": row.get('description', ''),
                "entities": eval(row['entities']) if isinstance(row['entities'], str) else row['entities'],
                # Mock indicators if not in DB, or extract from logic
                "indicators": [f"Risk Score: {row.get('risk_score', 0):.2f}", "Pattern Detected via Graph Scan"]
            })
        return anomalies
    except Exception as e:
        print(e)
        return []

@app.get("/api/graph/{case_id}")
async def get_graph_data(case_id: str):
    """Get graph nodes and edges for a specific case"""
    try:
        # 1. Get entities involved in the case
        # This logic assumes case_id maps to pattern_id
        anomalies_df = pd.read_sql(f"SELECT * FROM anomalies WHERE pattern_id = '{case_id}'", engine)
        if anomalies_df.empty:
            raise HTTPException(status_code=404, detail="Case not found")
            
        entity_ids = eval(anomalies_df.iloc[0]['entities']) if isinstance(anomalies_df.iloc[0]['entities'], str) else anomalies_df.iloc[0]['entities']
        
        # 2. Fetch entity details
        placeholders = ','.join(['?'] * len(entity_ids))
        entities_query = f"SELECT * FROM entities WHERE entity_id IN ({placeholders})"
        entities_df = pd.read_sql(entities_query, engine, params=entity_ids)
        
        nodes = []
        for _, row in entities_df.iterrows():
            nodes.append({
                "id": row['entity_id'],
                "name": row['name'],
                "type": row['type'],
                "risk": float(row.get('risk_score', 0)),
                "country": row.get('country_code', 'UNK')
            })
            
        # 3. Fetch transactions between these entities
        tx_query = f"""
            SELECT * FROM transactions 
            WHERE from_entity_id IN ({placeholders}) 
            AND to_entity_id IN ({placeholders})
        """
        # We need to pass the params twice because we used placeholders twice
        tx_df = pd.read_sql(tx_query, engine, params=entity_ids + entity_ids)
        
        links = []
        for _, row in tx_df.iterrows():
            links.append({
                "from": row['from_entity_id'],
                "to": row['to_entity_id'],
                "amount": float(row['amount']),
                "timestamp": str(row['timestamp']),
                "suspicious": bool(row['suspicious'])
            })
            
        return {"entities": nodes, "transactions": links}
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)