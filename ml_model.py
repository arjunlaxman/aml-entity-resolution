"""
Machine Learning Anomaly Detection Models
Implements multiple ML algorithms for AML detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import networkx as nx
import joblib
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self, models_dir: str = "./models"):
        """
        Initialize anomaly detection system with multiple models
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.threshold_scores = {
            'isolation_forest': 0.6,
            'lof': 0.65,
            'autoencoder': 0.7,
            'ensemble': 0.65
        }
        
    def prepare_features(self, entities_df: pd.DataFrame, 
                        transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML models
        """
        print("Engineering features for ML models...")
        
        features_list = []
        
        for _, entity in entities_df.iterrows():
            entity_id = entity['entity_id']
            
            # Get entity's transactions
            sent_txs = transactions_df[transactions_df['from_entity_id'] == entity_id]
            received_txs = transactions_df[transactions_df['to_entity_id'] == entity_id]
            all_txs = pd.concat([sent_txs, received_txs])
            
            # Transaction-based features
            features = {
                'entity_id': entity_id,
                
                # Basic entity features
                'risk_score': entity['risk_score'],
                'kyc_score': entity['kyc_score'],
                'account_age_days': entity['account_age_days'],
                'is_shell': 1 if entity['type'] == 'shell' else 0,
                'is_unknown': 1 if entity['type'] == 'unknown' else 0,
                'is_high_risk_country': 1 if entity['country_code'] in ['KY', 'PA', 'VG', 'MT', 'RU'] else 0,
                
                # Transaction volume features
                'total_sent_amount': sent_txs['amount'].sum(),
                'total_received_amount': received_txs['amount'].sum(),
                'avg_transaction_amount': all_txs['amount'].mean() if len(all_txs) > 0 else 0,
                'std_transaction_amount': all_txs['amount'].std() if len(all_txs) > 1 else 0,
                'max_transaction_amount': all_txs['amount'].max() if len(all_txs) > 0 else 0,
                
                # Transaction count features
                'num_sent_transactions': len(sent_txs),
                'num_received_transactions': len(received_txs),
                'total_transactions': len(all_txs),
                
                # Velocity features
                'transactions_per_day': self._calculate_velocity(all_txs),
                'rapid_movement_flag': self._detect_rapid_movement(all_txs),
                
                # Network features
                'unique_counterparties': len(set(sent_txs['to_entity_id'].tolist() + 
                                               received_txs['from_entity_id'].tolist())),
                'degree_centrality': entity.get('degree_centrality', 0),
                'betweenness_centrality': entity.get('betweenness_centrality', 0),
                'pagerank_score': entity.get('pagerank_score', 0),
                
                # Pattern features
                'round_amount_ratio': self._calculate_round_amount_ratio(all_txs),
                'cross_border_ratio': all_txs['cross_border'].mean() if 'cross_border' in all_txs.columns and len(all_txs) > 0 else 0,
                'suspicious_ratio': all_txs['suspicious'].mean() if 'suspicious' in all_txs.columns and len(all_txs) > 0 else 0,
                
                # Time-based features
                'night_transaction_ratio': self._calculate_night_ratio(all_txs),
                'weekend_transaction_ratio': self._calculate_weekend_ratio(all_txs),
                
                # Structuring detection
                'below_threshold_ratio': self._calculate_below_threshold_ratio(all_txs),
                'similar_amount_ratio': self._calculate_similar_amount_ratio(all_txs),
                
                # Flow features
                'in_out_ratio': (received_txs['amount'].sum() / sent_txs['amount'].sum()) if sent_txs['amount'].sum() > 0 else 0,
                'flow_through_score': self._calculate_flow_through_score(sent_txs, received_txs)
            }
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Handle infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        # Store feature columns for later use
        self.feature_columns = [col for col in features_df.columns if col != 'entity_id']
        
        print(f"Generated {len(self.feature_columns)} features for {len(features_df)} entities")
        
        return features_df
    
    def train_models(self, features_df: pd.DataFrame, labels: pd.Series = None):
        """
        Train multiple anomaly detection models
        """
        print("Training anomaly detection models...")
        
        # Prepare feature matrix
        X = features_df[self.feature_columns].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['standard'] = scaler
        
        # 1. Isolation Forest
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        iso_forest.fit(X_scaled)
        self.models['isolation_forest'] = iso_forest
        
        # 2. Local Outlier Factor
        print("Training Local Outlier Factor...")
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        lof.fit(X_scaled)
        self.models['lof'] = lof
        
        # 3. Autoencoder
        print("Training Autoencoder...")
        autoencoder = self._build_autoencoder(X_scaled)
        self.models['autoencoder'] = autoencoder
        
        # 4. If labels provided, train supervised model
        if labels is not None:
            print("Training Random Forest (supervised)...")
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            rf.fit(X_scaled, labels)
            self.models['random_forest'] = rf
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))
        
        # Save models
        self._save_models()
        
        print("Model training complete!")
    
    def _build_autoencoder(self, X: np.ndarray) -> MLPRegressor:
        """
        Build and train an autoencoder for anomaly detection
        """
        # Simple autoencoder using MLPRegressor
        encoding_dim = max(2, X.shape[1] // 3)
        
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(encoding_dim,),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        # Train to reconstruct input
        autoencoder.fit(X, X)
        
        return autoencoder
    
    def detect_anomalies(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using trained models
        """
        print("Detecting anomalies...")
        
        # Prepare features
        X = features_df[self.feature_columns].values
        X_scaled = self.scalers['standard'].transform(X)
        
        results_df = features_df[['entity_id']].copy()
        
        # 1. Isolation Forest scores
        if 'isolation_forest' in self.models:
            iso_scores = self.models['isolation_forest'].score_samples(X_scaled)
            # Normalize to 0-1 range
            iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
            # Invert so higher score means more anomalous
            results_df['isolation_forest_score'] = 1 - iso_scores_norm
        
        # 2. LOF scores
        if 'lof' in self.models:
            lof_scores = self.models['lof'].score_samples(X_scaled)
            lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
            results_df['lof_score'] = 1 - lof_scores_norm
        
        # 3. Autoencoder reconstruction error
        if 'autoencoder' in self.models:
            reconstructed = self.models['autoencoder'].predict(X_scaled)
            reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
            # Normalize
            error_norm = (reconstruction_error - reconstruction_error.min()) / (reconstruction_error.max() - reconstruction_error.min())
            results_df['autoencoder_score'] = error_norm
        
        # 4. Ensemble score (average of all methods)
        score_columns = [col for col in results_df.columns if col.endswith('_score')]
        results_df['ensemble_score'] = results_df[score_columns].mean(axis=1)
        
        # 5. Determine if anomaly based on thresholds
        results_df['is_anomaly'] = results_df['ensemble_score'] > self.threshold_scores['ensemble']
        results_df['anomaly_confidence'] = results_df['ensemble_score']
        
        # Rank anomalies
        results_df['anomaly_rank'] = results_df['ensemble_score'].rank(ascending=False, method='dense')
        
        print(f"Detected {results_df['is_anomaly'].sum()} anomalies out of {len(results_df)} entities")
        
        return results_df
    
    def detect_patterns(self, entities_df: pd.DataFrame, 
                       transactions_df: pd.DataFrame,
                       anomaly_scores_df: pd.DataFrame) -> List[Dict]:
        """
        Detect specific money laundering patterns
        """
        print("Detecting specific ML patterns...")
        
        patterns = []
        
        # 1. Layering Detection
        layering_patterns = self._detect_layering(transactions_df, anomaly_scores_df)
        patterns.extend(layering_patterns)
        
        # 2. Smurfing Detection
        smurfing_patterns = self._detect_smurfing(transactions_df, anomaly_scores_df)
        patterns.extend(smurfing_patterns)
        
        # 3. Round-tripping Detection
        roundtrip_patterns = self._detect_round_tripping(transactions_df, anomaly_scores_df)
        patterns.extend(roundtrip_patterns)
        
        # 4. Rapid Movement Detection
        rapid_patterns = self._detect_rapid_movement_patterns(transactions_df, anomaly_scores_df)
        patterns.extend(rapid_patterns)
        
        print(f"Detected {len(patterns)} suspicious patterns")
        
        return patterns
    
    def _detect_layering(self, transactions_df: pd.DataFrame, 
                        anomaly_scores_df: pd.DataFrame) -> List[Dict]:
        """
        Detect layering patterns (multiple hops through shell companies)
        """
        patterns = []
        
        # Build transaction graph
        G = nx.DiGraph()
        for _, tx in transactions_df.iterrows():
            G.add_edge(tx['from_entity_id'], tx['to_entity_id'],
                      weight=tx['amount'], transaction_id=tx['transaction_id'],
                      timestamp=tx['timestamp'])
        
        # Find paths of length 3+ with decreasing amounts
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(G, source, target, cutoff=5))
                        for path in paths:
                            if len(path) >= 3:
                                # Check if amounts decrease along path
                                amounts = []
                                transactions = []
                                
                                for i in range(len(path) - 1):
                                    edge_data = G.get_edge_data(path[i], path[i+1])
                                    if edge_data:
                                        amounts.append(edge_data['weight'])
                                        transactions.append(edge_data['transaction_id'])
                                
                                if len(amounts) >= 2:
                                    # Check for decreasing pattern (money laundering fee)
                                    decreasing = all(amounts[i] >= amounts[i+1] * 0.95 for i in range(len(amounts)-1))
                                    
                                    if decreasing and amounts[0] > 10000:
                                        # Get risk scores
                                        path_risk = anomaly_scores_df[
                                            anomaly_scores_df['entity_id'].isin(path)
                                        ]['ensemble_score'].mean()
                                        
                                        if path_risk > 0.5:
                                            patterns.append({
                                                'pattern_id': f"LAY_{len(patterns)+1:04d}",
                                                'pattern_type': 'layering',
                                                'severity': 'critical' if path_risk > 0.8 else 'high',
                                                'risk_score': path_risk,
                                                'confidence': 0.85,
                                                'entities': path,
                                                'transactions': transactions,
                                                'total_amount': amounts[0],
                                                'hop_count': len(path) - 1,
                                                'amount_decay': (amounts[0] - amounts[-1]) / amounts[0] if amounts[0] > 0 else 0,
                                                'description': f"Layering pattern through {len(path)-1} hops with {((amounts[0] - amounts[-1]) / amounts[0] * 100):.1f}% value decay"
                                            })
                    except:
                        continue
        
        return patterns[:10]  # Limit to top 10 patterns
    
    def _detect_smurfing(self, transactions_df: pd.DataFrame, 
                        anomaly_scores_df: pd.DataFrame) -> List[Dict]:
        """
        Detect smurfing/structuring patterns
        """
        patterns = []
        
        # Group transactions by receiving entity
        for entity_id, group in transactions_df.groupby('to_entity_id'):
            # Look for multiple deposits below threshold
            threshold = 10000
            below_threshold = group[group['amount'] < threshold]
            
            if len(below_threshold) >= 3:
                # Check if amounts are suspiciously similar
                amounts = below_threshold['amount'].values
                mean_amount = amounts.mean()
                std_amount = amounts.std()
                
                if std_amount < mean_amount * 0.1 and mean_amount > 8000:
                    # Suspicious structuring detected
                    entities = list(set(below_threshold['from_entity_id'].tolist() + [entity_id]))
                    
                    # Get risk score
                    entity_risk = anomaly_scores_df[
                        anomaly_scores_df['entity_id'].isin(entities)
                    ]['ensemble_score'].mean()
                    
                    if entity_risk > 0.4:
                        patterns.append({
                            'pattern_id': f"SMF_{len(patterns)+1:04d}",
                            'pattern_type': 'smurfing',
                            'severity': 'high' if entity_risk > 0.7 else 'medium',
                            'risk_score': entity_risk,
                            'confidence': 0.90,
                            'entities': entities,
                            'transactions': below_threshold['transaction_id'].tolist(),
                            'total_amount': amounts.sum(),
                            'hop_count': 1,
                            'num_deposits': len(below_threshold),
                            'avg_deposit': mean_amount,
                            'description': f"Smurfing pattern: {len(below_threshold)} deposits averaging ${mean_amount:.2f} below reporting threshold"
                        })
        
        return patterns[:10]
    
    def _detect_round_tripping(self, transactions_df: pd.DataFrame, 
                              anomaly_scores_df: pd.DataFrame) -> List[Dict]:
        """
        Detect round-tripping patterns (money returns to source)
        """
        patterns = []
        
        # Build transaction graph
        G = nx.DiGraph()
        for _, tx in transactions_df.iterrows():
            G.add_edge(tx['from_entity_id'], tx['to_entity_id'],
                      weight=tx['amount'], transaction_id=tx['transaction_id'])
        
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(G))
            
            for cycle in cycles[:20]:  # Limit processing
                if 3 <= len(cycle) <= 6:  # Reasonable cycle length
                    # Calculate total flow through cycle
                    total_amount = 0
                    transactions = []
                    
                    for i in range(len(cycle)):
                        from_node = cycle[i]
                        to_node = cycle[(i+1) % len(cycle)]
                        edge_data = G.get_edge_data(from_node, to_node)
                        
                        if edge_data:
                            total_amount += edge_data['weight']
                            transactions.append(edge_data['transaction_id'])
                    
                    if total_amount > 20000:  # Significant amount
                        # Get risk score
                        cycle_risk = anomaly_scores_df[
                            anomaly_scores_df['entity_id'].isin(cycle)
                        ]['ensemble_score'].mean()
                        
                        if cycle_risk > 0.5:
                            patterns.append({
                                'pattern_id': f"RND_{len(patterns)+1:04d}",
                                'pattern_type': 'round_tripping',
                                'severity': 'high',
                                'risk_score': cycle_risk,
                                'confidence': 0.88,
                                'entities': cycle,
                                'transactions': transactions,
                                'total_amount': total_amount,
                                'hop_count': len(cycle),
                                'description': f"Round-tripping detected through {len(cycle)} entities with ${total_amount:,.2f} total flow"
                            })
        except:
            pass
        
        return patterns[:5]
    
    def _detect_rapid_movement_patterns(self, transactions_df: pd.DataFrame, 
                                       anomaly_scores_df: pd.DataFrame) -> List[Dict]:
        """
        Detect rapid fund movement patterns
        """
        patterns = []
        
        # Convert timestamp to datetime if needed
        if isinstance(transactions_df['timestamp'].iloc[0], str):
            transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
        
        # Look for rapid sequences
        for entity_id in transactions_df['from_entity_id'].unique():
            entity_txs = transactions_df[transactions_df['from_entity_id'] == entity_id].sort_values('timestamp')
            
            if len(entity_txs) >= 3:
                # Calculate time between transactions
                time_diffs = entity_txs['timestamp'].diff().dt.total_seconds() / 3600  # Hours
                
                # Find rapid sequences (transactions within 1 hour)
                rapid_mask = time_diffs < 1
                rapid_count = rapid_mask.sum()
                
                if rapid_count >= 2:
                    rapid_txs = entity_txs[rapid_mask | rapid_mask.shift(-1)]
                    
                    if len(rapid_txs) > 0 and rapid_txs['amount'].sum() > 15000:
                        entities = list(set(rapid_txs['from_entity_id'].tolist() + 
                                          rapid_txs['to_entity_id'].tolist()))
                        
                        # Get risk score
                        entity_risk = anomaly_scores_df[
                            anomaly_scores_df['entity_id'].isin(entities)
                        ]['ensemble_score'].mean()
                        
                        if entity_risk > 0.5:
                            patterns.append({
                                'pattern_id': f"RPD_{len(patterns)+1:04d}",
                                'pattern_type': 'rapid_movement',
                                'severity': 'high' if rapid_txs['amount'].sum() > 50000 else 'medium',
                                'risk_score': entity_risk,
                                'confidence': 0.82,
                                'entities': entities,
                                'transactions': rapid_txs['transaction_id'].tolist(),
                                'total_amount': rapid_txs['amount'].sum(),
                                'hop_count': len(rapid_txs),
                                'time_span_minutes': int((rapid_txs['timestamp'].max() - rapid_txs['timestamp'].min()).total_seconds() / 60),
                                'description': f"Rapid fund movement: {len(rapid_txs)} transactions totaling ${rapid_txs['amount'].sum():,.2f} within {int(time_diffs[rapid_mask].sum())} hours"
                            })
        
        return patterns[:10]
    
    # Helper methods for feature engineering
    def _calculate_velocity(self, txs: pd.DataFrame) -> float:
        if len(txs) < 2:
            return 0
        
        if isinstance(txs['timestamp'].iloc[0], str):
            txs['timestamp'] = pd.to_datetime(txs['timestamp'])
        
        time_span = (txs['timestamp'].max() - txs['timestamp'].min()).days
        if time_span > 0:
            return len(txs) / time_span
        return len(txs)
    
    def _detect_rapid_movement(self, txs: pd.DataFrame) -> int:
        if len(txs) < 2:
            return 0
        
        if isinstance(txs['timestamp'].iloc[0], str):
            txs = txs.copy()
            txs['timestamp'] = pd.to_datetime(txs['timestamp'])
        
        txs = txs.sort_values('timestamp')
        time_diffs = txs['timestamp'].diff().dt.total_seconds() / 3600  # Hours
        
        # Flag if multiple transactions within 1 hour
        return int((time_diffs < 1).sum() > 1)
    
    def _calculate_round_amount_ratio(self, txs: pd.DataFrame) -> float:
        if len(txs) == 0:
            return 0
        
        round_amounts = txs['amount'].apply(lambda x: x % 1000 == 0)
        return round_amounts.mean()
    
    def _calculate_night_ratio(self, txs: pd.DataFrame) -> float:
        if len(txs) == 0:
            return 0
        
        if isinstance(txs['timestamp'].iloc[0], str):
            txs = txs.copy()
            txs['timestamp'] = pd.to_datetime(txs['timestamp'])
        
        hours = txs['timestamp'].dt.hour
        night_txs = ((hours >= 22) | (hours <= 6)).sum()
        return night_txs / len(txs)
    
    def _calculate_weekend_ratio(self, txs: pd.DataFrame) -> float:
        if len(txs) == 0:
            return 0
        
        if isinstance(txs['timestamp'].iloc[0], str):
            txs = txs.copy()
            txs['timestamp'] = pd.to_datetime(txs['timestamp'])
        
        weekday = txs['timestamp'].dt.dayofweek
        weekend_txs = (weekday >= 5).sum()
        return weekend_txs / len(txs)
    
    def _calculate_below_threshold_ratio(self, txs: pd.DataFrame) -> float:
        if len(txs) == 0:
            return 0
        
        threshold = 10000
        below = (txs['amount'] < threshold) & (txs['amount'] > threshold * 0.8)
        return below.mean()
    
    def _calculate_similar_amount_ratio(self, txs: pd.DataFrame) -> float:
        if len(txs) < 2:
            return 0
        
        amounts = txs['amount'].values
        # Check how many amounts are within 5% of each other
        similar_count = 0
        
        for i in range(len(amounts)):
            for j in range(i+1, len(amounts)):
                if abs(amounts[i] - amounts[j]) / max(amounts[i], amounts[j]) < 0.05:
                    similar_count += 1
        
        total_pairs = len(amounts) * (len(amounts) - 1) / 2
        return similar_count / total_pairs if total_pairs > 0 else 0
    
    def _calculate_flow_through_score(self, sent_txs: pd.DataFrame, 
                                     received_txs: pd.DataFrame) -> float:
        """
        Calculate how much money flows through without staying
        """
        if len(sent_txs) == 0 or len(received_txs) == 0:
            return 0
        
        total_received = received_txs['amount'].sum()
        total_sent = sent_txs['amount'].sum()
        
        if total_received > 0:
            flow_through = min(total_sent / total_received, 1.0)
            
            # Check timing - flow-through happens quickly
            if isinstance(received_txs['timestamp'].iloc[0], str):
                received_txs = received_txs.copy()
                sent_txs = sent_txs.copy()
                received_txs['timestamp'] = pd.to_datetime(received_txs['timestamp'])
                sent_txs['timestamp'] = pd.to_datetime(sent_txs['timestamp'])
            
            avg_received_time = received_txs['timestamp'].mean()
            avg_sent_time = sent_txs['timestamp'].mean()
            
            if avg_sent_time > avg_received_time:
                time_diff = (avg_sent_time - avg_received_time).days
                if time_diff < 7:  # Money sent within a week
                    return flow_through * 1.5  # Boost score for quick flow-through
            
            return flow_through
        
        return 0
    
    def _save_models(self):
        """
        Save trained models to disk
        """
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            joblib.dump(model, f"{self.models_dir}/{model_name}.pkl")
        
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{self.models_dir}/scaler_{scaler_name}.pkl")
        
        # Save feature columns
        with open(f"{self.models_dir}/feature_columns.json", 'w') as f:
            json.dump(self.feature_columns, f)
        
        print(f"Models saved to {self.models_dir}/")
    
    def load_models(self):
        """
        Load pre-trained models from disk
        """
        import os
        
        if not os.path.exists(self.models_dir):
            raise ValueError(f"Models directory {self.models_dir} not found")
        
        # Load models
        for model_file in os.listdir(self.models_dir):
            if model_file.endswith('.pkl'):
                if model_file.startswith('scaler_'):
                    scaler_name = model_file.replace('scaler_', '').replace('.pkl', '')
                    self.scalers[scaler_name] = joblib.load(f"{self.models_dir}/{model_file}")
                else:
                    model_name = model_file.replace('.pkl', '')
                    self.models[model_name] = joblib.load(f"{self.models_dir}/{model_file}")
        
        # Load feature columns
        with open(f"{self.models_dir}/feature_columns.json", 'r') as f:
            self.feature_columns = json.load(f)
        
        print(f"Loaded {len(self.models)} models and {len(self.scalers)} scalers")

def main():
    """
    Example usage of anomaly detection
    """
    import os
    
    # Load data
    if os.path.exists('./data/entities.csv') and os.path.exists('./data/transactions.csv'):
        entities_df = pd.read_csv('./data/entities.csv')
        transactions_df = pd.read_csv('./data/transactions.csv')
    else:
        print("Generating sample data first...")
        from data_generator import AMLDataGenerator
        generator = AMLDataGenerator(num_entities=500, num_transactions=2000)
        entities_df = generator.generate_entities()
        transactions_df = generator.generate_transactions()
    
    # Initialize detector
    detector = AnomalyDetector()
    
    # Prepare features
    features_df = detector.prepare_features(entities_df, transactions_df)
    
    # Train models
    detector.train_models(features_df)
    
    # Detect anomalies
    anomaly_scores = detector.detect_anomalies(features_df)
    
    # Detect patterns
    patterns = detector.detect_patterns(entities_df, transactions_df, anomaly_scores)
    
    print("\n" + "=" * 50)
    print("Anomaly Detection Results")
    print("=" * 50)
    print(f"Total Entities Analyzed: {len(anomaly_scores)}")
    print(f"Anomalies Detected: {anomaly_scores['is_anomaly'].sum()}")
    print(f"Suspicious Patterns Found: {len(patterns)}")
    
    # Show top anomalies
    top_anomalies = anomaly_scores.nlargest(5, 'ensemble_score')
    print("\nTop 5 Anomalous Entities:")
    for _, row in top_anomalies.iterrows():
        entity = entities_df[entities_df['entity_id'] == row['entity_id']].iloc[0]
        print(f"  {row['entity_id']}: {entity['name']} (Score: {row['ensemble_score']:.3f})")
    
    # Save results
    os.makedirs('./data', exist_ok=True)
    anomaly_scores.to_csv('./data/anomaly_scores.csv', index=False)
    pd.DataFrame(patterns).to_csv('./data/detected_patterns.csv', index=False)
    
    print(f"\nResults saved to ./data/")
    
    return anomaly_scores, patterns

if __name__ == "__main__":
    main()