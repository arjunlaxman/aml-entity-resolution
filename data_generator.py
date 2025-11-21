"""
Synthetic AML Data Generator
Generates realistic transaction data with embedded money laundering patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib
import json
from faker import Faker
from typing import List, Dict, Tuple
import networkx as nx

fake = Faker()
np.random.seed(42)
random.seed(42)

class AMLDataGenerator:
    def __init__(self, num_entities=1000, num_transactions=10000):
        self.num_entities = num_entities
        self.num_transactions = num_transactions
        self.entities = []
        self.transactions = []
        self.anomaly_patterns = []
        
        # High-risk countries for AML
        self.high_risk_countries = ['KY', 'PA', 'VG', 'MT', 'CY', 'SC', 'VU', 'LR', 'RU', 'CN']
        self.low_risk_countries = ['US', 'GB', 'DE', 'CA', 'AU', 'JP', 'FR', 'CH', 'SE', 'NO']
        self.medium_risk_countries = ['MX', 'BR', 'IN', 'ZA', 'TH', 'MY', 'SG', 'HK', 'AE', 'TR']
        
    def generate_entities(self) -> pd.DataFrame:
        """Generate realistic entities with various risk profiles"""
        entities = []
        
        # Generate different entity types
        for i in range(self.num_entities):
            # Determine entity type distribution
            rand = random.random()
            if rand < 0.70:  # 70% personal accounts
                entity_type = 'personal'
                name = fake.name()
                base_risk = np.random.beta(2, 8)  # Most personal accounts are low risk
            elif rand < 0.85:  # 15% legitimate businesses
                entity_type = 'business'
                name = fake.company()
                base_risk = np.random.beta(3, 7)
            elif rand < 0.95:  # 10% shell companies
                entity_type = 'shell'
                name = self._generate_shell_company_name()
                base_risk = np.random.beta(8, 2)  # Most shell companies are high risk
            else:  # 5% unknown/crypto wallets
                entity_type = 'unknown'
                name = f"Wallet_{fake.sha1()[:12].upper()}"
                base_risk = np.random.beta(6, 4)
            
            # Assign country based on risk profile
            if base_risk > 0.7:
                country = random.choice(self.high_risk_countries)
            elif base_risk > 0.3:
                country = random.choice(self.medium_risk_countries)
            else:
                country = random.choice(self.low_risk_countries)
            
            # Generate entity resolution features
            email_domain = fake.domain_name() if entity_type != 'unknown' else None
            phone_hash = hashlib.md5(fake.phone_number().encode()).hexdigest() if random.random() > 0.3 else None
            address_hash = hashlib.md5(fake.address().encode()).hexdigest() if random.random() > 0.4 else None
            tax_id_hash = hashlib.md5(fake.ssn().encode()).hexdigest() if entity_type == 'business' else None
            
            # KYC score inversely correlated with risk
            kyc_score = max(0, min(1, 1 - base_risk + np.random.normal(0, 0.1)))
            
            entity = {
                'entity_id': f'E{str(i+1).zfill(4)}',
                'name': name,
                'type': entity_type,
                'risk_score': round(base_risk, 2),
                'country_code': country,
                'kyc_score': round(kyc_score, 2),
                'account_age_days': np.random.randint(30, 3650),
                'total_transaction_count': 0,
                'total_transaction_volume': 0,
                'suspicious_flag_count': 0,
                'phone_hash': phone_hash,
                'email_domain': email_domain,
                'address_hash': address_hash,
                'tax_id_hash': tax_id_hash,
                'device_fingerprint': fake.sha256()[:16] if random.random() > 0.5 else None,
                'cluster_id': None,  # Will be set by entity resolution
                'degree_centrality': 0,
                'betweenness_centrality': 0,
                'pagerank_score': 0,
                'community_id': None
            }
            entities.append(entity)
        
        self.entities = pd.DataFrame(entities)
        return self.entities
    
    def _generate_shell_company_name(self) -> str:
        """Generate realistic shell company names"""
        prefixes = ['Global', 'International', 'Offshore', 'Prime', 'Alpha', 'Delta', 'Universal']
        middles = ['Holdings', 'Investments', 'Capital', 'Trust', 'Assets', 'Financial']
        suffixes = ['Ltd', 'Corp', 'LLC', 'Group', 'Inc', 'SA', 'GmbH']
        
        return f"{random.choice(prefixes)}_{random.choice(middles)}_{random.choice(suffixes)}".upper()
    
    def generate_transactions(self) -> pd.DataFrame:
        """Generate transactions with embedded ML patterns"""
        transactions = []
        current_time = datetime.now() - timedelta(days=90)
        
        # Generate normal transactions (70%)
        normal_count = int(self.num_transactions * 0.7)
        for _ in range(normal_count):
            transaction = self._generate_normal_transaction(current_time)
            transactions.append(transaction)
            current_time += timedelta(minutes=random.randint(5, 120))
        
        # Generate suspicious patterns (30%)
        # 1. Layering patterns (10%)
        layering_count = int(self.num_transactions * 0.10)
        for _ in range(layering_count // 5):  # Each pattern has ~5 transactions
            pattern_txs, pattern_info = self._generate_layering_pattern(current_time)
            transactions.extend(pattern_txs)
            self.anomaly_patterns.append(pattern_info)
            current_time += timedelta(hours=random.randint(1, 6))
        
        # 2. Smurfing patterns (10%)
        smurfing_count = int(self.num_transactions * 0.10)
        for _ in range(smurfing_count // 4):  # Each pattern has ~4 transactions
            pattern_txs, pattern_info = self._generate_smurfing_pattern(current_time)
            transactions.extend(pattern_txs)
            self.anomaly_patterns.append(pattern_info)
            current_time += timedelta(hours=random.randint(2, 8))
        
        # 3. Round-tripping patterns (5%)
        round_trip_count = int(self.num_transactions * 0.05)
        for _ in range(round_trip_count // 3):  # Each pattern has ~3 transactions
            pattern_txs, pattern_info = self._generate_round_tripping(current_time)
            transactions.extend(pattern_txs)
            self.anomaly_patterns.append(pattern_info)
            current_time += timedelta(hours=random.randint(4, 12))
        
        # 4. Rapid movement patterns (5%)
        rapid_count = int(self.num_transactions * 0.05)
        for _ in range(rapid_count // 3):
            pattern_txs, pattern_info = self._generate_rapid_movement(current_time)
            transactions.extend(pattern_txs)
            self.anomaly_patterns.append(pattern_info)
            current_time += timedelta(minutes=random.randint(30, 180))
        
        self.transactions = pd.DataFrame(transactions)
        
        # Calculate graph features
        self._calculate_graph_features()
        
        return self.transactions
    
    def _generate_normal_transaction(self, timestamp: datetime) -> Dict:
        """Generate a normal, non-suspicious transaction"""
        # Select low-risk entities
        low_risk_entities = self.entities[self.entities['risk_score'] < 0.4]
        if len(low_risk_entities) < 2:
            low_risk_entities = self.entities
        
        from_entity = low_risk_entities.sample(1).iloc[0]
        to_entity = low_risk_entities[low_risk_entities['entity_id'] != from_entity['entity_id']].sample(1).iloc[0]
        
        # Normal transaction amounts
        amount = np.random.lognormal(8, 1.5)  # Log-normal distribution for amounts
        amount = min(amount, 50000)  # Cap at 50k for normal transactions
        
        return {
            'transaction_id': f"T{fake.uuid4()[:8].upper()}",
            'from_entity_id': from_entity['entity_id'],
            'to_entity_id': to_entity['entity_id'],
            'amount': round(amount, 2),
            'currency': 'USD',
            'transaction_type': random.choice(['wire', 'ach', 'check', 'debit']),
            'payment_method': random.choice(['bank_transfer', 'online', 'mobile', 'atm']),
            'timestamp': timestamp,
            'origin_country': from_entity['country_code'],
            'destination_country': to_entity['country_code'],
            'cross_border': from_entity['country_code'] != to_entity['country_code'],
            'suspicious': False,
            'flagged_by_rules': False,
            'risk_score': round(random.uniform(0, 0.3), 2),
            'is_round_amount': amount % 100 == 0,
            'velocity_flag': False,
            'structuring_flag': False
        }
    
    def _generate_layering_pattern(self, start_time: datetime) -> Tuple[List[Dict], Dict]:
        """Generate a layering pattern through shell companies"""
        # Select entities for layering
        shells = self.entities[self.entities['type'] == 'shell'].sample(min(3, len(self.entities[self.entities['type'] == 'shell'])))
        if len(shells) < 2:
            shells = self.entities.sample(3)
        
        source = self.entities[self.entities['type'] == 'personal'].sample(1).iloc[0]
        
        transactions = []
        entities_involved = [source['entity_id']]
        initial_amount = random.uniform(30000, 100000)
        current_amount = initial_amount
        current_time = start_time
        
        # Create chain of transactions
        from_entity = source
        for i, shell in shells.iterrows():
            to_entity = shell
            # Decay amount by 2-3% (fees/conversion)
            current_amount *= random.uniform(0.97, 0.98)
            
            transaction = {
                'transaction_id': f"T{fake.uuid4()[:8].upper()}",
                'from_entity_id': from_entity['entity_id'],
                'to_entity_id': to_entity['entity_id'],
                'amount': round(current_amount, 2),
                'currency': 'USD',
                'transaction_type': 'wire',
                'payment_method': 'bank_transfer',
                'timestamp': current_time,
                'origin_country': from_entity['country_code'],
                'destination_country': to_entity['country_code'],
                'cross_border': True,
                'suspicious': True,
                'flagged_by_rules': True,
                'risk_score': round(random.uniform(0.7, 0.95), 2),
                'is_round_amount': False,
                'velocity_flag': True,
                'structuring_flag': False
            }
            transactions.append(transaction)
            entities_involved.append(to_entity['entity_id'])
            
            from_entity = to_entity
            current_time += timedelta(hours=random.randint(12, 48))
        
        pattern_info = {
            'pattern_id': f"P{fake.uuid4()[:8].upper()}",
            'pattern_type': 'layering',
            'entities': entities_involved,
            'transactions': [t['transaction_id'] for t in transactions],
            'total_amount': initial_amount,
            'hop_count': len(transactions)
        }
        
        return transactions, pattern_info
    
    def _generate_smurfing_pattern(self, start_time: datetime) -> Tuple[List[Dict], Dict]:
        """Generate structuring/smurfing pattern"""
        # Multiple small deposits below reporting threshold
        sources = self.entities[self.entities['type'] == 'personal'].sample(min(4, len(self.entities[self.entities['type'] == 'personal'])))
        aggregator = self.entities[self.entities['type'].isin(['shell', 'business'])].sample(1).iloc[0]
        
        transactions = []
        entities_involved = [aggregator['entity_id']]
        total_amount = 0
        current_time = start_time
        
        # Generate multiple deposits just below 10k threshold
        for _, source in sources.iterrows():
            amount = random.uniform(9000, 9900)
            total_amount += amount
            
            transaction = {
                'transaction_id': f"T{fake.uuid4()[:8].upper()}",
                'from_entity_id': source['entity_id'],
                'to_entity_id': aggregator['entity_id'],
                'amount': round(amount, 2),
                'currency': 'USD',
                'transaction_type': 'deposit',
                'payment_method': random.choice(['cash', 'check', 'atm']),
                'timestamp': current_time,
                'origin_country': source['country_code'],
                'destination_country': aggregator['country_code'],
                'cross_border': source['country_code'] != aggregator['country_code'],
                'suspicious': True,
                'flagged_by_rules': True,
                'risk_score': round(random.uniform(0.6, 0.85), 2),
                'is_round_amount': False,
                'velocity_flag': False,
                'structuring_flag': True
            }
            transactions.append(transaction)
            entities_involved.append(source['entity_id'])
            current_time += timedelta(minutes=random.randint(30, 120))
        
        pattern_info = {
            'pattern_id': f"P{fake.uuid4()[:8].upper()}",
            'pattern_type': 'smurfing',
            'entities': entities_involved,
            'transactions': [t['transaction_id'] for t in transactions],
            'total_amount': total_amount,
            'hop_count': 1
        }
        
        return transactions, pattern_info
    
    def _generate_round_tripping(self, start_time: datetime) -> Tuple[List[Dict], Dict]:
        """Generate round-tripping pattern (money returns to source)"""
        entities = self.entities.sample(4)
        source = entities.iloc[0]
        intermediaries = entities.iloc[1:3]
        
        transactions = []
        entities_involved = [source['entity_id']]
        amount = random.uniform(20000, 80000)
        current_time = start_time
        
        # Create circular flow
        chain = [source] + list(intermediaries.itertuples(index=False)) + [source]
        
        for i in range(len(chain) - 1):
            from_entity = chain[i]
            to_entity = chain[i + 1]
            
            # Small decay for realism
            amount *= random.uniform(0.98, 0.99)
            
            transaction = {
                'transaction_id': f"T{fake.uuid4()[:8].upper()}",
                'from_entity_id': from_entity['entity_id'] if hasattr(from_entity, 'entity_id') else from_entity[0],
                'to_entity_id': to_entity['entity_id'] if hasattr(to_entity, 'entity_id') else to_entity[0],
                'amount': round(amount, 2),
                'currency': 'USD',
                'transaction_type': 'wire',
                'payment_method': 'bank_transfer',
                'timestamp': current_time,
                'origin_country': from_entity['country_code'] if hasattr(from_entity, 'country_code') else from_entity[3],
                'destination_country': to_entity['country_code'] if hasattr(to_entity, 'country_code') else to_entity[3],
                'cross_border': random.random() > 0.3,
                'suspicious': True,
                'flagged_by_rules': True,
                'risk_score': round(random.uniform(0.65, 0.9), 2),
                'is_round_amount': False,
                'velocity_flag': True,
                'structuring_flag': False
            }
            transactions.append(transaction)
            
            entity_id = to_entity['entity_id'] if hasattr(to_entity, 'entity_id') else to_entity[0]
            if entity_id not in entities_involved:
                entities_involved.append(entity_id)
            
            current_time += timedelta(hours=random.randint(6, 24))
        
        pattern_info = {
            'pattern_id': f"P{fake.uuid4()[:8].upper()}",
            'pattern_type': 'round_tripping',
            'entities': entities_involved,
            'transactions': [t['transaction_id'] for t in transactions],
            'total_amount': amount,
            'hop_count': len(transactions)
        }
        
        return transactions, pattern_info
    
    def _generate_rapid_movement(self, start_time: datetime) -> Tuple[List[Dict], Dict]:
        """Generate rapid movement pattern (velocity)"""
        entities = self.entities.sample(3)
        
        transactions = []
        entities_involved = [e['entity_id'] for _, e in entities.iterrows()]
        amount = random.uniform(15000, 50000)
        current_time = start_time
        
        # Rapid transactions within minutes
        for i in range(len(entities) - 1):
            from_entity = entities.iloc[i]
            to_entity = entities.iloc[i + 1]
            
            transaction = {
                'transaction_id': f"T{fake.uuid4()[:8].upper()}",
                'from_entity_id': from_entity['entity_id'],
                'to_entity_id': to_entity['entity_id'],
                'amount': round(amount, 2),
                'currency': 'USD',
                'transaction_type': 'instant',
                'payment_method': 'wire',
                'timestamp': current_time,
                'origin_country': from_entity['country_code'],
                'destination_country': to_entity['country_code'],
                'cross_border': from_entity['country_code'] != to_entity['country_code'],
                'suspicious': True,
                'flagged_by_rules': True,
                'risk_score': round(random.uniform(0.7, 0.95), 2),
                'is_round_amount': False,
                'velocity_flag': True,
                'structuring_flag': False
            }
            transactions.append(transaction)
            current_time += timedelta(minutes=random.randint(1, 10))  # Very rapid
        
        pattern_info = {
            'pattern_id': f"P{fake.uuid4()[:8].upper()}",
            'pattern_type': 'rapid_movement',
            'entities': entities_involved,
            'transactions': [t['transaction_id'] for t in transactions],
            'total_amount': amount * len(transactions),
            'hop_count': len(transactions)
        }
        
        return transactions, pattern_info
    
    def _calculate_graph_features(self):
        """Calculate graph-based features for entities and transactions"""
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for _, entity in self.entities.iterrows():
            G.add_node(entity['entity_id'], **entity.to_dict())
        
        # Add edges
        for _, transaction in self.transactions.iterrows():
            if G.has_node(transaction['from_entity_id']) and G.has_node(transaction['to_entity_id']):
                G.add_edge(
                    transaction['from_entity_id'],
                    transaction['to_entity_id'],
                    weight=transaction['amount'],
                    transaction_id=transaction['transaction_id']
                )
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G)
        
        # Detect communities
        G_undirected = G.to_undirected()
        communities = nx.community.greedy_modularity_communities(G_undirected)
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
        
        # Update entity features
        for entity_id in self.entities['entity_id']:
            if entity_id in G:
                idx = self.entities[self.entities['entity_id'] == entity_id].index[0]
                self.entities.at[idx, 'degree_centrality'] = round(degree_centrality.get(entity_id, 0), 4)
                self.entities.at[idx, 'betweenness_centrality'] = round(betweenness_centrality.get(entity_id, 0), 4)
                self.entities.at[idx, 'pagerank_score'] = round(pagerank.get(entity_id, 0), 4)
                self.entities.at[idx, 'community_id'] = community_map.get(entity_id, -1)
    
    def export_to_csv(self, output_dir: str = "./data"):
        """Export generated data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export entities
        self.entities.to_csv(f"{output_dir}/entities.csv", index=False)
        print(f"Exported {len(self.entities)} entities to {output_dir}/entities.csv")
        
        # Export transactions
        self.transactions.to_csv(f"{output_dir}/transactions.csv", index=False)
        print(f"Exported {len(self.transactions)} transactions to {output_dir}/transactions.csv")
        
        # Export anomaly patterns
        patterns_df = pd.DataFrame(self.anomaly_patterns)
        patterns_df.to_csv(f"{output_dir}/anomaly_patterns.csv", index=False)
        print(f"Exported {len(patterns_df)} anomaly patterns to {output_dir}/anomaly_patterns.csv")
        
        # Generate summary statistics
        summary = {
            'total_entities': len(self.entities),
            'entity_types': self.entities['type'].value_counts().to_dict(),
            'total_transactions': len(self.transactions),
            'suspicious_transactions': self.transactions['suspicious'].sum(),
            'total_volume': self.transactions['amount'].sum(),
            'avg_transaction_amount': self.transactions['amount'].mean(),
            'cross_border_percentage': (self.transactions['cross_border'].sum() / len(self.transactions) * 100),
            'high_risk_entities': len(self.entities[self.entities['risk_score'] > 0.7]),
            'detected_patterns': len(self.anomaly_patterns),
            'pattern_types': pd.DataFrame(self.anomaly_patterns)['pattern_type'].value_counts().to_dict() if self.anomaly_patterns else {}
        }
        
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary statistics saved to {output_dir}/summary.json")
        
        return summary

def main():
    """Generate synthetic AML dataset"""
    print("Generating synthetic AML dataset...")
    print("-" * 50)
    
    # Initialize generator
    generator = AMLDataGenerator(num_entities=1000, num_transactions=5000)
    
    # Generate data
    print("Generating entities...")
    entities = generator.generate_entities()
    
    print("Generating transactions with ML patterns...")
    transactions = generator.generate_transactions()
    
    # Export to CSV
    summary = generator.export_to_csv()
    
    print("\n" + "=" * 50)
    print("Dataset Generation Complete!")
    print("=" * 50)
    print(f"Entities: {summary['total_entities']}")
    print(f"Transactions: {summary['total_transactions']}")
    print(f"Suspicious Transactions: {summary['suspicious_transactions']}")
    print(f"Total Volume: ${summary['total_volume']:,.2f}")
    print(f"High Risk Entities: {summary['high_risk_entities']}")
    print(f"Detected Patterns: {summary['detected_patterns']}")
    
    return entities, transactions

if __name__ == "__main__":
    main()