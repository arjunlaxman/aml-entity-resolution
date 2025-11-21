"""
Entity Resolution Module
Advanced entity matching and clustering using multiple similarity techniques
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import recordlinkage
from fuzzywuzzy import fuzz
import jellyfish
import hashlib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime

class EntityResolver:
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize Entity Resolution system
        
        Args:
            similarity_threshold: Minimum similarity score to consider entities as matches
        """
        self.similarity_threshold = similarity_threshold
        self.resolution_results = []
        self.cluster_map = {}
        
    def resolve_entities(self, entities_df: pd.DataFrame, transactions_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Main entity resolution pipeline
        
        Args:
            entities_df: DataFrame with entity information
            transactions_df: Optional transactions for behavior-based matching
            
        Returns:
            DataFrame with cluster assignments
        """
        print("Starting entity resolution process...")
        
        # Step 1: Exact matching on hashed fields
        exact_clusters = self._exact_matching(entities_df)
        
        # Step 2: Fuzzy name matching
        fuzzy_clusters = self._fuzzy_name_matching(entities_df)
        
        # Step 3: Behavioral matching (if transactions provided)
        behavioral_clusters = []
        if transactions_df is not None:
            behavioral_clusters = self._behavioral_matching(entities_df, transactions_df)
        
        # Step 4: Network-based matching
        network_clusters = self._network_based_matching(entities_df, transactions_df)
        
        # Step 5: Merge all clustering results
        final_clusters = self._merge_clusters(exact_clusters, fuzzy_clusters, 
                                             behavioral_clusters, network_clusters)
        
        # Step 6: Assign cluster IDs to entities
        entities_df = self._assign_clusters(entities_df, final_clusters)
        
        # Step 7: Calculate cluster risk scores
        entities_df = self._calculate_cluster_risk(entities_df)
        
        print(f"Entity resolution complete. Found {len(final_clusters)} clusters.")
        
        return entities_df
    
    def _exact_matching(self, entities_df: pd.DataFrame) -> List[Set[str]]:
        """
        Perform exact matching on hashed fields
        """
        clusters = []
        matched_entities = set()
        
        # Match on phone hash
        for field in ['phone_hash', 'email_domain', 'address_hash', 'tax_id_hash', 'device_fingerprint']:
            if field not in entities_df.columns:
                continue
                
            field_groups = entities_df[entities_df[field].notna()].groupby(field)
            
            for value, group in field_groups:
                if len(group) > 1:
                    cluster = set(group['entity_id'].values)
                    # Check if any entity is already matched
                    if not cluster.intersection(matched_entities):
                        clusters.append(cluster)
                        matched_entities.update(cluster)
                    else:
                        # Merge with existing cluster
                        for i, existing_cluster in enumerate(clusters):
                            if cluster.intersection(existing_cluster):
                                clusters[i] = existing_cluster.union(cluster)
                                matched_entities.update(cluster)
                                break
        
        print(f"Exact matching found {len(clusters)} clusters")
        return clusters
    
    def _fuzzy_name_matching(self, entities_df: pd.DataFrame) -> List[Set[str]]:
        """
        Perform fuzzy matching on entity names
        """
        clusters = []
        name_data = entities_df[['entity_id', 'name']].copy()
        
        # Create record pairs for comparison
        indexer = recordlinkage.Index()
        indexer.block('type')  # Block on entity type for efficiency
        candidate_pairs = indexer.index(name_data)
        
        # Compare records
        compare = recordlinkage.Compare()
        compare.string('name', 'name', method='jarowinkler', threshold=0.85, label='name_similarity')
        
        # Handle the indexing properly
        name_data_indexed = name_data.set_index('entity_id')
        feature_vectors = compare.compute(candidate_pairs, name_data_indexed)
        
        # Find matches
        matches = feature_vectors[feature_vectors['name_similarity'] >= self.similarity_threshold]
        
        # Create clusters from matches
        G = nx.Graph()
        for (entity1, entity2), score in matches.iterrows():
            G.add_edge(entity1, entity2, weight=score['name_similarity'])
        
        # Find connected components (clusters)
        for component in nx.connected_components(G):
            if len(component) > 1:
                clusters.append(component)
        
        print(f"Fuzzy name matching found {len(clusters)} clusters")
        return clusters
    
    def _behavioral_matching(self, entities_df: pd.DataFrame, 
                            transactions_df: pd.DataFrame) -> List[Set[str]]:
        """
        Match entities based on transaction behavior patterns
        """
        clusters = []
        
        # Create behavioral features
        behavioral_features = []
        
        for entity_id in entities_df['entity_id']:
            # Get entity's transactions
            sent_txs = transactions_df[transactions_df['from_entity_id'] == entity_id]
            received_txs = transactions_df[transactions_df['to_entity_id'] == entity_id]
            
            features = {
                'entity_id': entity_id,
                'avg_sent_amount': sent_txs['amount'].mean() if len(sent_txs) > 0 else 0,
                'avg_received_amount': received_txs['amount'].mean() if len(received_txs) > 0 else 0,
                'transaction_frequency': len(sent_txs) + len(received_txs),
                'unique_counterparties': len(set(sent_txs['to_entity_id'].tolist() + 
                                               received_txs['from_entity_id'].tolist())),
                'preferred_hour': self._get_preferred_hour(sent_txs, received_txs),
                'cross_border_ratio': self._get_cross_border_ratio(sent_txs, received_txs)
            }
            behavioral_features.append(features)
        
        behavior_df = pd.DataFrame(behavioral_features)
        
        # Normalize features
        feature_cols = ['avg_sent_amount', 'avg_received_amount', 'transaction_frequency', 
                       'unique_counterparties', 'preferred_hour', 'cross_border_ratio']
        
        # Handle case where all values are 0 or NaN
        behavior_matrix = behavior_df[feature_cols].fillna(0).values
        if behavior_matrix.max() > 0:
            scaler = StandardScaler()
            behavior_normalized = scaler.fit_transform(behavior_matrix)
            
            # Cluster using DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
            cluster_labels = clustering.fit_predict(behavior_normalized)
            
            # Create clusters
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label != -1:  # -1 is noise in DBSCAN
                    cluster_indices = np.where(cluster_labels == label)[0]
                    cluster = set(behavior_df.iloc[cluster_indices]['entity_id'].values)
                    clusters.append(cluster)
        
        print(f"Behavioral matching found {len(clusters)} clusters")
        return clusters
    
    def _network_based_matching(self, entities_df: pd.DataFrame, 
                               transactions_df: pd.DataFrame = None) -> List[Set[str]]:
        """
        Match entities based on network proximity and patterns
        """
        clusters = []
        
        if transactions_df is None or len(transactions_df) == 0:
            return clusters
        
        # Build transaction network
        G = nx.DiGraph()
        
        for _, entity in entities_df.iterrows():
            G.add_node(entity['entity_id'], **entity.to_dict())
        
        for _, tx in transactions_df.iterrows():
            G.add_edge(tx['from_entity_id'], tx['to_entity_id'], 
                      weight=tx['amount'], transaction_id=tx['transaction_id'])
        
        # Find entities with similar network positions
        # Using ego networks (1-hop neighborhoods)
        for node in G.nodes():
            if G.degree(node) > 0:
                ego_net = nx.ego_graph(G, node, radius=1)
                
                # Find nodes with very similar neighborhoods
                for other_node in G.nodes():
                    if other_node != node and G.degree(other_node) > 0:
                        other_ego = nx.ego_graph(G, other_node, radius=1)
                        
                        # Calculate Jaccard similarity of neighborhoods
                        neighbors1 = set(ego_net.nodes()) - {node}
                        neighbors2 = set(other_ego.nodes()) - {other_node}
                        
                        if len(neighbors1) > 0 and len(neighbors2) > 0:
                            intersection = neighbors1.intersection(neighbors2)
                            union = neighbors1.union(neighbors2)
                            jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                            
                            if jaccard > 0.7:  # High neighborhood overlap
                                clusters.append({node, other_node})
        
        # Merge overlapping clusters
        merged_clusters = self._merge_overlapping_sets(clusters)
        
        print(f"Network-based matching found {len(merged_clusters)} clusters")
        return merged_clusters
    
    def _merge_clusters(self, *cluster_lists: List[List[Set[str]]]) -> List[Set[str]]:
        """
        Merge clusters from different matching methods
        """
        all_clusters = []
        for cluster_list in cluster_lists:
            all_clusters.extend(cluster_list)
        
        # Merge overlapping clusters
        final_clusters = self._merge_overlapping_sets(all_clusters)
        
        return final_clusters
    
    def _merge_overlapping_sets(self, sets_list: List[Set[str]]) -> List[Set[str]]:
        """
        Merge sets that have overlapping elements
        """
        if not sets_list:
            return []
        
        # Convert to list for processing
        clusters = [s for s in sets_list if s]
        
        merged = []
        while clusters:
            current = clusters.pop(0)
            merged_with_current = [current]
            
            i = 0
            while i < len(clusters):
                if current.intersection(clusters[i]):
                    current = current.union(clusters[i])
                    merged_with_current.append(clusters[i])
                    clusters.pop(i)
                else:
                    i += 1
            
            merged.append(current)
        
        return merged
    
    def _assign_clusters(self, entities_df: pd.DataFrame, 
                        clusters: List[Set[str]]) -> pd.DataFrame:
        """
        Assign cluster IDs to entities
        """
        entities_df['cluster_id'] = entities_df['entity_id']
        
        for i, cluster in enumerate(clusters):
            cluster_id = f"C{str(i+1).zfill(4)}"
            for entity_id in cluster:
                entities_df.loc[entities_df['entity_id'] == entity_id, 'cluster_id'] = cluster_id
                
            # Store cluster information
            self.cluster_map[cluster_id] = {
                'entities': list(cluster),
                'size': len(cluster),
                'created_at': datetime.now()
            }
        
        return entities_df
    
    def _calculate_cluster_risk(self, entities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores for entity clusters
        """
        cluster_risks = entities_df.groupby('cluster_id').agg({
            'risk_score': ['mean', 'max'],
            'entity_id': 'count',
            'type': lambda x: 'shell' if 'shell' in x.values else x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        })
        
        cluster_risks.columns = ['avg_risk', 'max_risk', 'cluster_size', 'dominant_type']
        
        # Calculate composite cluster risk score
        cluster_risks['cluster_risk_score'] = (
            cluster_risks['avg_risk'] * 0.4 + 
            cluster_risks['max_risk'] * 0.3 +
            (cluster_risks['cluster_size'] > 3).astype(float) * 0.2 +
            (cluster_risks['dominant_type'] == 'shell').astype(float) * 0.1
        )
        
        # Merge back to entities
        for cluster_id, row in cluster_risks.iterrows():
            entities_df.loc[entities_df['cluster_id'] == cluster_id, 'cluster_risk_score'] = row['cluster_risk_score']
        
        return entities_df
    
    def _get_preferred_hour(self, sent_txs: pd.DataFrame, received_txs: pd.DataFrame) -> int:
        """
        Get the preferred transaction hour for an entity
        """
        all_timestamps = pd.concat([sent_txs['timestamp'], received_txs['timestamp']])
        
        if len(all_timestamps) == 0:
            return 12  # Default noon
        
        # Convert to datetime if string
        if isinstance(all_timestamps.iloc[0], str):
            all_timestamps = pd.to_datetime(all_timestamps)
        
        hours = all_timestamps.dt.hour
        if len(hours) > 0:
            return int(hours.mode()[0]) if len(hours.mode()) > 0 else 12
        return 12
    
    def _get_cross_border_ratio(self, sent_txs: pd.DataFrame, received_txs: pd.DataFrame) -> float:
        """
        Calculate ratio of cross-border transactions
        """
        all_txs = pd.concat([sent_txs, received_txs])
        
        if len(all_txs) == 0:
            return 0.0
        
        if 'cross_border' in all_txs.columns:
            return all_txs['cross_border'].mean()
        return 0.0
    
    def get_resolution_report(self) -> Dict:
        """
        Generate entity resolution summary report
        """
        report = {
            'total_clusters': len(self.cluster_map),
            'cluster_sizes': {},
            'resolution_methods': {
                'exact_matching': 0,
                'fuzzy_matching': 0,
                'behavioral_matching': 0,
                'network_matching': 0
            },
            'high_risk_clusters': [],
            'large_clusters': []
        }
        
        for cluster_id, cluster_info in self.cluster_map.items():
            size = cluster_info['size']
            
            # Track cluster size distribution
            size_bucket = f"{(size // 5) * 5}-{(size // 5 + 1) * 5 - 1}" if size < 20 else "20+"
            report['cluster_sizes'][size_bucket] = report['cluster_sizes'].get(size_bucket, 0) + 1
            
            # Track large clusters
            if size > 5:
                report['large_clusters'].append({
                    'cluster_id': cluster_id,
                    'size': size,
                    'entities': cluster_info['entities'][:5]  # First 5 entities
                })
        
        return report

def main():
    """
    Example usage of Entity Resolution
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
    
    # Initialize resolver
    resolver = EntityResolver(similarity_threshold=0.85)
    
    # Perform resolution
    resolved_entities = resolver.resolve_entities(entities_df, transactions_df)
    
    # Get report
    report = resolver.get_resolution_report()
    
    print("\n" + "=" * 50)
    print("Entity Resolution Report")
    print("=" * 50)
    print(f"Total Clusters Found: {report['total_clusters']}")
    print(f"Cluster Size Distribution: {report['cluster_sizes']}")
    print(f"Large Clusters (>5 entities): {len(report['large_clusters'])}")
    
    # Save resolved entities
    os.makedirs('./data', exist_ok=True)
    resolved_entities.to_csv('./data/resolved_entities.csv', index=False)
    print(f"\nSaved resolved entities to ./data/resolved_entities.csv")
    
    return resolved_entities

if __name__ == "__main__":
    main()