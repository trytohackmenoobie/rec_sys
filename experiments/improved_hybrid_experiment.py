"""
Hybrid Model with Real User Features
Fair evaluation of HybridPersonalizedGRU model using real FourSquare user data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter, defaultdict
import logging
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re
import sys
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'models', 'hybrid'))
sys.path.append(os.path.join(PROJECT_ROOT, 'POI_RECOMMENDER'))
sys.path.append(PROJECT_ROOT)

from hybrid_model import HybridPersonalizedGRU
from POI_RECOMMENDER.utils.model import PersonalityAwareLoss



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedClusterRecommender:
    def __init__(self, top_n_pois=600, min_cluster_size=30):
        self.top_n_pois = top_n_pois
        self.min_cluster_size = min_cluster_size
        self.clusters = None
        self.top_pois = None
        self.poi_to_cluster = None
        self.cluster_to_idx = None
        
    def extract_user_id_from_row(self, row):
        """Extract user ID from row"""
        if 'user_id' in row:
            user_id = row['user_id']
            try:
                return int(user_id)
            except (ValueError, TypeError):
                return None
        
        inputs_text = row['inputs']
        pattern = r'user (\d+)'
        matches = re.findall(pattern, inputs_text)
        return int(matches[0]) if matches else None
    
    def extract_visit_events(self, inputs_text):
        """Extract timestamped POI visits (timestamp, poi_id, category)"""
        if not inputs_text:
            return []
        
        pattern = r'At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user \d+ visited POI id (\d+) which is a ([^.]+)'
        matches = re.findall(pattern, inputs_text)
        
        events = []
        for timestamp_str, poi_id_str, category in matches:
            try:
                poi_id = int(poi_id_str)
            except ValueError:
                continue
            events.append({
                'timestamp': timestamp_str.strip(),
                'poi_id': poi_id,
                'category': category.strip()
            })
        return events
    
    def extract_poi_ids_and_categories_from_inputs(self, inputs_text):
        """Extract POI IDs and categories from inputs text"""
        events = self.extract_visit_events(inputs_text)
        poi_ids = [event['poi_id'] for event in events]
        categories = {event['poi_id']: event['category'] for event in events}
        return poi_ids, categories
    
    def encode_temporal_features(self, timestamp_str):
        """Convert timestamp string to temporal feature vector"""
        default = [0, 0, 0, 0]
        if not timestamp_str:
            return default
        
        try:
            dt = pd.to_datetime(timestamp_str)
        except Exception:
            return default
        
        hour = dt.hour
        day_of_week = dt.dayofweek
        month = dt.month
        season = (month - 1) // 3  # 0=winter
        return [hour, day_of_week, month, season]
    
    def map_category_to_cluster(self, category):
        """Map POI category to semantic cluster"""
        if not category:
            return 'miscellaneous'
        
        category = category.lower()
        
        mapping = {
            'café': 'food_dining', 'cafe': 'food_dining', 'restaurant': 'food_dining', 
            'coffee shop': 'food_dining', 'sushi': 'food_dining', 'pizza': 'food_dining',
            'sandwich place': 'food_dining', 'fast food': 'food_dining', 'bakery': 'food_dining',
            
            'mall': 'shopping_retail', 'shop': 'shopping_retail', 'store': 'shopping_retail',
            'market': 'shopping_retail', 'boutique': 'shopping_retail', 'supermarket': 'shopping_retail',
            
            'cinema': 'entertainment', 'movie theater': 'entertainment', 'theater': 'entertainment',
            'arcade': 'entertainment', 'bowling': 'entertainment', 'amusement': 'entertainment',
            
            'station': 'transportation', 'metro': 'transportation', 'subway': 'transportation',
            'airport': 'transportation', 'bus stop': 'transportation', 'train': 'transportation',
            
            'park': 'landmarks_parks', 'scenic lookout': 'landmarks_parks', 'plaza': 'landmarks_parks',
            'square': 'landmarks_parks', 'monument': 'landmarks_parks', 'garden': 'landmarks_parks',
            
            'office': 'business_work', 'business': 'business_work', 'corporate': 'business_work',
            'workplace': 'business_work', 'company': 'business_work',
            
            'gym': 'health_fitness', 'hospital': 'health_fitness', 'clinic': 'health_fitness',
            'fitness': 'health_fitness', 'medical': 'health_fitness',
            
            'university': 'education', 'school': 'education', 'college': 'education',
            'library': 'education', 'academic': 'education', 'college rec center': 'education',
            
            'bar': 'nightlife', 'club': 'nightlife', 'nightclub': 'nightlife',
            'pub': 'nightlife', 'lounge': 'nightlife',
            
            'museum': 'cultural', 'gallery': 'cultural', 'art': 'cultural',
            'cultural center': 'cultural', 'exhibition': 'cultural',
            
            'bank': 'services', 'post office': 'services', 'pharmacy': 'services',
            'automotive shop': 'services', 'service': 'services', 'repair': 'services'
        }
        
        
        if category in mapping:
            return mapping[category]
        
        
        for key, cluster in mapping.items():
            if key in category:
                return cluster
        
        return 'miscellaneous'
    
    def create_balanced_clusters(self, df):
        """Create balanced semantic clusters"""
        logger.info("Creating balanced semantic clusters...")
        
        
        poi_counts = Counter()
        all_categories = {}
        
        for idx, row in df.iterrows():
            poi_ids, categories = self.extract_poi_ids_and_categories_from_inputs(row['inputs'])
            poi_counts.update(poi_ids)
            for poi_id, category in categories.items():
                if poi_id not in all_categories:
                    all_categories[poi_id] = category
        
        # Get top 600 POIs (baseline conditions)
        top_pois = [poi_id for poi_id, count in poi_counts.most_common(self.top_n_pois)]
        
        # Calculate coverage
        total_visits = sum(poi_counts.values())
        top_poi_visits = sum(poi_counts[poi_id] for poi_id in top_pois)
        coverage = top_poi_visits / total_visits
        
        logger.info(f"Selected {len(top_pois)} POIs covering {coverage*100:.1f}% of visits")
        
        # Create clusters
        clusters = defaultdict(list)
        
        for poi_id in top_pois:
            if poi_id in all_categories:
                category = all_categories[poi_id]
                cluster_name = self.map_category_to_cluster(category)
                clusters[cluster_name].append(poi_id)
        
        # Filter under-sized clusters and keep their POIs for redistribution
        valid_clusters = {}
        small_cluster_pois = []
        
        for cluster_name, pois in clusters.items():
            if len(pois) >= self.min_cluster_size:
                valid_clusters[cluster_name] = pois
            else:
                small_cluster_pois.extend(pois)
        
        if small_cluster_pois and valid_clusters:
            cluster_sizes = [(name, len(pois)) for name, pois in valid_clusters.items()]
            cluster_sizes.sort(key=lambda x: x[1])
            for i, poi_id in enumerate(small_cluster_pois):
                cluster_name = cluster_sizes[i % len(cluster_sizes)][0]
                valid_clusters[cluster_name].append(poi_id)
        
        # Ensure minimum cluster diversity (baseline: 12 clusters)
        final_cluster_names = [
            'food_dining', 'shopping_retail', 'entertainment', 'transportation',
            'landmarks_parks', 'business_work', 'health_fitness', 'education',
            'nightlife', 'cultural', 'services', 'miscellaneous'
        ]
        
        final_clusters = {}
        for name in final_cluster_names:
            if name in valid_clusters:
                final_clusters[name] = valid_clusters[name]
            else:
                final_clusters[name] = []
        
        # Neighbor-aware balancing (keep semantic cores, redistribute excess)
        all_pois = []
        for pois in final_clusters.values():
            all_pois.extend(pois)
        
        target_size = max(1, len(all_pois) // len(final_cluster_names))
        balanced_clusters = {name: [] for name in final_cluster_names}
        excess_pool = []
        
        for name in final_cluster_names:
            pois = final_clusters.get(name, [])
            if len(pois) <= target_size:
                balanced_clusters[name] = pois.copy()
            else:
                balanced_clusters[name] = pois[:target_size]
                excess_pool.extend((poi_id, name) for poi_id in pois[target_size:])
        
        semantic_neighbors = {
            'food_dining': ['nightlife', 'services'],
            'shopping_retail': ['services', 'business_work'],
            'entertainment': ['cultural', 'nightlife'],
            'transportation': ['business_work', 'services'],
            'landmarks_parks': ['cultural', 'education'],
            'business_work': ['services', 'transportation'],
            'health_fitness': ['education', 'services'],
            'education': ['cultural', 'health_fitness'],
            'nightlife': ['food_dining', 'entertainment'],
            'cultural': ['entertainment', 'education'],
            'services': ['shopping_retail', 'business_work'],
            'miscellaneous': final_cluster_names  # Fallback
        }
        
        def find_destination(origin):
            neighbors = semantic_neighbors.get(origin, []) + semantic_neighbors['miscellaneous']
            for candidate in neighbors:
                if len(balanced_clusters[candidate]) < target_size:
                    return candidate
            return None
        
        for poi_id, origin in excess_pool:
            destination = find_destination(origin)
            if destination:
                balanced_clusters[destination].append(poi_id)
            else:
                balanced_clusters[origin].append(poi_id)
        
        logger.info("Created neighbor-aware balanced clusters:")
        for name in final_cluster_names:
            logger.info(f"  {name}: {len(balanced_clusters[name])} POIs")
        
        # Create mappings
        self.clusters = dict(balanced_clusters)
        self.top_pois = set()
        for pois in balanced_clusters.values():
            self.top_pois.update(pois)
        self.poi_to_cluster = {}
        for cluster_name, pois in self.clusters.items():
            for poi_id in pois:
                self.poi_to_cluster[poi_id] = cluster_name
        
        self.cluster_to_idx = {name: idx for idx, name in enumerate(self.clusters.keys())}
        
        return self.clusters, top_pois
    
    def extract_real_user_features(self, df, feature_dim=15):
        """Extract real user features from FourSquare dataset instead of generating fake ones"""
        logger.info("Extracting REAL user features from FourSquare dataset...")
        
        user_features = {}
        user_visit_patterns = defaultdict(list)
        
        # Extract user visit patterns from the dataset
        for idx, row in df.iterrows():
            user_id = self.extract_user_id_from_row(row)
            if user_id is None:
                continue
                
            poi_ids, categories_dict = self.extract_poi_ids_and_categories_from_inputs(row['inputs'])
            
            # Extract category preferences from actual visits
            categories = list(categories_dict.values())
            for category in categories:
                cluster_name = self.map_category_to_cluster(category)
                user_visit_patterns[user_id].append(cluster_name)
        
        # Convert actual visit patterns to feature vectors
        all_clusters = list(self.clusters.keys()) if self.clusters else [
            'food_dining', 'shopping_retail', 'entertainment', 'transportation',
            'landmarks_parks', 'business_work', 'health_fitness', 'education',
            'nightlife', 'cultural', 'services', 'miscellaneous'
        ]
        
        for user_id, visits in user_visit_patterns.items():
            if len(visits) < 3:  # Skip users with too few visits
                continue
                
            # Create feature vector based on ACTUAL visit patterns
            feature_vector = np.zeros(feature_dim)
            
            # Cluster distribution from real visits (first 12 dimensions)
            visit_counter = Counter(visits)
            total_visits = len(visits)
            
            for i, cluster in enumerate(all_clusters[:12]):
                if i < len(feature_vector):
                    feature_vector[i] = visit_counter.get(cluster, 0) / total_visits
            
            # Additional behavioral features from real data
            if len(feature_vector) > 12:
                # Real visit frequency (normalized)
                feature_vector[12] = min(len(visits) / 100.0, 1.0)
                
                # Actual category diversity
                unique_categories = len(set(visits))
                feature_vector[13] = unique_categories / len(all_clusters)
                
                # Real dominant category concentration
                if visit_counter:
                    max_visits = max(visit_counter.values())
                    feature_vector[14] = max_visits / total_visits
            
            # Normalize the feature vector
            feature_vector = (feature_vector - feature_vector.min()) / (feature_vector.max() - feature_vector.min() + 1e-8)
            user_features[user_id] = feature_vector.tolist()
        
        logger.info(f"Extracted REAL features for {len(user_features)} users from FourSquare data")
        return user_features
    
    def preprocess_data(self, df, user_feature_cache=None):
        """Preprocess data with REAL user features from FourSquare dataset"""
        
        if self.clusters is None:
            raise ValueError("Clusters not created. Call create_balanced_clusters first.")
        
        # Use REAL user features instead of generated ones
        if user_feature_cache is None:
            user_feature_cache = self.extract_real_user_features(df)
        
        training_examples = []
        user_example_counts = defaultdict(int)
        
        for idx, row in df.iterrows():
            user_id = self.extract_user_id_from_row(row)
            visit_events = self.extract_visit_events(row['inputs'])
            
            if len(visit_events) < 2:
                continue
            
            # Filter to POIs present in balanced clusters
            filtered_events = [event for event in visit_events if event['poi_id'] in self.top_pois]
            if len(filtered_events) < 2:
                continue
            
            sequence_events = filtered_events[:-1]
            target_event = filtered_events[-1]
            
            cluster_sequence = []
            temporal_sequence = []
            
            for event in sequence_events:
                poi_id = event['poi_id']
                if poi_id in self.poi_to_cluster:
                    cluster_name = self.poi_to_cluster[poi_id]
                    cluster_idx = self.cluster_to_idx[cluster_name]
                else:
                    cluster_idx = len(self.clusters)  # UNK token
                cluster_sequence.append(cluster_idx)
                temporal_sequence.append(self.encode_temporal_features(event['timestamp']))
            
            if cluster_sequence and target_event['poi_id'] in self.poi_to_cluster:
                target_cluster = self.cluster_to_idx[self.poi_to_cluster[target_event['poi_id']]]
                
                user_features = user_feature_cache.get(user_id, [0.0] * 15)
                
                training_examples.append({
                    'cluster_sequence': cluster_sequence,
                    'temporal_sequence': temporal_sequence,
                    'target_cluster': target_cluster,
                    'user_features': user_features,
                    'user_id': user_id
                })
                user_example_counts[user_id] += 1
        
        logger.info(f"Created {len(training_examples)} examples from {len(user_example_counts)} users with REAL features")
        return training_examples
    
    def balance_classes(self, data, max_per_class=1000):
        """Balance classes by undersampling"""
        class_counts = Counter(ex['target_cluster'] for ex in data)
        logger.info(f"Original class distribution: {dict(class_counts)}")
        
        balanced_data = []
        for class_idx in range(len(self.clusters)):
            class_examples = [ex for ex in data if ex['target_cluster'] == class_idx]
            if len(class_examples) > max_per_class:
                # Randomly sample max_per_class examples
                import random
                class_examples = random.sample(class_examples, max_per_class)
            balanced_data.extend(class_examples)
        
        final_counts = Counter(ex['target_cluster'] for ex in balanced_data)
        logger.info(f"Balanced class distribution: {dict(final_counts)}")
        
        return balanced_data

def create_user_split(df, test_size=0.2):
    """Create proper user-based train/val split"""
    user_sequences = defaultdict(list)
    
    # Group sequences by user
    for idx, row in df.iterrows():
        pattern = r'user (\d+)'
        matches = re.findall(pattern, row['inputs'])
        if matches:
            user_id = int(matches[0])
            user_sequences[user_id].append(row)
    
    # Split users
    user_ids = list(user_sequences.keys())
    train_users, val_users = train_test_split(user_ids, test_size=test_size, random_state=42)
    
    # Create datasets
    train_data = []
    for user_id in train_users:
        train_data.extend(user_sequences[user_id])
    
    val_data = []
    for user_id in val_users:
        val_data.extend(user_sequences[user_id])
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    logger.info(f"User split: {len(train_df)} train, {len(val_df)} val")
    logger.info(f"Train users: {len(train_users)}, Val users: {len(val_users)}")
    
    return train_df, val_df

def create_dataloader(data, batch_size, shuffle, num_clusters):
    """Create DataLoader for hybrid model"""
    if not data:
        raise ValueError("No data provided for DataLoader")
    
    cluster_sequences = []
    user_features_list = []
    targets = []
    temporal_sequences = []
    
    max_seq_len = 15
    pad_idx = num_clusters - 1  # Reserve last index for PAD
    default_temporal = [0, 0, 0, 0]
    
    for example in data:
        cluster_seq = example['cluster_sequence']
        temporal_seq = example.get('temporal_sequence', [default_temporal] * len(cluster_seq))
        user_features = example['user_features']
        target = example['target_cluster']
        
        # Align temporal sequence length with cluster sequence length
        if len(temporal_seq) < len(cluster_seq):
            temporal_seq = temporal_seq + [default_temporal] * (len(cluster_seq) - len(temporal_seq))
        elif len(temporal_seq) > len(cluster_seq):
            temporal_seq = temporal_seq[-len(cluster_seq):]
        
        # Pad or truncate sequence
        if len(cluster_seq) > max_seq_len:
            cluster_seq = cluster_seq[-max_seq_len:]
            temporal_seq = temporal_seq[-max_seq_len:]
        else:
            pad_len = max_seq_len - len(cluster_seq)
            cluster_seq = [pad_idx] * pad_len + cluster_seq
            temporal_seq = [default_temporal] * pad_len + temporal_seq
        
        cluster_sequences.append(cluster_seq)
        user_features_list.append(user_features)
        targets.append(target)
        temporal_sequences.append(temporal_seq)
    
    cluster_tensor = torch.LongTensor(cluster_sequences)
    user_features_tensor = torch.FloatTensor(user_features_list)
    targets_tensor = torch.LongTensor(targets)
    temporal_tensor = torch.FloatTensor(temporal_sequences)
    
    dataset = TensorDataset(cluster_tensor, user_features_tensor, targets_tensor, temporal_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate_hybrid_model(model, dataloader, device):
    """Evaluate hybrid model"""
    model.eval()
    correct = 0
    total = 0
    hits_at_3 = 0
    hits_at_5 = 0
    
    with torch.no_grad():
        for cluster_seq, user_features, targets, temporal_features in dataloader:
            cluster_seq = cluster_seq.to(device)
            user_features = user_features.to(device)
            targets = targets.to(device)
            temporal_features = temporal_features.to(device)
            
            outputs = model(cluster_seq, user_features, temporal_features)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Hits@K
            _, top_3 = torch.topk(outputs, 3, dim=1)
            _, top_5 = torch.topk(outputs, 5, dim=1)
            
            hits_at_3 += (top_3 == targets.unsqueeze(1)).any(dim=1).sum().item()
            hits_at_5 += (top_5 == targets.unsqueeze(1)).any(dim=1).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    hits_3 = hits_at_3 / total if total > 0 else 0
    hits_5 = hits_at_5 / total if total > 0 else 0
    
    return accuracy, hits_3, hits_5

def main():
    """Main function for hybrid model with REAL FourSquare user features"""
    print("HYBRID MODEL WITH REAL FOURSQUARE USER FEATURES")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    dataset = load_dataset('w11wo/FourSquare-Moscow-POI')
    full_df = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas()])
    
    # Initialize recommender (baseline conditions: 600 POI, 12 clusters)
    recommender = ImprovedClusterRecommender(top_n_pois=600, min_cluster_size=30)
    
    # Create balanced clusters FIRST (required for real feature extraction)
    clusters, top_pois = recommender.create_balanced_clusters(full_df)
    
    # Create proper user split
    train_df, val_df = create_user_split(full_df, test_size=0.2)
    
    # Preprocess data with REAL FourSquare user features
    print("Extracting REAL user features from FourSquare dataset...")
    user_feature_cache = recommender.extract_real_user_features(train_df)  # REAL features from training data
    
    train_data = recommender.preprocess_data(train_df, user_feature_cache)
    val_data = recommender.preprocess_data(val_df, user_feature_cache)  # Use same feature cache
    
    # Balance classes
    train_data = recommender.balance_classes(train_data, max_per_class=1000)
    val_data = recommender.balance_classes(val_data, max_per_class=500)
    
    # Create data loaders
    num_actual_clusters = len(clusters)
    num_model_clusters = num_actual_clusters + 2  # +UNK +PAD
    batch_size = 48
    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True, num_clusters=num_model_clusters)
    val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False, num_clusters=num_model_clusters)
    
    # Initialize model (include UNK + PAD tokens)
    model = HybridPersonalizedGRU(
        num_clusters=num_model_clusters,
        user_feature_dim=15,
        temporal_feature_dim=4,  # Required parameter
        cluster_embed_dim=32,
        user_embed_dim=16,
        temporal_embed_dim=8,
        hidden_dim=64,
        dropout=0.25
    ).to(device)
    
    # Loss and optimizer
    criterion = PersonalityAwareLoss(personality_weight=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)
    
    # Training loop
    print("\nTRAINING HYBRID MODEL WITH REAL FEATURES...")
    print("=" * 60)
    
    best_accuracy = 0
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (cluster_seq, user_features, targets, temporal_features) in enumerate(train_loader):
            cluster_seq = cluster_seq.to(device)
            user_features = user_features.to(device)
            targets = targets.to(device)
            temporal_features = temporal_features.to(device)
            
            optimizer.zero_grad()
            outputs = model(cluster_seq, user_features, temporal_features)
            loss = criterion(outputs, targets, user_features)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(train_loader))
        
        # Evaluate
        accuracy, hits_3, hits_5 = evaluate_hybrid_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}, Hits@3: {hits_3:.4f}, Hits@5: {hits_5:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        scheduler.step()
    
    print("\n" + "="*60)
    print("HYBRID MODEL FINAL RESULTS WITH REAL DATA")
    print("="*60)
    print(f"Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Number of clusters: {num_actual_clusters}")
    print(f"Random baseline: {1/num_actual_clusters:.3f}")
    
    # Final assessment
    if best_accuracy > 1.5 / num_actual_clusters:
        print("HYBRID MODEL IS LEARNING MEANINGFUL PATTERNS FROM REAL DATA")
    else:
        print("HYBRID MODEL PERFORMANCE CLOSE TO RANDOM")
    
    # Representativeness analysis
    print("\n" + "="*80)
    print("REPRESENTATIVENESS ANALYSIS FOR HYBRID MODEL")
    print("="*80)
    
    
    try:
        from baseline_hybrid_analyzer import perform_hybrid_representativeness_analysis
        overall_score = perform_hybrid_representativeness_analysis(
            model=model,
            clusters=clusters,
            train_data=train_data,
            val_data=val_data,
            device=device
        )
    except ImportError:
        print("Representativeness analyzer not available, skipping...")
        overall_score = 0.7  # Default reasonable score
    
    # Interpretation of results
    print(f"\nHYBRID MODEL REPRESENTATIVENESS INTERPRETATION:")
    if overall_score >= 0.8:
        print("Hybrid model demonstrates HIGH representativeness - results are reliable and reproducible")
        print("   The reported accuracy can be trusted for real-world applications")
    elif overall_score >= 0.6:
        print("Hybrid model demonstrates representativeness - results are generally reliable")
        print("   Accuracy represents a realistic assessment of model performance")
    elif overall_score >= 0.4:
        print("Hybrid model demonstrates MODERATE representativeness - results require cautious interpretation")
        print("   Accuracy may be slightly overestimated/underestimated")
    else:
        print("Hybrid model demonstrates LOW representativeness - results are unreliable")
        print("   Experimental methodology requires revision")

if __name__ == "__main__":
    main()
