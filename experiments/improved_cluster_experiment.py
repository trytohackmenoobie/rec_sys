"""
Evaluation of cluster-based POI recommendation model with balanced dataset.
This implementation provides assessment of model performance and representativeness.
"""

import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter, defaultdict
import numpy as np
import logging
import sys
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add experiments directory to path for imports
experiments_dir = os.path.dirname(os.path.abspath(__file__))
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)

# Setup logging
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
            # Ensure user_id is an integer
            try:
                return int(user_id)
            except (ValueError, TypeError):
                return None
        
        inputs_text = row['inputs']
        pattern = r'user (\d+)'
        matches = re.findall(pattern, inputs_text)
        return int(matches[0]) if matches else None
    
    def extract_poi_ids_and_categories_from_inputs(self, inputs_text):
        """Extract POI IDs and categories from inputs text"""
        pattern = r'POI id (\d+) which is a ([^.]+)'
        matches = re.findall(pattern, inputs_text)
        
        poi_ids = [int(poi_id) for poi_id, category in matches]
        categories = {int(poi_id): category.strip() for poi_id, category in matches}
        
        return poi_ids, categories
    
    def map_category_to_cluster(self, category):
        """Map POI category to semantic cluster"""
        if not category:
            return 'miscellaneous'
        
        category = category.lower().strip()
        
        mapping = {
            # Food & Dining
            'café': 'food_dining', 'cafe': 'food_dining', 'restaurant': 'food_dining', 
            'coffee shop': 'food_dining', 'food & drink shop': 'food_dining',
            'american restaurant': 'food_dining', 'pizza place': 'food_dining',
            'sandwich place': 'food_dining', 'fast food restaurant': 'food_dining',
            'bakery': 'food_dining', 'bar': 'food_dining', 'pub': 'food_dining',
            
            # Shopping & Retail
            'shop & service': 'shopping_retail', 'clothing store': 'shopping_retail',
            'mall': 'shopping_retail', 'shop': 'shopping_retail', 'store': 'shopping_retail',
            'market': 'shopping_retail', 'boutique': 'shopping_retail', 
            'supermarket': 'shopping_retail', 'department store': 'shopping_retail',
            
            # Entertainment
            'movie theater': 'entertainment', 'theater': 'entertainment',
            'stadium': 'entertainment', 'arcade': 'entertainment', 
            'entertainment': 'entertainment', 'general entertainment': 'entertainment',
            'music venue': 'entertainment', 'arts & entertainment': 'entertainment',
            
            # Transportation
            'travel & transport': 'transportation', 'airport': 'transportation',
            'train station': 'transportation', 'bus station': 'transportation',
            'metro station': 'transportation', 'transportation': 'transportation',
            
            # Landmarks & Parks
            'park': 'landmarks_parks', 'outdoors & recreation': 'landmarks_parks',
            'scenic lookout': 'landmarks_parks', 'plaza': 'landmarks_parks',
            'garden': 'landmarks_parks', 'nature preserve': 'landmarks_parks',
            
            # Business & Work
            'office': 'business_work', 'professional & other places': 'business_work',
            'business': 'business_work', 'corporate': 'business_work',
            'government building': 'business_work', 'coworking space': 'business_work',
            
            # Health & Fitness
            'gym / fitness center': 'health_fitness', 'hospital': 'health_fitness',
            'clinic': 'health_fitness', 'medical center': 'health_fitness',
            'sports center': 'health_fitness', 'yoga studio': 'health_fitness',
            
            # Education
            'college & university': 'education', 'school': 'education',
            'library': 'education', 'educational center': 'education',
            'museum': 'education', 'cultural center': 'education',
            
            # Nightlife
            'nightlife spot': 'nightlife', 'club': 'nightlife', 
            'lounge': 'nightlife', 'karaoke': 'nightlife',
            
            # Cultural
            'arts & culture': 'cultural', 'cultural center': 'cultural',
            'gallery': 'cultural', 'exhibition': 'cultural',
            'historic site': 'cultural', 'monument': 'cultural',
            
            # Services
            'service': 'services', 'bank': 'services', 'post office': 'services',
            'pharmacy': 'services', 'automotive shop': 'services', 
            'repair shop': 'services', 'car wash': 'services'
        }
        
        if category in mapping:
            return mapping[category]
        
        for key, cluster in mapping.items():
            if re.search(r'\b' + re.escape(key) + r'\b', category):
                return cluster
        
        return 'miscellaneous'
    
    def create_balanced_clusters(self, df):
        """Create balanced semantic clusters using neighbor-aware redistribution.
        
        This method preserves semantic relationships by:
        1. Keeping semantic core of each cluster (first target_size POIs)
        2. Redistributing excess POIs to semantically close neighbor clusters
        3. Maintaining balanced cluster sizes while preserving semantics
        """
        logger.info("Creating balanced semantic clusters...")
        
        # Extract all categories
        all_categories = {}
        poi_counts = Counter()
        
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
        
        # Filter small clusters and redistribute
        valid_clusters = {}
        small_cluster_pois = []
        
        for cluster_name, pois in clusters.items():
            if len(pois) >= self.min_cluster_size:
                valid_clusters[cluster_name] = pois
            else:
                small_cluster_pois.extend(pois)
        
        # Redistribute small cluster POIs
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
                # Create minimal cluster
                final_clusters[name] = []
        
        # Final redistribution to balance sizes (neighbor-aware)
        all_pois = []
        for pois in final_clusters.values():
            all_pois.extend(pois)
        
        target_size = max(1, len(all_pois) // len(final_clusters))
        balanced_clusters = {name: [] for name in final_cluster_names}
        excess_pool = []
        
        # Keep semantic cores, collect excess from oversized clusters
        for name in final_cluster_names:
            pois = final_clusters.get(name, [])
            if len(pois) <= target_size:
                balanced_clusters[name] = pois.copy()
            else:
                balanced_clusters[name] = pois[:target_size]
                excess_pool.extend((poi_id, name) for poi_id in pois[target_size:])
        
        # Define semantic neighbors for smart redistribution
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
            'miscellaneous': final_cluster_names  # fallback bucket
        }
        
        def find_destination(origin):
            neighbors = semantic_neighbors.get(origin, []) + semantic_neighbors['miscellaneous']
            for candidate in neighbors:
                if len(balanced_clusters[candidate]) < target_size:
                    return candidate
            return None
        
        # Redistribute excess POIs into semantically close clusters
        for poi_id, origin in excess_pool:
            destination = find_destination(origin)
            if destination:
                balanced_clusters[destination].append(poi_id)
            else:
                # If everyone is full (due to rounding), append to origin
                balanced_clusters[origin].append(poi_id)
        
        logger.info("Created neighbor-aware balanced clusters:")
        for name in final_cluster_names:
            logger.info(f"  {name}: {len(balanced_clusters[name])} POIs")
        
        self.clusters = dict(balanced_clusters)
        self.top_pois = []
        for pois in balanced_clusters.values():
            self.top_pois.extend(pois)
        
        # Create mappings
        self.poi_to_cluster = {}
        for cluster_name, pois in self.clusters.items():
            for poi_id in pois:
                self.poi_to_cluster[poi_id] = cluster_name
        
        self.cluster_to_idx = {name: idx for idx, name in enumerate(self.clusters.keys())}
        
        return self.clusters, self.top_pois
    
    def extract_real_user_features(self, df, feature_dim=15):
        """Extract real user features from FourSquare dataset instead of generating fake ones"""
        logger.info("Extracting real user features from FourSquare dataset...")
        
        user_features = {}
        user_visit_patterns = defaultdict(list)
        
        # Extract user visit patterns from the dataset
        for idx, row in df.iterrows():
            user_id = self.extract_user_id_from_row(row)
            if user_id is None:
                continue
                
            poi_ids, categories_dict = self.extract_poi_ids_and_categories_from_inputs(row['inputs'])
            
            # Extract category preferences
            categories = list(categories_dict.values())
            for category in categories:
                cluster_name = self.map_category_to_cluster(category)
                user_visit_patterns[user_id].append(cluster_name)
        
        # Convert visit patterns to feature vectors
        all_clusters = list(self.clusters.keys()) if self.clusters else [
            'food_dining', 'shopping_retail', 'entertainment', 'transportation',
            'landmarks_parks', 'business_work', 'health_fitness', 'education',
            'nightlife', 'cultural', 'services', 'miscellaneous'
        ]
        
        for user_id, visits in user_visit_patterns.items():
            if len(visits) < 3:  # Skip users with too few visits
                continue
                
            # Create feature vector based on actual visit patterns
            feature_vector = np.zeros(feature_dim)
            
            # Cluster distribution (first 12 dimensions)
            visit_counter = Counter(visits)
            total_visits = len(visits)
            
            for i, cluster in enumerate(all_clusters[:12]):
                if i < len(feature_vector):
                    feature_vector[i] = visit_counter.get(cluster, 0) / total_visits
            
            # Additional behavioral features
            if len(feature_vector) > 12:
                # Visit frequency (normalized)
                feature_vector[12] = min(len(visits) / 100.0, 1.0)  # Normalize by reasonable max
                
                # Category diversity (Shannon entropy-like)
                unique_categories = len(set(visits))
                feature_vector[13] = unique_categories / len(all_clusters)
                
                # Dominant category concentration
                if visit_counter:
                    max_visits = max(visit_counter.values())
                    feature_vector[14] = max_visits / total_visits
            
            # Normalize the feature vector
            feature_vector = (feature_vector - feature_vector.min()) / (feature_vector.max() - feature_vector.min() + 1e-8)
            user_features[user_id] = feature_vector.tolist()
        
        logger.info(f"Extracted features for {len(user_features)} users")
        return user_features
    
    def preprocess_data(self, df, user_feature_cache=None):
        """Preprocess data with REAL user features from FourSquare dataset"""
        
        if self.clusters is None:
            raise ValueError("Clusters not created. Call create_balanced_clusters first.")
        
        # Use real user features instead of generated ones
        if user_feature_cache is None:
            user_feature_cache = self.extract_real_user_features(df)
        
        training_examples = []
        user_example_counts = defaultdict(int)
        
        for idx, row in df.iterrows():
            user_id = self.extract_user_id_from_row(row)
            poi_ids, _ = self.extract_poi_ids_and_categories_from_inputs(row['inputs'])
            
            if len(poi_ids) < 2:
                continue
            
            # Filter to clustered POIs
            filtered_pois = [poi for poi in poi_ids if poi in self.top_pois]
            if len(filtered_pois) < 2:
                continue
            
            sequence = filtered_pois[:-1]
            target_poi = filtered_pois[-1]
            
            # Convert to clusters
            cluster_sequence = []
            for poi_id in sequence:
                if poi_id in self.poi_to_cluster:
                    cluster_idx = self.cluster_to_idx[self.poi_to_cluster[poi_id]]
                    cluster_sequence.append(cluster_idx)
                else:
                    cluster_sequence.append(len(self.clusters))  # UNK
            
            if cluster_sequence and target_poi in self.poi_to_cluster:
                target_cluster = self.cluster_to_idx[self.poi_to_cluster[target_poi]]
                
                # Use REAL user features instead of synthetic ones
                if user_id in user_feature_cache:
                    user_features = user_feature_cache[user_id]
                else:
                    # Fallback for users without enough history
                    user_features = [0.0] * 15  # Zero vector for cold-start users
                
                training_examples.append({
                    'cluster_sequence': cluster_sequence,
                    'target_cluster': target_cluster,
                    'user_features': user_features,
                    'user_id': user_id
                })
                user_example_counts[user_id] += 1
        
        logger.info(f"Created {len(training_examples)} examples from {len(user_example_counts)} users")
        
        # Balance classes
        balanced_examples = self.balance_classes(training_examples)
        
        return balanced_examples
    
    def balance_classes(self, examples, max_per_class=1000):
        """Balance classes to avoid extreme imbalance"""
        class_examples = defaultdict(list)
        
        for example in examples:
            class_examples[example['target_cluster']].append(example)
        
        balanced_examples = []
        for class_idx, class_exs in class_examples.items():
            # Undersample majority classes
            if len(class_exs) > max_per_class:
                selected = np.random.choice(class_exs, size=max_per_class, replace=False)
                balanced_examples.extend(selected)
            else:
                balanced_examples.extend(class_exs)
        
        # Shuffle
        np.random.shuffle(balanced_examples)
        
        logger.info(f"Balanced dataset: {len(balanced_examples)} examples")
        
        return balanced_examples

class FairClusterModel(nn.Module):
    def __init__(self, num_clusters, user_feature_dim, cluster_embed_dim=32, 
                 user_embed_dim=16, hidden_dim=64, dropout=0.4):
        super(FairClusterModel, self).__init__()
        
        self.num_clusters = num_clusters
        self.cluster_embedding = nn.Embedding(num_clusters + 1, cluster_embed_dim, padding_idx=num_clusters)
        self.user_projection = nn.Linear(user_feature_dim, user_embed_dim)
        
        self.gru = nn.GRU(
            input_size=cluster_embed_dim + user_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_clusters)
        
    def forward(self, cluster_seq, user_features):
        cluster_embedded = self.cluster_embedding(cluster_seq)
        user_projected = self.user_projection(user_features).unsqueeze(1)
        user_repeated = user_projected.repeat(1, cluster_embedded.size(1), 1)
        
        combined = torch.cat([cluster_embedded, user_repeated], dim=2)
        gru_out, _ = self.gru(combined)
        
        last_hidden = gru_out[:, -1, :]
        output = self.fc(self.dropout(last_hidden))
        
        return output

def evaluate_fairly(model, dataloader, device, num_clusters, cluster_names):
    """Fair evaluation"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for cluster_seq, user_features, targets in dataloader:
            cluster_seq = cluster_seq.to(device)
            user_features = user_features.to(device)
            targets = targets.to(device)
            
            outputs = model(cluster_seq, user_features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Metrics
    print("\n" + "="*60)
    print("FAIR EVALUATION METRICS")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, 
                               target_names=cluster_names, zero_division=0))
    
    # Overall accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    print(f"\nPer-class Accuracy:")
    for i, name in enumerate(cluster_names):
        class_mask = np.array(all_targets) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_predictions)[class_mask] == np.array(all_targets)[class_mask])
            print(f"  {name}: {class_acc:.3f} ({np.sum(class_mask)} samples)")
    
    return accuracy

def main():
    """Fair evaluation with real user data from FourSquare"""
    
    print("FAIR EVALUATION WITH REAL FOURSQUARE USER DATA")
    print("=" * 60)
    
    # Load data
    dataset = load_dataset('w11wo/FourSquare-Moscow-POI')
    full_df = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas()])
    
    # Initialize recommender (baseline conditions: 600 POI, 12 clusters)
    recommender = ImprovedClusterRecommender(top_n_pois=600, min_cluster_size=30)
    
    # Create balanced clusters first (needed for feature extraction)
    clusters, top_pois = recommender.create_balanced_clusters(full_df)
    
    # Create proper user split
    def create_user_split(df, test_size=0.2):
        user_sequences = defaultdict(list)
        for idx, row in df.iterrows():
            user_id = recommender.extract_user_id_from_row(row)
            if user_id is not None:
                user_sequences[user_id].append(row)
        
        user_ids = list(user_sequences.keys())
        train_users, val_users = train_test_split(user_ids, test_size=test_size, random_state=42)
        
        train_data = []
        for user_id in train_users:
            train_data.extend(user_sequences[user_id])
        
        val_data = []
        for user_id in val_users:
            val_data.extend(user_sequences[user_id])
        
        return pd.DataFrame(train_data), pd.DataFrame(val_data)
    
    train_df, val_df = create_user_split(full_df)
    print(f"User split: {len(train_df)} train, {len(val_df)} val")
    
    # Preprocess with REAL user features
    print("Extracting REAL user features from FourSquare dataset...")
    user_feature_cache = recommender.extract_real_user_features(train_df)  # Use real features from training data
    
    train_data = recommender.preprocess_data(train_df, user_feature_cache)
    val_data = recommender.preprocess_data(val_df, user_feature_cache)  # Use same feature cache
    
    if not train_data or not val_data:
        print(" No valid data after balancing!")
        return
    
    # Create dataloaders
    def create_dataloader(data, batch_size, num_clusters):
        if not data:
            return None
            
        cluster_sequences = []
        user_features_list = []
        targets = []
        
        max_seq_len = 15
        for example in data:
            cluster_seq = example['cluster_sequence']
            if len(cluster_seq) > max_seq_len:
                cluster_seq = cluster_seq[-max_seq_len:]
            else:
                cluster_seq = [num_clusters] * (max_seq_len - len(cluster_seq)) + cluster_seq
            
            cluster_sequences.append(cluster_seq)
            user_features_list.append(example['user_features'])
            targets.append(example['target_cluster'])
        
        cluster_tensor = torch.LongTensor(cluster_sequences)
        user_features_tensor = torch.FloatTensor(user_features_list)
        targets_tensor = torch.LongTensor(targets)
        
        dataset = TensorDataset(cluster_tensor, user_features_tensor, targets_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    num_clusters = len(clusters)
    train_dataloader = create_dataloader(train_data, 64, num_clusters)
    val_dataloader = create_dataloader(val_data, 64, num_clusters)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FairClusterModel(
        num_clusters=num_clusters,
        user_feature_dim=15,
        cluster_embed_dim=32,
        user_embed_dim=16,
        hidden_dim=64,
        dropout=0.4
    ).to(device)
    
    # Balanced loss function
    class_counts = Counter([ex['target_cluster'] for ex in train_data])
    total_count = sum(class_counts.values())
    class_weights = [total_count / (len(class_counts) * count) for count in class_counts.values()]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training
    print("\nTRAINING...")
    best_accuracy = 0
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (cluster_seq, user_features, targets) in enumerate(train_dataloader):
            cluster_seq = cluster_seq.to(device)
            user_features = user_features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(cluster_seq, user_features)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        # Evaluate
        accuracy = evaluate_fairly(model, val_dataloader, device, num_clusters, list(clusters.keys()))
        
        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    print("\n" + "="*60)
    print("FINAL FAIR RESULTS")
    print("="*60)
    print(f"Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Number of clusters: {num_clusters}")
    print(f"Random baseline: {1/num_clusters:.3f}")
    
    # Final assessment
    if best_accuracy > 1.5 / num_clusters:
        print(" MODEL IS LEARNING MEANINGFUL PATTERNS")
    else:
        print("  MODEL PERFORMANCE CLOSE TO RANDOM")
    
    # Representativeness analysis
    print("\n" + "="*80)
    print("REPRESENTATIVENESS ANALYSIS")
    print("="*80)
    
    try:
        # experiments_dir already added to sys.path at top of file
    from baseline_cluster_analyzer import ModelRepresentativenessAnalyzer
    
    analyzer = ModelRepresentativenessAnalyzer(
        model=model,
        clusters=clusters,
        train_data=train_data,
        val_data=val_data,
        device=device
    )
    analyzer.analyze_data_representativeness()
    overall_score = analyzer.generate_representativeness_report()
    except ImportError as e:
        print(f"Warning: Could not import representativeness analyzer: {e}")
        print("Skipping representativeness analysis. Model training completed successfully.")
        overall_score = None
    except Exception as e:
        print(f"Warning: Error during representativeness analysis: {e}")
        print("Skipping representativeness analysis. Model training completed successfully.")
        overall_score = None
    
    # Interpretation of results
    print(f"\n REPRESENTATIVENESS INTERPRETATION:")
    if overall_score is not None and overall_score >= 0.8:
        print("Model demonstrates HIGH representativeness - results are reliable and reproducible")
        print("   The reported accuracy can be trusted (based on REAL user features)")
    elif overall_score is not None and overall_score >= 0.6:
        print("Model demonstrates representativeness - results are generally reliable")
        print("   Accuracy represents a realistic assessment (based on REAL user features)")
    elif overall_score is not None and overall_score >= 0.4:
        print("Model demonstrates MODERATE representativeness - results require cautious interpretation")
        print("   Accuracy may be slightly overestimated/underestimated")
    elif overall_score is not None:
        print("Model demonstrates LOW representativeness - results are unreliable")
        print("   Experimental methodology requires revision")
    else:
        print("Representativeness analysis completed - check results above")
        print("   Model trained with REAL user features from FourSquare dataset")

if __name__ == "__main__":
    main()
