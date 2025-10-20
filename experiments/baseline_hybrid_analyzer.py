"""
Hybrid Model Representativeness Analyzer
Analyzer for HybridPersonalizedGRU model with temporal features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.calibration import calibration_curve
from scipy.stats import entropy
import torch
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridModelRepresentativenessAnalyzer:
    def __init__(self, model, clusters, train_data, val_data, device):
        self.model = model
        self.clusters = clusters
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.cluster_names = list(clusters.keys())
        
    def analyze_data_representativeness(self):
        """Analysis of data representativeness"""
        print("\n" + "="*60)
        print("HYBRID MODEL REPRESENTATIVENESS ANALYSIS")
        print("="*60)
        
        # 1. Data distribution analysis
        self._analyze_data_distribution()
        
        # 2. User representativeness analysis
        self._analyze_user_representativeness()
        
        # 3. Temporal patterns analysis
        self._analyze_temporal_patterns()
        
        # 4. Spatial patterns analysis
        self._analyze_spatial_patterns()
        
        # 5. Model behavior analysis
        self._analyze_model_behavior()
        
        # 6. Generalization analysis
        self._analyze_generalization()
        
    def _analyze_data_distribution(self):
        """Analyze data distribution"""
        print("\n1. DATA DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        # Distribution across clusters in train and val
        train_targets = [ex['target_cluster'] for ex in self.train_data]
        val_targets = [ex['target_cluster'] for ex in self.val_data]
        
        train_counts = Counter(train_targets)
        val_counts = Counter(val_targets)
        
        print("Cluster distribution:")
        for cluster_idx in range(len(self.cluster_names)):
            train_count = train_counts.get(cluster_idx, 0)
            val_count = val_counts.get(cluster_idx, 0)
            
            train_pct = train_count / len(train_targets) * 100
            val_pct = val_count / len(val_targets) * 100
            
            diff_pct = abs(train_pct - val_pct)
            
            status = "OK" if diff_pct < 5 else "WARN" if diff_pct < 10 else "BAD"
            
            print(f"  {self.cluster_names[cluster_idx]:<15}: "
                  f"Train {train_pct:5.1f}% | Val {val_pct:5.1f}% | Diff {diff_pct:4.1f}% {status}")
        
        # Sequence statistics
        train_seq_lens = [len(ex['cluster_sequence']) for ex in self.train_data]
        val_seq_lens = [len(ex['cluster_sequence']) for ex in self.val_data]
        
        print(f"\nSequence length statistics:")
        print(f"  Train: min={min(train_seq_lens)}, max={max(train_seq_lens)}, "
              f"avg={np.mean(train_seq_lens):.1f}")
        print(f"  Val:   min={min(val_seq_lens)}, max={max(val_seq_lens)}, "
              f"avg={np.mean(val_seq_lens):.1f}")
        
        # KL divergence of distributions
        train_dist = [train_counts.get(i, 0) / len(train_targets) for i in range(len(self.cluster_names))]
        val_dist = [val_counts.get(i, 0) / len(val_targets) for i in range(len(self.cluster_names))]
        
        # Add small epsilon to avoid log(0)
        train_dist = [max(d, 1e-10) for d in train_dist]
        val_dist = [max(d, 1e-10) for d in val_dist]
        
        kl_div = entropy(train_dist, val_dist)
        print(f"  KL Divergence: {kl_div:.4f} ({'Good' if kl_div < 0.1 else 'Warning' if kl_div < 0.3 else 'High'})")
    
    def _analyze_user_representativeness(self):
        """Analyze user representativeness"""
        print("\n2. USER REPRESENTATIVENESS ANALYSIS")
        print("-" * 40)
        
        train_users = set(ex['user_id'] for ex in self.train_data if ex['user_id'] is not None)
        val_users = set(ex['user_id'] for ex in self.val_data if ex['user_id'] is not None)
        
        overlap = train_users.intersection(val_users)
        overlap_pct = len(overlap) / len(val_users) * 100 if val_users else 0
        
        print(f"User statistics:")
        print(f"  Train users: {len(train_users)}")
        print(f"  Val users: {len(val_users)}")
        print(f"  User overlap: {len(overlap)} ({overlap_pct:.1f}%) {'' if overlap_pct == 0 else ' CRITICAL'}")
        
        # User activity
        train_user_activity = Counter(ex['user_id'] for ex in self.train_data if ex['user_id'] is not None)
        val_user_activity = Counter(ex['user_id'] for ex in self.val_data if ex['user_id'] is not None)
        
        print(f"\nUser activity:")
        if train_user_activity:
            print(f"  Train - Avg sequences per user: {np.mean(list(train_user_activity.values())):.1f}")
        if val_user_activity:
            print(f"  Val   - Avg sequences per user: {np.mean(list(val_user_activity.values())):.1f}")
        
        # Check for cold users
        cold_users_val = len([uid for uid, count in val_user_activity.items() if count <= 2])
        cold_pct = cold_users_val / len(val_users) * 100 if val_users else 0
        
        print(f"  Cold users in val (≤2 sequences): {cold_users_val} ({cold_pct:.1f}%)")
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns"""
        print("\n3. TEMPORAL PATTERNS ANALYSIS")
        print("-" * 40)
        
        # Analyze sequence length as proxy for temporal patterns
        train_sequences = [ex['cluster_sequence'] for ex in self.train_data]
        val_sequences = [ex['cluster_sequence'] for ex in self.val_data]
        
        # Unique transition patterns
        def get_transition_patterns(sequences):
            patterns = Counter()
            for seq in sequences:
                if len(seq) >= 2:
                    for i in range(len(seq)-1):
                        patterns[(seq[i], seq[i+1])] += 1
            return patterns
        
        train_transitions = get_transition_patterns(train_sequences)
        val_transitions = get_transition_patterns(val_sequences)
        
        print(f"Transition patterns:")
        print(f"  Unique transitions in train: {len(train_transitions)}")
        print(f"  Unique transitions in val: {len(val_transitions)}")
        
        # Common transitions
        common_transitions = set(train_transitions.keys()).intersection(set(val_transitions.keys()))
        common_pct = len(common_transitions) / len(val_transitions) * 100 if val_transitions else 0
        
        print(f"  Common transitions: {len(common_transitions)} ({common_pct:.1f}%)")
        
        # Top transitions
        print(f"\nTop 5 transitions in train:")
        for (src, dst), count in train_transitions.most_common(5):
            if src < len(self.cluster_names) and dst < len(self.cluster_names):
                print(f"  {self.cluster_names[src]} → {self.cluster_names[dst]}: {count}")
    
    def _analyze_spatial_patterns(self):
        """Analyze spatial patterns through embeddings"""
        print("\n4. SPATIAL PATTERNS ANALYSIS")
        print("-" * 40)
        
        # Analyze cluster embeddings
        cluster_embeddings = self.model.cluster_embedding.weight.data.cpu().numpy()
        
        # PCA for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(cluster_embeddings[:-1])  # Exclude UNK token
        
        print(f"Cluster embedding analysis:")
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Cluster coherence (distances between embeddings)
        similarities = cosine_similarity(cluster_embeddings[:-1])
        np.fill_diagonal(similarities, 0)  # Exclude self-similarities
        
        avg_similarity = similarities.mean()
        print(f"  Average inter-cluster similarity: {avg_similarity:.3f}")
        print(f"  Similarity range: {similarities.min():.3f} - {similarities.max():.3f}")
    
    def _analyze_model_behavior(self):
        """Analyze model behavior"""
        print("\n5. MODEL BEHAVIOR ANALYSIS")
        print("-" * 40)
        
        self.model.eval()
        
        # Analyze model confidence
        train_confidences = []
        val_confidences = []
        
        with torch.no_grad():
            # Train confidence
            for batch in self._create_batches(self.train_data, batch_size=64):
                cluster_seq, user_features, targets = batch
                cluster_seq = cluster_seq.to(self.device)
                user_features = user_features.to(self.device)
                
                # Create temporal features for hybrid model
                batch_size, seq_len = cluster_seq.size()
                temporal_features = torch.randn(batch_size, seq_len, 4).to(self.device)
                
                outputs = self.model(cluster_seq, user_features, temporal_features)
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probabilities, dim=1)
                train_confidences.extend(max_probs.cpu().numpy())
            
            # Val confidence
            for batch in self._create_batches(self.val_data, batch_size=64):
                cluster_seq, user_features, targets = batch
                cluster_seq = cluster_seq.to(self.device)
                user_features = user_features.to(self.device)
                
                # Create temporal features for hybrid model
                batch_size, seq_len = cluster_seq.size()
                temporal_features = torch.randn(batch_size, seq_len, 4).to(self.device)
                
                outputs = self.model(cluster_seq, user_features, temporal_features)
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probabilities, dim=1)
                val_confidences.extend(max_probs.cpu().numpy())
        
        print(f"Model confidence analysis:")
        print(f"  Train - Avg confidence: {np.mean(train_confidences):.3f}")
        print(f"  Val   - Avg confidence: {np.mean(val_confidences):.3f}")
        print(f"  Confidence difference: {abs(np.mean(train_confidences) - np.mean(val_confidences)):.3f}")
        
        # Calibration analysis
        all_val_probs = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch in self._create_batches(self.val_data, batch_size=64):
                cluster_seq, user_features, targets = batch
                cluster_seq = cluster_seq.to(self.device)
                user_features = user_features.to(self.device)
                targets = targets.to(self.device)
                
                # Create temporal features for hybrid model
                batch_size, seq_len = cluster_seq.size()
                temporal_features = torch.randn(batch_size, seq_len, 4).to(self.device)
                
                outputs = self.model(cluster_seq, user_features, temporal_features)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_val_probs.extend(probabilities.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
        
        all_val_probs = np.array(all_val_probs)
        all_val_targets = np.array(all_val_targets)
        
        # Calibration for each class
        calibration_errors = []
        for class_idx in range(len(self.cluster_names)):
            class_probs = all_val_probs[:, class_idx]
            class_targets = (all_val_targets == class_idx).astype(int)
            
            if len(np.unique(class_targets)) > 1:  # Need both classes
                try:
                    prob_true, prob_pred = calibration_curve(class_targets, class_probs, n_bins=5)
                    calibration_error = np.mean(np.abs(prob_true - prob_pred))
                    calibration_errors.append(calibration_error)
                except:
                    pass
        
        avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 0
        print(f"  Average calibration error: {avg_calibration_error:.3f} "
              f"({'Good' if avg_calibration_error < 0.1 else 'Fair' if avg_calibration_error < 0.2 else 'Poor'})")
    
    def _analyze_generalization(self):
        """Analyze generalization ability"""
        print("\n6. GENERALIZATION ANALYSIS")
        print("-" * 40)
        
        # Calculate accuracy by classes
        self.model.eval()
        
        train_class_correct = [0] * len(self.cluster_names)
        train_class_total = [0] * len(self.cluster_names)
        val_class_correct = [0] * len(self.cluster_names)
        val_class_total = [0] * len(self.cluster_names)
        
        with torch.no_grad():
            # Train accuracy by classes
            for batch in self._create_batches(self.train_data, batch_size=64):
                cluster_seq, user_features, targets = batch
                cluster_seq = cluster_seq.to(self.device)
                user_features = user_features.to(self.device)
                targets = targets.to(self.device)
                
                # Create temporal features for hybrid model
                batch_size, seq_len = cluster_seq.size()
                temporal_features = torch.randn(batch_size, seq_len, 4).to(self.device)
                
                outputs = self.model(cluster_seq, user_features, temporal_features)
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(len(self.cluster_names)):
                    class_mask = targets == i
                    if class_mask.any():
                        train_class_correct[i] += (predicted[class_mask] == targets[class_mask]).sum().item()
                        train_class_total[i] += class_mask.sum().item()
            
            # Val accuracy by classes
            for batch in self._create_batches(self.val_data, batch_size=64):
                cluster_seq, user_features, targets = batch
                cluster_seq = cluster_seq.to(self.device)
                user_features = user_features.to(self.device)
                targets = targets.to(self.device)
                
                # Create temporal features for hybrid model
                batch_size, seq_len = cluster_seq.size()
                temporal_features = torch.randn(batch_size, seq_len, 4).to(self.device)
                
                outputs = self.model(cluster_seq, user_features, temporal_features)
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(len(self.cluster_names)):
                    class_mask = targets == i
                    if class_mask.any():
                        val_class_correct[i] += (predicted[class_mask] == targets[class_mask]).sum().item()
                        val_class_total[i] += class_mask.sum().item()
        
        # Analyze performance gaps
        performance_gaps = []
        print("Performance gap analysis (Train vs Val):")
        
        for i in range(len(self.cluster_names)):
            if train_class_total[i] > 0 and val_class_total[i] > 0:
                train_acc = train_class_correct[i] / train_class_total[i]
                val_acc = val_class_correct[i] / val_class_total[i]
                gap = train_acc - val_acc
                performance_gaps.append(gap)
                
                status = "OK" if gap < 0.1 else "WARN" if gap < 0.2 else "BAD"
                print(f"  {self.cluster_names[i]:<15}: Train {train_acc:.3f} | Val {val_acc:.3f} | Gap {gap:+.3f} {status}")
        
        avg_gap = np.mean(performance_gaps) if performance_gaps else 0
        max_gap = max(performance_gaps) if performance_gaps else 0
        
        print(f"\nGeneralization summary:")
        print(f"  Average performance gap: {avg_gap:.3f} ({'Good' if avg_gap < 0.1 else 'Fair' if avg_gap < 0.15 else 'Overfitting'})")
        print(f"  Maximum performance gap: {max_gap:.3f}")
        
        # Overall generalization score
        generalization_score = 1.0 - min(avg_gap * 2, 1.0)  # Normalize to 0-1
        print(f"  Generalization score: {generalization_score:.3f}")
        
        return generalization_score
    
    def _create_batches(self, data, batch_size=64):
        """Create batches for analysis"""
        cluster_sequences = []
        user_features_list = []
        targets = []
        
        max_seq_len = 15
        num_clusters = len(self.cluster_names)
        
        for example in data:
            cluster_seq = example['cluster_sequence']
            if len(cluster_seq) > max_seq_len:
                cluster_seq = cluster_seq[-max_seq_len:]
            else:
                cluster_seq = [num_clusters] * (max_seq_len - len(cluster_seq)) + cluster_seq
            
            cluster_sequences.append(cluster_seq)
            user_features_list.append(example['user_features'])
            targets.append(example['target_cluster'])
            
            if len(cluster_sequences) >= batch_size:
                yield (
                    torch.LongTensor(cluster_sequences),
                    torch.FloatTensor(user_features_list),
                    torch.LongTensor(targets)
                )
                cluster_sequences = []
                user_features_list = []
                targets = []
        
        if cluster_sequences:
            yield (
                torch.LongTensor(cluster_sequences),
                torch.FloatTensor(user_features_list),
                torch.LongTensor(targets)
            )
    
    def generate_representativeness_report(self):
        """Generate final representativeness report"""
        print("\n" + "="*60)
        print("HYBRID MODEL REPRESENTATIVENESS REPORT")
        print("="*60)
        
        # Collect all metrics
        metrics = {}
        
        # 1. Data Distribution Score
        train_targets = [ex['target_cluster'] for ex in self.train_data]
        val_targets = [ex['target_cluster'] for ex in self.val_data]
        
        train_dist = [train_targets.count(i) / len(train_targets) for i in range(len(self.cluster_names))]
        val_dist = [val_targets.count(i) / len(val_targets) for i in range(len(self.cluster_names))]
        
        distribution_score = 1.0 - min(np.sum(np.abs(np.array(train_dist) - np.array(val_dist))), 1.0)
        metrics['Data Distribution'] = distribution_score
        
        # 2. User Representativeness Score
        train_users = set(ex['user_id'] for ex in self.train_data if ex['user_id'] is not None)
        val_users = set(ex['user_id'] for ex in self.val_data if ex['user_id'] is not None)
        overlap_pct = len(train_users.intersection(val_users)) / len(val_users) if val_users else 0
        user_score = 1.0 - overlap_pct
        metrics['User Separation'] = user_score
        
        # 3. Generalization Score (computed in _analyze_generalization)
        generalization_score = self._analyze_generalization()
        metrics['Generalization'] = generalization_score
        
        print("\nREPRESENTATIVENESS SCORES (0-1 scale):")
        for metric, score in metrics.items():
            status = " Excellent" if score >= 0.9 else " Good" if score >= 0.7 else "Fair" if score >= 0.5 else " Poor"
            print(f"  {metric:<20}: {score:.3f} - {status}")
        
        overall_score = np.mean(list(metrics.values()))
        overall_status = "HIGHLY REPRESENTATIVE" if overall_score >= 0.8 else "REPRESENTATIVE" if overall_score >= 0.6 else "MODERATELY REPRESENTATIVE" if overall_score >= 0.4 else "POORLY REPRESENTATIVE"
        
        print(f"\nOVERALL REPRESENTATIVENESS: {overall_score:.3f} - {overall_status}")
        
        return overall_score

def perform_hybrid_representativeness_analysis(model, clusters, train_data, val_data, device):
    """Perform representativeness analysis for hybrid model"""
    
    analyzer = HybridModelRepresentativenessAnalyzer(model, clusters, train_data, val_data, device)
    
    # Conduct all analyses
    analyzer.analyze_data_representativeness()
    
    # Generate final report
    overall_score = analyzer.generate_representativeness_report()
    
    return overall_score

if __name__ == "__main__":
    print("Hybrid Model Representativeness Analyzer")
    print("Import and use with trained hybrid models.")
