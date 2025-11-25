# Methodology

## Dataset and Preprocessing

### Data Source

**Primary Dataset:** FourSquare Moscow POI dataset (w11wo/FourSquare-Moscow-POI) contains 52,972 user check-in sequences across Moscow, Russia. Each sequence includes user ID, POI locations, timestamps, and semantic categories.

**Knowledge Graph Dataset:** A separate curated dataset of 98 high-quality restaurants and bars in Moscow was constructed for knowledge graph-based recommendation system evaluation. This dataset was collected through a multi-stage process:

1. **Initial Collection:** Extracted 60,964 establishments from FourSquare Open Street Places dataset (S3 snapshot, September 2025)
2. **Quality Filtering:** Applied scoring algorithm to identify 8,541 quality establishments (excluding fast food chains)
3. **Geographic Clustering:** Used K-Means clustering (50 clusters) to ensure balanced spatial distribution
4. **Multi-Stage Selection:** Selected top 1,355 establishments, then top 100 for manual enrichment
5. **Manual Enrichment:** Manually collected and verified 32 attributes per establishment from Яндекс Еда (Yandex Food)
6. **Final Dataset:** 98 restaurants and bars with complete attribute information

**Detailed methodology:** See `Dataset_Collection_Methodology.md` for complete documentation of the data collection, filtering, scoring, geographic clustering, manual enrichment, and knowledge graph construction process.

### POI Selection and Clustering
We selected the top 600 POIs based on visit frequency, covering 26.1% of total visits. POIs were grouped into 12 balanced semantic clusters:
- Food & Dining (50 POIs)
- Shopping & Retail (50 POIs)
- Entertainment (50 POIs)
- Transportation (50 POIs)
- Landmarks & Parks (50 POIs)
- Business & Work (50 POIs)
- Health & Fitness (50 POIs)
- Education (50 POIs)
- Nightlife (50 POIs)
- Cultural (50 POIs)
- Services (50 POIs)
- Miscellaneous (50 POIs)

### User Feature Extraction
Real user features were extracted from actual visit patterns:
1. **Cluster Distribution**: Frequency of visits to each semantic cluster
2. **Visit Frequency**: Normalized total visit count
3. **Category Diversity**: Shannon entropy of visited categories
4. **Dominant Category**: Concentration of visits to most frequent cluster

### Data Split
Users were randomly split 80/20 for training/validation with zero overlap to ensure proper evaluation and prevent data leakage.

## Model Architectures

### Baseline Cluster Model
Traditional cluster-based recommendation using:
- LSTM for sequence modeling
- Synthetic user features (randomly generated)
- Cluster embeddings (32 dimensions)
- User embeddings (16 dimensions)

### Baseline Hybrid Model
Combined approach using:
- Cluster sequence modeling
- Synthetic user profiles
- Temporal features (4 dimensions)
- Multi-modal fusion architecture

### Improved Cluster Model
Enhanced cluster-based approach with:
- Real user features from FourSquare data
- **Neighbor-aware semantic clustering** (preserves semantic relationships)
- Advanced sequence modeling
- Real behavioral pattern integration

### Improved Hybrid Model
Advanced hybrid architecture incorporating:
- Authentic user behavioral features
- Temporal pattern modeling
- Spatial cluster relationships
- Multi-dimensional user profiling

## Knowledge Graph Construction (Separate Study)

A parallel study was conducted using a manually curated dataset of 98 Moscow restaurants and bars to construct a knowledge graph-based recommendation system. This study implements three recommendation schemes (Content-Based, Geographically Weighted, and Hybrid) without user interaction data, focusing on restaurant-restaurant similarity.

**Key Components:**
- **Dataset:** 98 establishments manually enriched with 32 attributes from Яндекс Еда
- **Graph Structure:** 176 nodes, 623 edges (NetworkX)
- **Schemes:** Content-based, Geo-weighted, Hybrid
- **Evaluation:** Precision@K, Recall@K, NDCG, Geo Relevance, Attribute Coverage, Personalization

**Detailed Methodology:** See `Knowledge_Graph_Methodology.md` and `Dataset_Collection_Methodology.md` for complete documentation.

## Training Configuration

### Hyperparameters
- Improved Cluster: LR=0.001, Batch=32, Epochs=10 (Adam, StepLR 8/0.9)
- Improved Hybrid: LR=0.0008, Batch=48, Epochs=20 (Adam, StepLR 5/0.85)
- Hidden Dimensions: 64 (bidirectional GRU)
- Dropout: 0.25 on MLP/projection layers
- Optimizer: Adam + weight decay 1e-4

### Loss Functions
- Cross-entropy for classification
- Personality-aware loss for improved models
- Regularization: L2 with weight decay

## Evaluation Methodology

### Metrics
1. **Accuracy**: Top-1 recommendation correctness
2. **Hits@K**: Success rate in top-K recommendations
3. **Representativeness Score**: Comprehensive evaluation of model reliability

### Representativeness Analysis
Six-dimensional assessment framework:

#### 1. Data Distribution Analysis
- Cluster balance between train/validation sets
- Sequence length distribution
- KL divergence measurement

#### 2. User Representativeness Analysis
- User overlap verification (target: 0%)
- Activity pattern comparison
- Cold user identification

#### 3. Temporal Patterns Analysis
- Transition pattern consistency
- Sequence behavior validation
- Temporal stability assessment

#### 4. Spatial Patterns Analysis
- Cluster embedding analysis using PCA
- Inter-cluster similarity measurement
- Spatial relationship validation

#### 5. Model Behavior Analysis
- Confidence calibration
- Prediction stability
- Model uncertainty assessment

#### 6. Generalization Analysis
- Train/validation performance gaps
- Class-wise generalization
- Overfitting detection

### Statistical Analysis
- Performance comparisons using paired t-tests
- Confidence intervals for all metrics
- Effect size calculations
- Multiple comparison corrections

## Experimental Design

### Controlled Variables
- Dataset: FourSquare Moscow POI
- POI count: 600
- Cluster count: 12
- Random seed: 42
- Hardware: CPU-based training

### Independent Variables
- Model architecture (Cluster vs Hybrid)
- User feature type (Synthetic vs Real)
- Implementation approach (Baseline vs Improved)

### Dependent Variables
- Accuracy
- Representativeness score
- Hits@3 and Hits@5
- Training time
- Model stability

## Validation Protocol

### Cross-Validation
- User-based splits to prevent leakage
- Temporal validation for sequence models
- Bootstrap sampling for confidence intervals

### Statistical Significance
- Minimum 95% confidence level
- Effect size threshold: Cohen's d > 0.2
- Multiple comparison corrections applied

### Reproducibility
- Fixed random seeds
- Version-controlled dependencies
- Automated experiment execution
- Comprehensive logging

## Implementation Details

### Software Environment
- Python 3.11
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn for visualization

### Hardware Requirements
- CPU: Multi-core processor
- RAM: 8GB minimum
- Storage: 2GB for dataset and models

### Execution Pipeline
1. Data loading and preprocessing
2. Model initialization and training
3. Evaluation and metric computation
4. Representativeness analysis
5. Result aggregation and visualization
