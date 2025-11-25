# Research Integration Summary: POI Recommendation and Knowledge Graph Studies

## Overview

This research project encompasses two complementary studies on restaurant/POI recommendation systems:

1. **POI Recommendation with Real User Features:** Deep learning-based sequence modeling using FourSquare check-in data
2. **Knowledge Graph-Based Restaurant Recommendation:** Graph-based recommendation without user data using manually curated restaurant dataset

Both studies contribute to the broader goal of developing effective recommendation systems for location-based services, each addressing different aspects of the recommendation problem.

---

## Study 1: POI Recommendation with Real User Features

### Dataset
- **Source:** FourSquare Moscow POI (w11wo/FourSquare-Moscow-POI)
- **Size:** 52,972 user check-in sequences
- **POIs:** 600 top POIs (26.1% coverage)
- **Clusters:** 12 semantic categories
- **User Features:** Real behavioral patterns extracted from sequences

### Models
1. **Baseline Cluster:** 18.7% accuracy, 0.921 representativeness
2. **Baseline Hybrid:** 16.8% accuracy, 0.946 representativeness
3. **Improved Cluster:** 24.11% accuracy, 0.804 representativeness, 37.8% Hits@3, 52.1% Hits@5
4. **Improved Hybrid:** 19.59% accuracy, 0.855 representativeness, 40.4% Hits@3, 59.4% Hits@5

### Key Contributions
- Real user feature extraction from behavioral patterns
- Neighbor-aware semantic clustering
- Comprehensive representativeness analysis
- User-based train/validation split

---

## Study 2: Knowledge Graph-Based Restaurant Recommendation

### Dataset
- **Source:** FourSquare Open Street Places (S3, September 2025)
- **Initial:** 60,964 Moscow establishments
- **Filtered:** 8,541 quality establishments
- **Selected:** 1,355 → 100 → **98 final establishments**
- **Enrichment:** Manual data collection from Яндекс Еда
- **Attributes:** 32 manually verified attributes per establishment

### Knowledge Graph
- **Nodes:** 176 (98 restaurants, 3 price levels, ~50 areas, ~15 atmospheres, 9 features)
- **Edges:** 623 (has_price_level, located_in, has_atmosphere, offers_feature)
- **Library:** NetworkX (Python)

### Recommendation Schemes
1. **Content-Based:** Precision@5 = 1.0000, Recall@5 = 0.3333, Personalization = 0.7045
2. **Geographically Weighted:** Precision@5 = 0.7163, Geo Relevance = 0.4145
3. **Hybrid:** Precision@5 = 0.7755, NDCG@5 = 0.9188, Geo Relevance = 0.3452

### Key Contributions
- Systematic data collection from FourSquare S3
- Quality-based filtering and scoring algorithm
- Geographic clustering for balanced distribution
- Manual enrichment for data accuracy
- Knowledge graph construction without user data
- Three recommendation schemes with comprehensive evaluation

---

## Integration Points

### Complementary Approaches

**Study 1 (Deep Learning):**
- **Strengths:** User behavior modeling, sequence learning, temporal patterns
- **Limitations:** Requires user interaction data, complex training

**Study 2 (Knowledge Graph):**
- **Strengths:** No user data required, interpretable, attribute-based
- **Limitations:** No user preferences, smaller dataset, manual enrichment

### Combined Value

1. **Methodological Diversity:** Two different approaches to the same problem
2. **Data Sources:** Different datasets (check-ins vs. place data)
3. **Evaluation Perspectives:** User-based vs. content-based evaluation
4. **Practical Applications:** Different use cases (with/without user data)

---

## Research Declaration: Data Collection Process

### FourSquare S3 Data Extraction

**Source Declaration:**
- **Dataset:** FourSquare Open Street Places
- **Snapshot:** September 9, 2025
- **Location:** `s3://fsq-os-places-us-east-1/release/dt=2025-09-09/`
- **Access Method:** DuckDB direct S3 Parquet reading (public access)
- **Initial Extraction:** 60,964 establishments in Moscow

**Extraction Process:**
1. Connected to S3 bucket using DuckDB
2. Queried Parquet files directly (no download required)
3. Filtered by location (Moscow: locality, region, admin_region)
4. Extracted key fields: ID, name, coordinates, address, contacts, categories

### Quality Filtering Algorithm

**Step 1: Category-Based Filtering**
- Target: `Dining and Drinking` category
- Keywords: restaurant, bar, cafe, coffee, pub, bistro, steakhouse, gastropub
- Result: 12,762 food establishments

**Step 2: Fast Food Exclusion**
- Blacklist: 20+ fast food chain names (KFC, McDonald's, Додо пицца, etc.)
- Low-quality keywords: столовая, кафетерий, фудкорт, fast food
- Result: 8,541 quality establishments

**Step 3: Scoring Algorithm**
```python
Score Components:
- Category quality: 7-20 points (restaurant=15, steakhouse=18, bar=12, etc.)
- Contact information: 5-8 points per field (website, phone, Instagram)
- Premium indicators: +8 to +10 points (авторск, гастропаб, premium, luxury)
- Chain penalties: -5 points (сеть, chain, филиал, №1, №2)
Score Range: 8-44 points
```

**Step 4: Geographic Clustering**
- Method: K-Means (50 clusters, random_state=42)
- Purpose: Balanced spatial distribution
- Implementation: Scikit-learn KMeans on latitude/longitude
- Result: 44 clusters represented in final selection

**Step 5: Multi-Stage Selection**
- Stage 1: Top 1,355 establishments (balanced: 500 restaurants, 500 bars, 355 coffee)
- Stage 2: Top 100 by score for manual enrichment
- Stage 3: Final 98 after data cleaning

### Manual Data Enrichment Process

**Source:** Яндекс Еда (Yandex Food) - https://eda.yandex.ru

**Enrichment Method:**
1. For each of top 100 establishments:
   - Opened Яндекс Еда page
   - Manually collected and verified information
   - Cross-referenced with FourSquare data
   - Verified contact information

**Attributes Collected (32 total):**

**Basic (4):** fsq_place_id, name, address, coordinates

**Contact (3):** tel, website, instagram

**Categorical (5):** type, price_level, atmosphere, metro, is_central

**Features (9):** terrace, parking, menu_vegan, menu_seasonal, menu_grill, menu_kids, menu_diet, menu_exotic, menu_hot_dogs

**Pricing (3):** min_price, max_price, avg_price_calculated

**Temporal (4):** working_hours, open_time, close_time, hours_of_operation

**Quality (4):** rating, score, atmosphere_score, geo_cluster

**Time Investment:** Extensive manual curation process (estimated 2-3 hours per establishment for top 100)

**Quality Assurance:**
- All data manually verified
- Contact information cross-checked
- Coordinates validated
- Pricing information verified
- Operating hours confirmed

### Data Reduction Summary

| Stage | Count | Description |
|-------|-------|-------------|
| S3 Dataset | 60,964 | All Moscow establishments |
| Category Filter | 12,762 | Food-related (79.1% reduction) |
| Quality Filter | 8,541 | Excluding fast food (33.1% reduction) |
| Top 1500 | 1,355 | Balanced selection with geo-clustering (84.1% reduction)<br/>[DataLens Dashboard](https://datalens.yandex/w3048zrqxgtyh) |
| Top 100 | 100 | For enrichment (92.6% reduction) |
| **Final Dataset** | **98** | **After cleaning (99.84% total reduction)** |

### Knowledge Graph Construction

**Process:**
1. Load cleaned dataset (98 establishments, 32 attributes)
2. Initialize NetworkX DiGraph
3. Create restaurant nodes with all attributes
4. Create attribute nodes (PriceLevel, Area, Atmosphere, Feature)
5. Create edges representing relationships
6. Result: 176 nodes, 623 edges

**Node Creation Details:**
- Each restaurant: 1 node with 15+ attributes
- Price levels: 3 nodes (low, mid, high)
- Metro areas: ~50 nodes (unique stations)
- Atmosphere: ~15 nodes (unique scores 74-100)
- Features: 9 nodes (terrace, parking, 7 menu options)

**Edge Creation Details:**
- Each restaurant → 1 price level edge
- Each restaurant → 1 metro area edge
- Each restaurant → 1 atmosphere edge
- Each restaurant → 0-9 feature edges (depending on features)

**Total:** 98 restaurants × (1 + 1 + 1 + avg 3.4 features) ≈ 623 edges

---

## Research Contributions

### Methodological Contributions

1. **Systematic Data Collection:** Multi-stage pipeline from 60K+ to 98 high-quality establishments
2. **Quality Scoring Algorithm:** Transparent, reproducible scoring system
3. **Geographic Clustering:** Balanced spatial distribution methodology
4. **Manual Enrichment Process:** Detailed documentation of manual data collection
5. **Knowledge Graph Construction:** NetworkX-based implementation (alternative to GP 2)
6. **Evaluation Without User Data:** Comprehensive metrics for content-based evaluation

### Practical Contributions

1. **Moscow Restaurant Dataset:** Curated dataset of 98 establishments with 32 attributes
2. **Knowledge Graph:** 176-node, 623-edge graph for recommendation
3. **Recommendation System:** Three working schemes with evaluation
4. **Reproducible Pipeline:** Complete code and documentation

### Academic Contributions

1. **Comparison with Original Research:** Adaptation of GP 2 methodology to NetworkX
2. **No-User-Data Evaluation:** Methods for evaluating without user interactions
3. **Manual Enrichment Documentation:** Transparent process documentation
4. **Dual Study Approach:** Two complementary recommendation methodologies

---

## Declaration of Data Collection Process

### Ethical Declaration

**Data Sources:**
- FourSquare Open Street Places: Public dataset, open access
- Яндекс Еда: Publicly available restaurant information
- No personal data collected
- No user tracking or behavior monitoring

**Collection Methods:**
- Automated: S3 data extraction via DuckDB
- Automated: Quality filtering and scoring
- Automated: Geographic clustering
- **Manual:** Data enrichment from Яндекс Еда (public information only)

**Usage:**
- Research and academic purposes only
- Proper attribution to data sources
- No commercial use without permission

### Reproducibility Declaration

**Code Availability:**
- Data collection notebook: `notebooks/frsqr_2025.ipynb`
- Knowledge graph notebook: `notebooks/recsys_knowledge_graph.ipynb`
- Final dataset: `data/processed/workingrest.csv`
- Additional datasets: `data/processed/moscow_*.csv`

**Parameters Documented:**
- All scoring weights
- Clustering parameters (50 clusters, random_state=42)
- Selection criteria
- Graph construction logic

**Data Access:**
- FourSquare S3: Public access (no authentication)
- Final dataset: Available in repository
- Manual enrichment: Process fully documented

---

## Integration into Single Research Framework

### Unified Research Statement

This research presents a comprehensive evaluation of restaurant/POI recommendation systems through two complementary approaches:

1. **Deep Learning Approach:** Sequence-based recommendation using real user behavioral features from FourSquare check-in data, achieving 24.11% accuracy with the Improved Cluster Model.

2. **Knowledge Graph Approach:** Attribute-based recommendation using a manually curated dataset of 98 Moscow restaurants, achieving 77.55% precision@5 with the Hybrid scheme, without requiring user interaction data.

Both studies contribute to the understanding of recommendation system design, evaluation methodologies, and practical implementation strategies for location-based services.

### Key Research Questions Addressed

1. **How to collect and curate high-quality restaurant data?**
   - Answer: Multi-stage filtering, quality scoring, geographic clustering, manual enrichment

2. **How to build recommendation systems without user data?**
   - Answer: Knowledge graph with attribute similarity, geographic proximity, and rating similarity

3. **How to evaluate recommendation systems without ground truth?**
   - Answer: Precision@K, Recall@K, NDCG, Geo Relevance, Attribute Coverage, Personalization

4. **How do different approaches compare?**
   - Answer: Deep learning excels with user data; knowledge graphs work without user data

---

## Publication Readiness

### Documentation Completeness

✅ **Data Collection:** Fully documented (S3 extraction, filtering, scoring, clustering)  
✅ **Manual Enrichment:** Process described in detail  
✅ **Knowledge Graph:** Construction methodology documented  
✅ **Recommendation Schemes:** All three schemes implemented and evaluated  
✅ **Evaluation Metrics:** Comprehensive metrics with results  
✅ **Comparison:** Comparison with original research  
✅ **Reproducibility:** All parameters and code documented  
✅ **Ethics:** Data sources and usage declared  

### Research Contributions

1. **Novel Dataset:** 98 manually curated Moscow restaurants with 32 attributes
2. **Knowledge Graph:** 176-node, 623-edge graph construction methodology
3. **Evaluation Framework:** Metrics for no-user-data scenarios
4. **Dual Study:** Integration of two complementary approaches

### Ready for Publication

- Complete methodology documentation
- Reproducible code and data
- Comprehensive evaluation results
- Ethical considerations addressed
- Comparison with related work
- Future work directions identified

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025  
**Status:** Ready for research integration

