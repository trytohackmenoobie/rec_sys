# Knowledge Graph-Based Restaurant Recommendation: Methodology

## Abstract

This document describes the methodology for constructing a knowledge graph-based restaurant recommendation system using a manually curated dataset of 98 Moscow restaurants and bars. The system implements three recommendation schemes (Content-Based, Geographically Weighted, and Hybrid) without user interaction data, focusing on restaurant-restaurant similarity based on attributes, geographic proximity, and ratings.

---

## 1. Dataset Construction

### 1.1 Data Collection Pipeline

**Stage 1: Initial Extraction (FourSquare S3)**
- **Source:** FourSquare Open Street Places, September 2025 snapshot
- **Location:** `s3://fsq-os-places-us-east-1/release/dt=2025-09-09/`
- **Method:** DuckDB direct S3 Parquet reading
- **Initial Count:** 60,964 establishments in Moscow

**Stage 2: Category Filtering**
- **Target Categories:** Dining and Drinking establishments
- **Keywords:** restaurant, bar, cafe, coffee, pub, bistro, steakhouse, gastropub
- **Result:** 12,762 food-related establishments

**Stage 3: Quality Filtering**
- **Exclusion:** Fast food chains, cafeterias, food courts
- **Scoring Algorithm:** 
  - Category quality: 7-20 points
  - Contact information: 5-8 points per field
  - Premium indicators: +8 to +10 points
  - Chain penalties: -5 points
- **Result:** 8,541 quality establishments

**Stage 4: Geographic Clustering**
- **Method:** K-Means clustering (50 clusters, random_state=42)
- **Purpose:** Ensure balanced spatial distribution
- **Result:** 44 clusters represented in final selection

**Stage 5: Multi-Stage Selection**
- **Top 1500:** 1,355 establishments selected (balanced by type and geography)
  - **Geographic filtering**: Filtering by latitude and longitude with geo-clustering to ensure weighted processing of restaurants across all districts of Moscow
  - **Visualization**: [DataLens Dashboard - 1500 Restaurants](https://datalens.yandex/w3048zrqxgtyh)
- **Top 100:** Highest scoring establishments for manual enrichment
- **Final:** 98 establishments after data cleaning

### 1.2 Manual Data Enrichment

**Source:** Яндекс Еда (Yandex Food)  
**Method:** Manual data collection and verification  
**Attributes Collected (32 total):**

**Basic Information:**
- FourSquare ID, name, address, coordinates

**Contact Information:**
- Phone, website, Instagram

**Categorical Attributes:**
- Type (Restaurant/Bar), price level, atmosphere score
- Metro station, central location flag

**Features:**
- Terrace, parking, menu options (vegan, seasonal, grill, kids, diet, exotic, hot dogs)

**Pricing:**
- Min/max price, calculated average price

**Temporal:**
- Working hours, open/close times, hours of operation

**Quality Metrics:**
- Rating (0-5), quality score (8-44), atmosphere score

### 1.3 Data Preprocessing

**Cleaning Steps:**
1. Remove rows with critical missing data (1 row removed)
2. Impute numerical values with medians
3. Impute categorical values with modes or 'N/A'
4. Convert data types (boolean, datetime, numeric)
5. Remove empty columns
6. Engineer temporal features (hours_of_operation)

**Final Dataset:**
- **98 establishments** (65 restaurants, 33 bars)
- **32 attributes** per establishment
- **100% completeness** for critical fields

---

## 2. Knowledge Graph Construction

### 2.1 Graph Architecture

**Library:** NetworkX (Python)  
**Graph Type:** Directed Graph (DiGraph)  
**Implementation:** Python-based (vs. GP 2 in original research)

**Graph Statistics:**
- **Total Nodes:** 176
- **Total Edges:** 623
- **Node Types:** 5 (Restaurant, PriceLevel, Area, Atmosphere, Feature)
- **Edge Types:** 4 (has_price_level, located_in, has_atmosphere, offers_feature)

### 2.2 Node Creation

**Restaurant Nodes (98 nodes):**
```python
G.add_node(restaurant_id,
    type='Restaurant',
    name=row['name'],
    address=row['address'],
    latitude=row['latitude'],
    longitude=row['longitude'],
    tel=row['tel'],
    website=row['website'],
    instagram=row['instagram'],
    score=row['score'],
    rating=row['rating'],
    working_hours=row['working_hours'],
    open_time=row['open_time'],
    close_time=row['close_time'],
    atmosphere_score=row['atmosphere_score'],
    avg_price_calculated=row['avg_price_calculated'],
    hours_of_operation=row['hours_of_operation'],
    is_central=row['is_central']
)
```

**PriceLevel Nodes (3 nodes):**
- Node ID: `PriceLevel_{price_level}`
- Values: low, mid, high
- Relation: `has_price_level`

**Area Nodes (~50 nodes):**
- Node ID: `Area_{metro_station}`
- Values: Metro station names (e.g., "Парк культуры", "Тверская")
- Relation: `located_in`

**Atmosphere Nodes (~15 nodes):**
- Node ID: `Atmosphere_{score_int}`
- Values: Integer atmosphere scores (74-100)
- Relation: `has_atmosphere`

**Feature Nodes (9 nodes):**
- Node ID: `Feature_{feature_name}`
- Values: Terrace, Parking, Menu Vegan, Menu Seasonal, Menu Grill, Menu Kids, Menu Diet, Menu Exotic, Menu Hot Dogs
- Relation: `offers_feature`

### 2.3 Edge Creation

**Edge Types:**
1. **has_price_level:** Restaurant → PriceLevel (98 edges)
2. **located_in:** Restaurant → Area/Metro (98 edges)
3. **has_atmosphere:** Restaurant → Atmosphere (98 edges)
4. **offers_feature:** Restaurant → Feature (~329 edges, variable)

**Total Edges:** 623

---

## 3. Similarity Functions

### 3.1 Attribute Similarity

**Components:**
1. **Price Level Similarity:** Binary (1.0 if same price level, 0.0 otherwise)
2. **Atmosphere Similarity:** Binary (1.0 if same atmosphere score, 0.0 otherwise)
3. **Feature Similarity:** Jaccard coefficient on feature sets

**Formula:**
```python
total_sim = (price_sim + atmosphere_sim + jaccard_sim) / 3.0
```

**Weighted Version:**
- Configurable feature weights (e.g., Menu Exotic: 3.0, Menu Vegan: 3.0)
- Weighted Jaccard similarity for features

### 3.2 Geographic Proximity

**Method:** Haversine distance formula
```python
R = 6371  # Earth radius in km
a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
c = 2 × atan2(√a, √(1−a))
distance = R × c
```

**Normalization:**
```python
proximity = 1 / (1 + distance * 0.02)
```

**Configurable Parameters:**
- `max_relevant_distance`: Maximum distance threshold (default: 100 km)

### 3.3 Rating Similarity

**Formula:**
```python
max_rating_diff = 5.0
similarity = 1 - (abs(rating1 - rating2) / max_rating_diff)
return max(0.0, similarity)
```

### 3.4 Metro Similarity (Enhanced)

**Logic:**
- Both have metro + both central: 1.0
- Both have metro (one/both not central): 0.7
- One has metro, one doesn't: 0.3
- Neither has metro: 0.1

---

## 4. Recommendation Schemes

### 4.1 Scheme 1: Content-Based

**Approach:** Attribute similarity only
```python
similarity_score = calculate_attribute_similarity(G, target, candidate)
```

**Characteristics:**
- Ignores geographic proximity
- Ignores ratings
- Focuses on attribute matching
- Highest personalization score

### 4.2 Scheme 2: Geographically Weighted

**Approach:** Attribute similarity + Geographic proximity
```python
combined_score = (attr_sim + geo_prox) / 2.0
```

**Characteristics:**
- Balances attributes and location
- Prioritizes nearby similar restaurants
- Highest geo-relevance score

### 4.3 Scheme 3: Hybrid

**Approach:** Weighted combination of all factors
```python
hybrid_score = (0.40 * attr_sim) + (0.30 * geo_prox) + (0.30 * rating_sim)
```

**Enhanced Version:**
- Configurable scenario weights
- Optional metro similarity component
- Dynamic distance thresholds
- Feature weights support

**Default Weights:**
- Attribute similarity: 40%
- Geographic proximity: 30%
- Rating similarity: 30%
- Metro similarity: 0% (optional)

---

## 5. Evaluation Methodology

### 5.1 Metrics

**Relevance Metrics:**
- **Precision@K:** Proportion of relevant items in top-K
- **Recall@K:** Coverage of relevant items
- **NDCG@K:** Normalized Discounted Cumulative Gain

**Diversity Metrics:**
- **Geo Relevance:** Average geographic proximity of recommendations
- **Attribute Coverage:** Number of unique attributes in recommendations
- **Personalization:** Average pairwise similarity of recommendations

### 5.2 Validation Strategy

**Hold-Out Validation:**
- 5 randomly selected target restaurants
- Calculate relevant items based on hybrid score
- Evaluate all three schemes

**Leave-One-Out Cross-Validation:**
- Each restaurant as target (98 iterations)
- More robust evaluation
- Average metrics across all targets

**Relevance Definition:**
- **Initial:** Based on hybrid score (circular dependency issue)
- **Revised:** Based on attribute similarity only (more independent)

### 5.3 Results

**Leave-One-Out Cross-Validation (Revised):**

| Scheme | Precision@5 | Recall@5 | NDCG@5 | Geo Relevance | Attribute Coverage | Personalization |
|--------|-------------|----------|--------|---------------|-------------------|-----------------|
| Content-Based | 1.0000 | 0.3333 | 1.0000 | 0.2104 | 9.0204 | 0.7045 |
| Geo-Weighted | 0.7163 | 0.2388 | 0.8686 | 0.4145 | 9.6531 | 0.5802 |
| Hybrid | 0.7755 | 0.2585 | 0.9188 | 0.3452 | 9.4286 | 0.6211 |

**Improved Leave-One-Out (with enhancements):**

| Scheme | Precision@5 | Recall@5 | NDCG@5 | Geo Relevance | Attribute Coverage | Personalization |
|--------|-------------|----------|--------|---------------|-------------------|-----------------|
| Content-Based | 0.9939 | 0.3313 | 0.9997 | 0.9367 | 9.0408 | 0.7040 |
| Geo-Weighted | 0.9776 | 0.3259 | 0.9975 | 0.9473 | 8.9796 | 0.7010 |
| Hybrid | 0.7980 | 0.2660 | 0.9611 | 0.9520 | 9.2857 | 0.6208 |

---

## 6. Comparison with Original Research

### 6.1 Key Differences

| Aspect | Original (GP 2) | This Study (NetworkX) |
|--------|-----------------|----------------------|
| **Language** | GP 2 (graph programming) | Python + NetworkX |
| **Users** | 138 users | 0 users |
| **Ratings** | 1,161 ratings | 0 ratings |
| **Dataset** | 935 restaurants | 98 restaurants/bars |
| **Enrichment** | Automated | Manual (Яндекс Еда) |
| **Schemes** | 3 with user-user similarity | 3 without user data |
| **Accuracy** | 84.97% (Scheme 3) | 77.55% (Hybrid, revised) |

### 6.2 Advantages

1. **Data Quality:** Manual enrichment ensures accuracy
2. **Rich Attributes:** 32 attributes vs. limited original
3. **Geographic Relevance:** Moscow-specific data
4. **Fresh Data:** 2025 dataset vs. older UCI data
5. **Flexibility:** Configurable weights and parameters

### 6.3 Limitations

1. **No User Data:** Cannot implement collaborative filtering
2. **Small Dataset:** 98 vs. 935 establishments
3. **Manual Process:** Time-intensive enrichment
4. **No Ground Truth:** Cannot evaluate against user preferences
5. **Single City:** Limited to Moscow

---

## 7. Implementation Details

### 7.1 Technologies

**Data Collection:**
- DuckDB for S3 Parquet reading
- Pandas for data manipulation
- Scikit-learn for clustering

**Knowledge Graph:**
- NetworkX for graph construction
- Pyvis for interactive graph visualization
- Folium for interactive maps

**Recommendation System:**
- Python 3.11+
- NumPy for numerical operations
- Math library for Haversine formula

### 7.1.1 Interactive Visualizations

Two interactive HTML visualizations have been generated and are available in `results/visualizations/`:

1. **Knowledge Graph Visualization** (`knowledge_graph_interactive.html`)
   - Interactive network visualization of the complete knowledge graph
   - Shows all 176 nodes (Restaurants, PriceLevels, Areas, Atmosphere, Features) and 623 edges
   - Color-coded by node type:
     - Blue: Restaurants
     - Green: Price Levels
     - Yellow: Areas (Metro stations)
     - Orange: Atmosphere scores
     - Pink: Features
   - Interactive features: zoom, pan, drag nodes, click to inspect
   - Generated using Pyvis library

2. **Restaurant Map Visualization** (`restaurant_map_100_interactive.html`)
   - Interactive map showing all 98 restaurants in Moscow
   - Color-coded markers by rating:
     - Green: Rating ≥ 4.5
     - Blue: Rating ≥ 4.0
     - Orange: Rating ≥ 3.5
     - Red: Rating < 3.5
   - Click markers to see: name, rating, average price, address
   - Generated using Folium library
   - Can be regenerated by running: `python scripts/generate_knowledge_graph_visualizations.py`

### 7.2 Code Structure

**Data Collection Notebook:** `notebooks/frsqr_2025.ipynb`
- S3 data extraction
- Quality filtering
- Geographic clustering
- Selection pipeline

**Knowledge Graph Notebook:** `notebooks/recsys_knowledge_graph.ipynb`
- Data cleaning and preprocessing
- Graph construction
- Interactive visualizations (pyvis network graph and folium restaurant map)
- Similarity functions with configurable weights
- Three recommendation schemes (Content-Based, Geographically Weighted, Hybrid)
- Evaluation metrics and validation strategies
- Improved schemes with feature weights, dynamic distance thresholds, and metro connectivity
- Comprehensive scenario evaluation

**Final Dataset:** `data/processed/workingrest.csv`
- 98 establishments
- 32 attributes
- Semicolon-delimited

### 7.3 Reproducibility

**Parameters:**
- Geographic clusters: 50 (K-Means, random_state=42)
- Scoring weights: Documented in code
- Selection criteria: Balanced by type and geography
- Graph construction: Deterministic node/edge creation

**Data Availability:**
- FourSquare S3: Public access
- Final dataset: `data/processed/workingrest.csv`
- Additional datasets: `data/processed/moscow_*.csv`
- Notebooks: `notebooks/frsqr_2025.ipynb`, `notebooks/recsys_knowledge_graph.ipynb`

---

## 8. Ethical Considerations

### 8.1 Data Sources

- **FourSquare:** Public Open Street Places dataset
- **Яндекс Еда:** Publicly available restaurant information
- **No Personal Data:** Only public restaurant information

### 8.2 Usage Rights

- Research and academic purposes only
- Proper attribution to data sources
- No commercial use without permission

---

## 9. Future Work

### 9.1 Dataset Expansion

1. Increase to 200-300 establishments
2. Include more cities
3. Add user interaction data (if available)

### 9.2 System Enhancements

1. Package recommendation logic into reusable module
2. Create RESTful API for production deployment
3. User preference learning
4. Temporal pattern integration
5. Multi-modal features (images, reviews)

### 9.3 Evaluation Improvements

1. User studies for ground truth
2. A/B testing framework
3. Online evaluation metrics
4. Long-term user satisfaction tracking

---

## References

1. Nitamayega et al. (2024). "Utilizing GP 2 for Restaurant Recommendation." *Ind. Journal on Computing*, Vol. 9, Issue. 1, pp. 8-19.
2. FourSquare Open Street Places: https://github.com/foursquare/fsq-os-places
3. NetworkX Documentation: https://networkx.org/
4. DuckDB Documentation: https://duckdb.org/docs/

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025

