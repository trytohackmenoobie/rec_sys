# Dataset Collection and Knowledge Graph Construction Methodology

## Abstract

This document describes the comprehensive methodology for collecting, curating, and enriching restaurant data from FourSquare Open Street Places dataset (September 2025 snapshot), followed by manual data enrichment and knowledge graph construction for restaurant recommendation system development. The process transforms raw location data from 60,964 Moscow establishments into a curated dataset of 98 high-quality restaurants with manually verified attributes, which serves as the foundation for a knowledge graph-based recommendation system.

---

## 1. Data Source and Initial Collection

### 1.1 FourSquare Open Street Places Dataset

**Source:** FourSquare Open Street Places (FSQ-OS-Places)  
**Snapshot Date:** September 9, 2025  
**Data Location:** Amazon S3 (`s3://fsq-os-places-us-east-1/release/dt=2025-09-09/`)  
**Access Method:** DuckDB with direct S3 Parquet file reading

**Dataset Structure:**
- **Places Table:** Contains 60,964 establishments in Moscow
- **Categories Table:** Hierarchical category taxonomy (6 levels)
- **Data Format:** Parquet files partitioned by date

**Key Fields Available:**
- `fsq_place_id`: Unique FourSquare identifier
- `name`: Establishment name
- `latitude`, `longitude`: Geographic coordinates
- `address`, `locality`, `region`: Location information
- `tel`, `website`, `instagram`: Contact information
- `fsq_category_ids`, `fsq_category_labels`: Category associations

### 1.2 Initial Data Extraction

**Query Strategy:**
```sql
SELECT
    fsq_place_id, name, latitude, longitude, address,
    locality, region, tel, website, instagram,
    fsq_category_labels
FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet')
WHERE locality = 'Moscow' OR region = 'Moscow'
```

**Initial Results:**
- Total establishments in Moscow: **60,964**
- Food-related establishments identified: **12,762**
- Quality establishments (after filtering): **8,541**

---

## 2. Data Filtering and Quality Assessment

### 2.1 Category-Based Filtering

**Target Categories:**
- Primary: `Dining and Drinking` (level1_category_name)
- Specific keywords: `restaurant`, `bar`, `cafe`, `coffee`, `pub`, `bistro`, `steakhouse`, `gastropub`, `wine bar`, `cocktail bar`

**Exclusion Criteria:**
Fast food chains and low-quality venues were systematically excluded:

**Fast Food Chains (Blacklist):**
- `додо пицца`, `dodo pizza`, `kfc`, `макдональд`, `mcdonald`, `бургер кинг`, `burger king`
- `субвей`, `subway`, `панда пицца`, `panda pizza`, `тарту`, `тартуга`
- `вкусно и точка`, `чикен`, `chicken`, `шаурма`, `шаверма`
- `макдоналдс`, `макдак`, `макдач`, `mcDonald`, `kfc`, `kentucky`

**Low-Quality Keywords:**
- `столовая`, `кафетерий`, `фудкорт`, `food court`, `фастфуд`, `fast food`
- `сетевое`, `сеть`, `chain`, `филиал`

**Fast Food Categories:**
- `Fast Food Restaurant`, `Food Court`, `Cafeteria`, `Corporate Cafeteria`

### 2.2 Quality Scoring Algorithm

Each establishment was scored based on multiple criteria:

**Category-Based Scoring:**
```python
premium_keywords = {
    'restaurant': 15, 'steakhouse': 18, 'bistro': 16, 'grill': 14,
    'bar': 12, 'wine bar': 17, 'cocktail bar': 16, 'pub': 10,
    'cafe': 8, 'coffee': 7, 'гастропаб': 19, 'авторск': 20
}
```

**Contact Information Scoring:**
- Website: +8 points
- Phone: +5 points
- Instagram: +6 points

**Premium Indicators:**
- Unique concepts: `авторск`, `гастропаб`, `винный`, `крафтов`, `craft`, `гастроном`, `уникальн`: +8 points
- Premium keywords: `премиум`, `premium`, `люкс`, `luxury`, `высок`, `gourmet`: +10 points

**Chain Penalties:**
- Chain indicators: `сеть`, `chain`, `филиал`, `№1`, `№2`, `№3`: -5 points

**Score Range:** 8-44 points

### 2.3 Geographic Clustering

**Method:** K-Means clustering with 50 geographic clusters  
**Purpose:** Ensure balanced spatial distribution across Moscow through filtering by latitude and longitude with geo-clustering to enable weighted processing of restaurants across all districts of Moscow  
**Implementation:**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
df['geo_cluster'] = kmeans.fit_predict(coords)
```

**Geographic Filtering Strategy:**
- Filtering by latitude and longitude coordinates
- Geo-clustering to ensure weighted representation across all Moscow districts
- Balanced selection prevents over-representation of central districts
- Each cluster represents a distinct geographic area of Moscow

**Results:**
- 50 geographic clusters created
- 44 clusters represented in final selection
- Maximum establishments per cluster: 60
- Minimum establishments per cluster: 1
- Balanced geographic distribution visualized in [DataLens Dashboard](https://datalens.yandex/w3048zrqxgtyh)

---

## 3. Multi-Stage Selection Process

### 3.1 Stage 1: Top 500 Selection

**Target Distribution:**
- Restaurants: 200
- Bars: 150
- Coffee/Cafes: 150

**Selection Criteria:**
1. Score-based ranking within each type
2. Geographic diversity (max per cluster: 25 restaurants, 20 bars, 15 coffee)
3. Contact information availability

**Result:** 500 establishments selected

### 3.2 Stage 2: Top 1500 Enhanced Selection

**Enhanced Criteria:**
- Increased target: 1500 establishments
- Balanced distribution: 500 restaurants, 500 bars, 500 coffee
- Stricter geographic control: max 25/20/15 per cluster per type
- Higher quality threshold
- **Geographic filtering**: Filtering by latitude and longitude with geo-clustering to ensure weighted processing of restaurants across all districts of Moscow

**Visualization:**
- **DataLens Dashboard**: [1500 Restaurants Visualization](https://datalens.yandex/w3048zrqxgtyh) - Interactive map showing all selected establishments with geographic clustering

**Final Statistics:**
- Selected: **1,355 establishments** (target 1500 not fully met due to quality constraints)
- Distribution: 500 restaurants, 500 bars, 355 coffee
- Average score: 23.75
- Score range: 8-44
- Contact information coverage:
  - Websites: 75.9%
  - Phones: 79.3%
  - Instagram: 27.2%

### 3.3 Stage 3: Top 100 for Manual Enrichment

**Selection:** Top 100 establishments by score from the 1,355 pool

**Top 15 Establishments (by score):**
1. G&T Gourmet (Restaurant) - 44 points
2. Винный базар (Restaurant) - 42 points
3. Støy Craft Bar (Restaurant) - 42 points
4. Crafter Bar (Bar) - 39 points
5. Невинный (Bar) - 39 points
6. HopHead Craft Beer Pub (Bar) - 39 points
7. Винный базар (Bar) - 39 points
8. Goos'to wine, винный бар (Bar) - 39 points
9. Винный базар (Bar) - 39 points
10. Howard Loves Craft (Bar) - 39 points
11. Craft Kitchen (Restaurant) - 37 points
12. Crafted Grill Bar (Restaurant) - 36 points
13. № 13 (Restaurant) - 34 points
14. Pinzeria by Bontempi (Restaurant) - 34 points
15. Cavina (Restaurant) - 34 points

**Geographic Coverage:**
- Unique geo-clusters in top-100: 44
- Average latitude: 55.759856
- Average longitude: 37.608681

---

## 4. Manual Data Enrichment Process

### 4.1 Data Source for Enrichment

**Primary Source:** Яндекс Еда (Yandex Food)  
**Enrichment Method:** Manual data collection and verification  
**Time Investment:** Extensive manual curation process

### 4.2 Enrichment Fields

For each of the top 100 establishments, the following attributes were manually collected:

**Basic Information:**
- `fsq_place_id`: FourSquare identifier (preserved)
- `name`: Establishment name (verified)
- `address`: Full address (verified)
- `latitude`, `longitude`: Coordinates (verified)

**Contact Information:**
- `tel`: Phone number (verified/updated)
- `website`: Official website (verified/updated)
- `instagram`: Instagram handle (verified/updated)

**Categorical Attributes:**
- `type`: Restaurant/Bar (verified)
- `price_level`: low/mid/high (manually assigned)
- `atmosphere`: Score 0-100% (manually assessed)
- `metro`: Nearest metro station (manually identified)
- `is_central`: Boolean (manually determined)

**Features:**
- `terrace`: Boolean (summer/none/No)
- `parking`: Boolean (True/False/No)
- `menu_vegan`: Boolean
- `menu_seasonal`: Boolean
- `menu_grill`: Boolean
- `menu_kids`: Boolean
- `menu_diet`: Boolean
- `menu_exotic`: Boolean
- `menu_hot_dogs`: Boolean

**Pricing:**
- `min_price`: Minimum price in RUB
- `max_price`: Maximum price in RUB
- `avg_price_calculated`: Calculated average price

**Temporal:**
- `working_hours`: Operating hours string
- `open_time`: Opening time
- `close_time`: Closing time

**Quality Metrics:**
- `rating`: Rating (0-5 scale)
- `score`: Original quality score (8-44)
- `atmosphere_score`: Calculated atmosphere score
- `geo_cluster`: Geographic cluster ID (0-49)

### 4.3 Data Quality Assurance

**Verification Steps:**
1. Cross-reference with Яндекс Еда for accuracy
2. Verify contact information (phone, website, Instagram)
3. Validate geographic coordinates
4. Confirm category classification
5. Verify pricing information
6. Check operating hours accuracy

**Final Dataset:**
- **Total Records:** 99 entries (1 row removed due to critical missing data)
- **Final Working Dataset:** 98 restaurants and bars
- **Data Completeness:** 100% for critical fields after imputation

---

## 5. Data Preprocessing and Cleaning

### 5.1 Critical Data Cleaning

**Step 1: Remove Critical Missing Data**
```python
critical_columns = ['fsq_place_id', 'name', 'address', 'latitude', 'longitude']
df.dropna(subset=critical_columns, inplace=True)
```
**Result:** 1 row removed, 98 rows retained

**Step 2: Handle Missing Values**

**Numerical Columns (Median Imputation):**
- `score`, `geo_cluster`, `rating`
- `min_price`, `max_price`
- `atmosphere_score`, `avg_price_calculated`

**Categorical Contact Information ('N/A' Imputation):**
- `tel`, `website`, `instagram`
- `working_hours`, `open_time`, `close_time`

**Categorical Attributes (Mode Imputation):**
- `type`, `is_central`, `price_level`
- `metro`

**Step 3: Data Type Conversion**

**Boolean Conversion:**
- `terrace`: 'No'/'summer'/'none' → boolean
- `parking`: 'No'/'False' → boolean
- `is_central`: 'True'/'False' → boolean
- `menu_*` columns: → boolean

**Numeric Conversion:**
- `atmosphere`: Remove '%', '\xa0', handle 'regular' → float64

**Step 4: Remove Empty Columns**
- `features`, `cuisine_details`, `notes`, `special_menu` (all empty)

### 5.2 Feature Engineering

**Temporal Feature Extraction:**
```python
df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
df['close_time'] = pd.to_datetime(df['close_time'], errors='coerce')

time_diff = df['close_time'] - df['open_time']
df['hours_of_operation'] = time_diff.apply(
    lambda x: x.total_seconds() / 3600 
    if x.total_seconds() >= 0 
    else (x + pd.Timedelta(days=1)).total_seconds() / 3600
)
```

**Statistics:**
- Mean hours of operation: 12.16 hours
- Range: 0-19 hours
- Median: 12 hours

**Final Dataset Structure:**
- **32 columns** (after cleaning)
- **98 rows** (restaurants and bars)
- **10 boolean columns**
- **10 float64 columns**
- **12 object columns**

---

## 6. Knowledge Graph Construction

### 6.1 Graph Architecture

**Library:** NetworkX (Python)  
**Graph Type:** Directed Graph (DiGraph)  
**Total Nodes:** 176  
**Total Edges:** 623

### 6.2 Node Types

**1. Restaurant Nodes (98 nodes)**
- **Node ID:** `fsq_place_id`
- **Node Type:** `'Restaurant'`
- **Attributes:**
  - `name`: Restaurant name
  - `address`: Full address
  - `latitude`, `longitude`: Coordinates
  - `tel`, `website`, `instagram`: Contact info
  - `score`: Quality score (8-44)
  - `rating`: Rating (0-5)
  - `working_hours`: Operating hours
  - `open_time`, `close_time`: Time objects
  - `atmosphere_score`: Calculated score
  - `avg_price_calculated`: Average price
  - `hours_of_operation`: Calculated hours
  - `is_central`: Boolean

**2. PriceLevel Nodes (Variable)**
- **Node ID:** `PriceLevel_{price_level}`
- **Node Type:** `'PriceLevel'`
- **Values:** low, mid, high
- **Relation:** `has_price_level`

**3. Area Nodes (Variable)**
- **Node ID:** `Area_{metro_station}`
- **Node Type:** `'Area'`
- **Values:** Metro station names (e.g., "Парк культуры", "Тверская")
- **Relation:** `located_in`

**4. Atmosphere Nodes (Variable)**
- **Node ID:** `Atmosphere_{score_int}`
- **Node Type:** `'Atmosphere'`
- **Values:** Integer atmosphere scores (74-100)
- **Relation:** `has_atmosphere`

**5. Feature Nodes (Variable)**
- **Node ID:** `Feature_{feature_name}`
- **Node Type:** `'Feature'`
- **Values:** 
  - `Terrace`, `Parking`
  - `Menu Vegan`, `Menu Seasonal`, `Menu Grill`
  - `Menu Kids`, `Menu Diet`, `Menu Exotic`, `Menu Hot Dogs`
- **Relation:** `offers_feature`

### 6.3 Edge Types

**Relationship Types:**
1. **`has_price_level`**: Restaurant → PriceLevel
2. **`located_in`**: Restaurant → Area (Metro station)
3. **`has_atmosphere`**: Restaurant → Atmosphere
4. **`offers_feature`**: Restaurant → Feature

**Edge Creation Logic:**
```python
for index, row in df.iterrows():
    restaurant_id = row['fsq_place_id']
    
    # Price Level
    price_level = row['price_level']
    price_level_node_id = f"PriceLevel_{price_level}"
    G.add_node(price_level_node_id, type='PriceLevel', name=price_level)
    G.add_edge(restaurant_id, price_level_node_id, relation='has_price_level')
    
    # Metro Area
    metro_area = row['metro']
    area_node_id = f"Area_{metro_area}"
    G.add_node(area_node_id, type='Area', name=metro_area)
    G.add_edge(restaurant_id, area_node_id, relation='located_in')
    
    # Atmosphere
    atmosphere_score_int = int(row['atmosphere'])
    atmosphere_node_id = f"Atmosphere_{atmosphere_score_int}"
    G.add_node(atmosphere_node_id, type='Atmosphere', score=atmosphere_score_int)
    G.add_edge(restaurant_id, atmosphere_node_id, relation='has_atmosphere')
    
    # Features
    feature_cols = ['terrace', 'parking', 'menu_vegan', 'menu_seasonal', 
                    'menu_grill', 'menu_kids', 'menu_diet', 'menu_exotic', 
                    'menu_hot_dogs']
    for col in feature_cols:
        if row[col]:
            feature_name = col.replace('_', ' ').title()
            feature_node_id = f"Feature_{feature_name}"
            G.add_node(feature_node_id, type='Feature', name=feature_name)
            G.add_edge(restaurant_id, feature_node_id, relation='offers_feature')
```

### 6.4 Graph Statistics

**Node Distribution:**
- Restaurant nodes: 98
- PriceLevel nodes: 3 (low, mid, high)
- Area nodes: ~50 (unique metro stations)
- Atmosphere nodes: ~15 (unique atmosphere scores)
- Feature nodes: 9 (unique features)

**Edge Distribution:**
- `has_price_level`: 98 edges
- `located_in`: 98 edges
- `has_atmosphere`: 98 edges
- `offers_feature`: ~329 edges (variable per restaurant)

**Total:** 176 nodes, 623 edges

---

## 7. Data Pipeline Summary

### 7.1 Complete Workflow

```
FourSquare S3 Dataset (60,964 establishments)
    ↓
Category Filtering (12,762 food establishments)
    ↓
Quality Filtering (8,541 quality establishments)
    ↓
Geographic Clustering (50 clusters)
    ↓
Scoring Algorithm (8-44 points)
    ↓
Top 1500 Selection (1,355 selected, with geographic filtering)
    ↓
DataLens Visualization: [1500 Restaurants Dashboard](https://datalens.yandex/w3048zrqxgtyh)
    ↓
Top 100 Selection (for manual enrichment)
    ↓
Manual Enrichment via Яндекс Еда
    ↓
Data Cleaning & Preprocessing
    ↓
Final Dataset (98 restaurants/bars, 32 attributes)
    ↓
Knowledge Graph Construction (176 nodes, 623 edges)
    ↓
Recommendation System Implementation
```

### 7.2 Data Reduction Statistics

| Stage | Count | Reduction |
|-------|-------|-----------|
| Initial S3 Dataset | 60,964 | - |
| Food Establishments | 12,762 | 79.1% reduction |
| Quality Filtered | 8,541 | 33.1% reduction |
| Top 1500 Selected | 1,355 | 84.1% reduction |
| Top 100 Selected | 100 | 92.6% reduction |
| Final Working Dataset | 98 | 2.0% reduction |
| **Final Reduction** | **98** | **99.84% reduction** |

### 7.3 Quality Metrics

**Data Completeness (Final Dataset):**
- Critical fields: 100%
- Contact information: 75-79%
- Menu features: 100% (boolean)
- Pricing: 100%
- Temporal: 100%

**Geographic Coverage:**
- Unique metro stations: ~50
- Geographic clusters: 44 represented
- Central vs. Non-central: Balanced distribution

**Category Distribution:**
- Restaurants: 65 (66.3%)
- Bars: 33 (33.7%)

---

## 8. Comparison with Original Research

### 8.1 Dataset Comparison

| Aspect | Original Research | This Study |
|--------|------------------|------------|
| **Data Source** | UCI Restaurant Data | FourSquare S3 (2025) |
| **Initial Size** | 935 restaurants | 60,964 establishments |
| **Final Size** | 935 restaurants | 98 restaurants/bars |
| **Users** | 138 users | 0 users (no user data) |
| **Ratings** | 1,161 ratings | 0 ratings |
| **Enrichment** | Automated | Manual (Яндекс Еда) |
| **Graph Nodes** | Variable | 176 nodes |
| **Graph Edges** | Variable | 623 edges |

### 8.2 Methodology Differences

**Original Research:**
- Used GP 2 (graph programming language)
- User-restaurant interactions available
- Collaborative filtering possible
- Three schemes with user-user similarity

**This Study:**
- Used NetworkX (Python library)
- No user data available
- Content-based and hybrid approaches only
- Focus on restaurant-restaurant similarity
- Manual data enrichment for quality

### 8.3 Advantages of This Approach

1. **Data Freshness:** 2025 dataset vs. older UCI dataset
2. **Data Quality:** Manual enrichment ensures accuracy
3. **Geographic Relevance:** Moscow-specific data
4. **Rich Attributes:** 32 attributes vs. limited original attributes
5. **Real-World Application:** Directly applicable to Moscow restaurant scene

### 8.4 Limitations

1. **No User Data:** Cannot implement collaborative filtering
2. **Small Dataset:** 98 establishments vs. 935 in original
3. **Manual Process:** Time-intensive enrichment
4. **No Ratings:** Cannot evaluate against user preferences
5. **Single City:** Limited to Moscow

---

## 9. Reproducibility and Documentation

### 9.1 Code Availability

**Notebooks:**
1. `notebooks/frsqr_2025.ipynb`: Data parsing and selection pipeline
2. `notebooks/recsys_knowledge_graph.ipynb`: Knowledge graph construction and recommendation system

**Datasets:**
1. `data/processed/workingrest.csv`: Final enriched dataset (98 establishments) ⭐ **Primary dataset**
2. `data/processed/moscow_top_100_restaurants.csv`: Top 100 for enrichment
3. `data/processed/moscow_data_lens_export.csv`: Data for visualization (Data Lens) - 1,355 establishments with geographic clustering
   - **Visualization**: [DataLens Dashboard - 1500 Restaurants](https://datalens.yandex/w3048zrqxgtyh)
   - Features geographic filtering by latitude/longitude and geo-clustering for balanced representation across Moscow districts
4. `data/processed/moscow_data_lens_small.csv`: Small version for visualization

### 9.2 Key Parameters

**Geographic Clustering:**
- Number of clusters: 50
- Random state: 42
- Algorithm: K-Means (sklearn)

**Scoring Weights:**
- Category base: 10-20 points
- Contact info: 5-8 points per field
- Premium indicators: +8 to +10 points
- Chain penalties: -5 points

**Selection Criteria:**
- Top 1500: Balanced by type and geography (filtering by latitude/longitude with geo-clustering for weighted processing across all Moscow districts)
  - **Visualization**: [DataLens Dashboard](https://datalens.yandex/w3048zrqxgtyh)
- Top 100: Score-based selection
- Final 98: After data cleaning

### 9.3 Data Access

**FourSquare S3 Access:**
- Bucket: `s3://fsq-os-places-us-east-1/release/dt=2025-09-09/`
- Format: Parquet files
- Access: Direct DuckDB queries (no authentication required for public data)

**Manual Enrichment Source:**
- Яндекс Еда (Yandex Food): https://eda.yandex.ru
- Manual verification and data entry

---

## 10. Ethical Considerations and Data Usage

### 10.1 Data Source Compliance

- **FourSquare Data:** Public Open Street Places dataset
- **Usage Rights:** Open data for research purposes
- **Attribution:** FourSquare Open Street Places acknowledged

### 10.2 Manual Data Collection

- **Source:** Яндекс Еда (publicly available information)
- **Purpose:** Research and academic study
- **Scope:** Restaurant information only (no personal data)
- **Verification:** All data manually verified for accuracy

### 10.3 Data Privacy

- **No Personal Information:** Dataset contains only public restaurant data
- **No User Tracking:** No user behavior or personal data collected
- **Public Information Only:** All data from publicly available sources

---

## 11. Future Improvements

### 11.1 Dataset Expansion

1. **Increase Sample Size:** Expand from 98 to 200-300 establishments
2. **Geographic Expansion:** Include other Russian cities
3. **Category Expansion:** Include cafes, bistros, gastropubs separately

### 11.2 Data Collection Automation

1. **API Integration:** Direct integration with Яндекс Еда API (if available)
2. **Web Scraping:** Automated data collection (with proper permissions)
3. **Crowdsourcing:** User-contributed data validation

### 11.3 Attribute Enhancement

1. **Cuisine Types:** Detailed cuisine classification
2. **Dietary Restrictions:** More detailed dietary options
3. **Accessibility:** Wheelchair access, dietary accommodations
4. **Reviews Integration:** Aggregate review scores and sentiment

---

## 12. Conclusion

This methodology demonstrates a comprehensive approach to building a high-quality restaurant dataset for knowledge graph-based recommendation systems. The multi-stage process—from initial S3 data extraction through manual enrichment to knowledge graph construction—ensures data quality and relevance for the Moscow restaurant scene.

The final dataset of 98 establishments, enriched with 32 manually verified attributes, provides a solid foundation for recommendation system development. The knowledge graph structure (176 nodes, 623 edges) enables sophisticated similarity calculations and recommendation schemes, even without user interaction data.

**Key Contributions:**
1. Systematic data collection from FourSquare S3
2. Quality-based filtering and scoring
3. Geographic clustering for balanced distribution
4. Manual enrichment for data accuracy
5. Comprehensive knowledge graph construction
6. Full documentation for reproducibility

This methodology can serve as a template for similar studies in other cities or domains, demonstrating the value of combining automated data collection with manual quality assurance.

---

## References

1. FourSquare Open Street Places Dataset: https://github.com/foursquare/fsq-os-places
2. DuckDB Documentation: https://duckdb.org/docs/
3. NetworkX Documentation: https://networkx.org/
4. Nitamayega et al. (2024). "Utilizing GP 2 for Restaurant Recommendation." *Ind. Journal on Computing*, Vol. 9, Issue. 1, pp. 8-19.

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025  
**Author:** Askhabaliev Gadzhi   
**Dataset Version:** workingrest.csv v1.0 (98 establishments)

