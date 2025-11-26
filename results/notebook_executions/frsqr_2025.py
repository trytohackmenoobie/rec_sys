#!/usr/bin/env python
# coding: utf-8

# In[2]:


import duckdb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# The code analyzes geographic data from Foursquare's Open Street Places dataset, specifically the September 9, 2025 snapshot. It helps understand the available data fields before performing more complex queries or analysis.

# In[2]:


conn = duckdb.connect()


print("Table structure for categories:")
categories_structure = conn.execute("""
DESCRIBE
FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/categories/parquet/**/*.parquet')
""").df()
print(categories_structure)

print("\nTable structure for places:")
places_structure = conn.execute("""
DESCRIBE
FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet')
""").df()
print(places_structure)

print("\nSample categories:")
sample_categories = conn.execute("""
SELECT *
FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/categories/parquet/**/*.parquet')
LIMIT 10
""").df()
print(sample_categories)


# Identifying relevant categories like restaurants, bars, and cafes, then match them with actual places in Moscow

# In[4]:


query = """
WITH target_categories AS (
    SELECT
        category_id,
        category_name,
        level1_category_name
    FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/categories/parquet/**/*.parquet')
    WHERE
        level1_category_name = 'Dining and Drinking'
        OR category_name ILIKE '%restaurant%'
        OR category_name ILIKE '%bar%'
        OR category_name ILIKE '%cafe%'
        OR category_name ILIKE '%coffee%'
        OR category_name ILIKE '%pub%'
        OR category_name ILIKE '%bistro%'
)
SELECT DISTINCT
    p.fsq_place_id,
    p.name,
    p.latitude,
    p.longitude,
    p.address,
    p.locality,
    p.region,
    p.fsq_category_labels,
    STRING_AGG(DISTINCT c.category_name, ', ') AS matched_categories
FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet') p
JOIN target_categories c ON ARRAY_CONTAINS(p.fsq_category_ids, c.category_id)
WHERE
    p.locality = 'Moscow'
    OR p.region = 'Moscow'
    OR p.admin_region = 'Moscow'
GROUP BY ALL
"""

df_result = conn.execute(query).df()
print(f"Found {len(df_result)} establishments in Moscow")

if len(df_result) > 0:
    print("\nDistribution by categories:")
    print(df_result['matched_categories'].value_counts().head(15))


# Scoring each establishment based on category importance and contact information availability, then select the top 500 venues with balanced representation across different types. Finally, I saved the curated dataset for further analysis.

# In[5]:


query = """
SELECT
    fsq_place_id,
    name,
    latitude,
    longitude,
    address,
    locality,
    region,
    tel,
    website,
    instagram,
    fsq_category_labels
FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet')
WHERE locality = 'Moscow' OR region = 'Moscow'
LIMIT 1000
"""

df_all = conn.execute(query).df()
print(f"Retrieved establishments for analysis: {len(df_all)}")

print("Category analysis:")
for i in range(min(10, len(df_all))):
    categories = df_all.iloc[i]['fsq_category_labels']
    print(f"{i+1}. {df_all.iloc[i]['name']}: {categories}")

has_categories = df_all['fsq_category_labels'].notna().sum()
print(f"Establishments with categories: {has_categories} out of {len(df_all)}")

query_all = """
SELECT
    fsq_place_id,
    name,
    latitude,
    longitude,
    address,
    locality,
    region,
    tel,
    website,
    instagram,
    fsq_category_labels
FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet')
WHERE locality = 'Moscow' OR region = 'Moscow'
"""

df = conn.execute(query_all).df()
print(f"Total establishments in Moscow: {len(df)}")

def safe_categories(categories):
    if categories is None:
        return []
    try:
        return [str(cat) for cat in categories]
    except:
        return []

df['categories_list'] = df['fsq_category_labels'].apply(safe_categories)

def is_food_establishment(categories_list):
    if not categories_list:
        return False

    food_keywords = ['restaurant', 'bar', 'cafe', 'coffee', 'pub', 'bakery', 'steakhouse',
                     'eatery', 'bistro', 'grill', 'pizzeria', 'food', 'dining']

    categories_str = ' '.join(categories_list).lower()
    return any(keyword in categories_str for keyword in food_keywords)

df_food = df[df['categories_list'].apply(is_food_establishment)]
print(f"Food establishments: {len(df_food)}")

if len(df_food) > 0:
    def calculate_score(row):
        score = 0
        categories = row['categories_list']

        for category in categories:
            cat_lower = category.lower()
            if 'restaurant' in cat_lower:
                score += 10
            elif 'steakhouse' in cat_lower:
                score += 9
            elif 'bar' in cat_lower:
                score += 7
            elif 'coffee' in cat_lower or 'cafe' in cat_lower:
                score += 5

        if pd.notna(row['website']) and row['website']:
            score += 3
        if pd.notna(row['tel']) and row['tel']:
            score += 2
        if pd.notna(row['instagram']) and row['instagram']:
            score += 2

        return score

    def categorize(row):
        categories = row['categories_list']
        for category in categories:
            cat_lower = category.lower()
            if 'restaurant' in cat_lower:
                return 'Restaurant'
            if 'bar' in cat_lower:
                return 'Bar'
            if 'coffee' in cat_lower or 'cafe' in cat_lower:
                return 'Coffee'
        return 'Other'

    df_food['score'] = df_food.apply(calculate_score, axis=1)
    df_food['type'] = df_food.apply(categorize, axis=1)

    def select_top_500(df):
        target_counts = {'Restaurant': 200, 'Bar': 150, 'Coffee': 150}
        selected_dfs = []

        for establishment_type, count in target_counts.items():
            type_df = df[df['type'] == establishment_type]
            if len(type_df) > count:
                selected_dfs.append(type_df.nlargest(count, 'score'))
            else:
                selected_dfs.append(type_df)

        result = pd.concat(selected_dfs, ignore_index=True)

        if len(result) > 500:
            result = result.nlargest(500, 'score')

        return result

    df_top_500 = select_top_500(df_food)
    print(f"Selected top 500 establishments: {len(df_top_500)}")
    print("Distribution by type:")
    print(df_top_500['type'].value_counts())

    df_top_500[['fsq_place_id', 'name', 'address', 'latitude', 'longitude',
                'tel', 'website', 'instagram', 'type', 'score']].to_csv(
        'moscow_top_500_with_scoring.csv', index=False, encoding='utf-8')

    print("Dataset with scoring saved")
else:
    print("No food establishments found. Saving all establishments for analysis.")
    df.to_csv('moscow_all_places.csv', index=False, encoding='utf-8')
    print("All establishments saved to moscow_all_places.csv for analysis")


# In[7]:


df_result = pd.read_csv('moscow_top_500_with_scoring.csv', encoding='utf-8')

print("First 10 establishments:")
print(df_result.head(20))

print("\nBasic data information:")
print(df_result.info())

print("\nDistribution by establishment types:")
print(df_result['type'].value_counts())

print("\nScoring statistics:")
print(df_result['score'].describe())


# Filtering out fast food chains and low-quality venues while identifying premium dining establishments like restaurants, bars, and cafes. I score each venue based on category quality and contact information, then select the top 500 quality establishments with balanced representation across different types.

# In[8]:


query = """
SELECT
    fsq_place_id,
    name,
    latitude,
    longitude,
    address,
    locality,
    region,
    tel,
    website,
    instagram,
    fsq_category_labels
FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet')
WHERE locality = 'Moscow' OR region = 'Moscow'
"""

df = conn.execute(query).df()
print(f"Total establishments in Moscow: {len(df)}")

def safe_categories(categories):
    if categories is None:
        return []
    try:
        return [str(cat) for cat in categories]
    except:
        return []

df['categories_list'] = df['fsq_category_labels'].apply(safe_categories)

fast_food_chains = [
    'додо пицца', 'dodo pizza', 'kfc', 'макдональд', 'mcdonald', 'бургер кинг',
    'burger king', 'субвей', 'subway', 'панда пицца', 'panda pizza', 'тарту',
    'тартуга', 'вкусно и точка', 'чикен', 'chicken', 'шаурма', 'шаверма'
]

low_quality_keywords = [
    'столовая', 'кафетерий', 'фудкорт', 'food court', 'фастфуд', 'fast food',
    'сетевое', 'сеть', 'chain'
]

def is_quality_food_establishment(row):
    """Checks if establishment is quality (not fast food)"""
    categories = row['categories_list']
    name = str(row['name']).lower()

    for chain in fast_food_chains:
        if chain in name:
            return False

    for keyword in low_quality_keywords:
        if keyword in name:
            return False

    if not categories:
        return False

    categories_str = ' '.join(categories).lower()

    fast_food_categories = ['fast food', 'food court', 'cafeteria']
    for cat in fast_food_categories:
        if cat in categories_str:
            return False

    quality_keywords = [
        'restaurant', 'bar', 'cafe', 'coffee', 'pub', 'steakhouse',
        'eatery', 'bistro', 'grill', 'wine bar', 'cocktail bar',
        'итальянск', 'французск', 'японск', 'европейск', 'авторск'
    ]

    return any(keyword in categories_str for keyword in quality_keywords)

df_quality = df[df.apply(is_quality_food_establishment, axis=1)]
print(f"Quality establishments: {len(df_quality)}")

def calculate_quality_score(row):
    score = 0
    categories = row['categories_list']
    name = str(row['name']).lower()

    premium_keywords = {
        'restaurant': 15,
        'steakhouse': 14,
        'bistro': 13,
        'grill': 12,
        'bar': 10,
        'wine bar': 12,
        'cocktail bar': 11,
        'pub': 8,
        'cafe': 7,
        'coffee': 6
    }

    if categories:
        categories_str = ' '.join(categories).lower()
        for keyword, points in premium_keywords.items():
            if keyword in categories_str:
                score += points
                break

    if pd.notna(row['website']) and row['website']:
        score += 5
    if pd.notna(row['tel']) and row['tel']:
        score += 3
    if pd.notna(row['instagram']) and row['instagram']:
        score += 4

    score += 10

    chain_indicators = ['#1', '#2', 'филиал', 'сеть']
    for indicator in chain_indicators:
        if indicator in name:
            score -= 5

    return score

def categorize_quality(row):
    categories = row['categories_list']
    if not categories:
        return 'Other'

    categories_str = ' '.join(categories).lower()

    if 'restaurant' in categories_str or 'steakhouse' in categories_str:
        return 'Restaurant'
    elif 'bar' in categories_str or 'pub' in categories_str:
        return 'Bar'
    elif 'cafe' in categories_str or 'coffee' in categories_str:
        return 'Coffee'
    return 'Other'

df_quality['score'] = df_quality.apply(calculate_quality_score, axis=1)
df_quality['type'] = df_quality.apply(categorize_quality, axis=1)

def select_quality_top_500(df):
    target_counts = {'Restaurant': 200, 'Bar': 150, 'Coffee': 150}
    selected_dfs = []

    for establishment_type, count in target_counts.items():
        type_df = df[df['type'] == establishment_type]
        if len(type_df) > count:
            selected_dfs.append(type_df.nlargest(count, 'score'))
        else:
            selected_dfs.append(type_df)

    result = pd.concat(selected_dfs, ignore_index=True)

    if len(result) > 500:
        result = result.nlargest(500, 'score')

    return result

df_top_quality = select_quality_top_500(df_quality)
print(f"Selected quality establishments: {len(df_top_quality)}")
print("Distribution by types:")
print(df_top_quality['type'].value_counts())

print("\nCheck - examples of selected establishments:")
sample_names = df_top_quality['name'].head(20).tolist()
for name in sample_names:
    print(f"  - {name}")

df_top_quality[['fsq_place_id', 'name', 'address', 'latitude', 'longitude',
                'tel', 'website', 'instagram', 'type', 'score']].to_csv(
    'moscow_quality_top_500.csv', index=False, encoding='utf-8')

print("\nQuality dataset saved as moscow_quality_top_500.csv")


# Dividing Moscow into geographic areas to ensure balanced spatial distribution.Using enhanced K-means clustering with 50 geographic areas, I ensure comprehensive spatial coverage across the city. I score each establishment with an improved algorithm that rewards premium categories, unique concepts, and contact information while penalizing chain operations. The final selection maintains perfect balance across restaurants, bars, and cafes with controlled geographic distribution, providing detailed statistics on scoring, contact availability, and spatial coverage for comprehensive analysis.

# In[ ]:


conn = duckdb.connect()

query = """
SELECT
    fsq_place_id,
    name,
    latitude,
    longitude,
    address,
    locality,
    region,
    tel,
    website,
    instagram,
    fsq_category_labels
FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2025-09-09/places/parquet/**/*.parquet')
WHERE locality = 'Moscow' OR region = 'Moscow'
"""

df = conn.execute(query).df()
print(f"Total establishments in Moscow: {len(df)}")

def safe_categories(categories):
    if categories is None:
        return []
    try:
        return [str(cat) for cat in categories]
    except:
        return []

df['categories_list'] = df['fsq_category_labels'].apply(safe_categories)

fast_food_chains = [
    'додо пицца', 'dodo pizza', 'kfc', 'макдональд', 'mcdonald', 'бургер кинг',
    'burger king', 'субвей', 'subway', 'панда пицца', 'panda pizza', 'тарту',
    'тартуга', 'вкусно и точка', 'чикен', 'chicken', 'шаурма', 'шаверма',
    'макдоналдс', 'бургер', 'burger', 'пицца', 'pizza', 'суши', 'sushi',
    'му-му', 'муму', 'moo moo', 'столовая', 'кафетерий', 'фудкорт', 'food court',
    'макдоналдс', 'макдак', 'макдач', 'mcDonald', 'kfc', 'kentucky'
]

def is_quality_food_establishment(row):
    categories = row['categories_list']
    name = str(row['name']).lower()

    for chain in fast_food_chains:
        if chain in name:
            return False

    if not categories:
        return False

    categories_str = ' '.join(categories).lower()

    fast_food_categories = ['fast food', 'food court', 'cafeteria']
    for cat in fast_food_categories:
        if cat in categories_str:
            return False

    quality_keywords = [
        'restaurant', 'bar', 'cafe', 'coffee', 'pub', 'steakhouse',
        'eatery', 'bistro', 'grill', 'wine bar', 'cocktail bar',
        'итальянск', 'французск', 'японск', 'европейск', 'авторск',
        'гастропаб', 'паб', 'винный', 'кофейн', 'кондитерск', 'гастроном'
    ]

    return any(keyword in categories_str for keyword in quality_keywords)

df_quality = df[df.apply(is_quality_food_establishment, axis=1)].copy()
print(f"Quality establishments: {len(df_quality)}")

def assign_geo_clusters_improved(df, n_clusters=50):
    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_copy = df.copy()
    df_copy['geo_cluster'] = kmeans.fit_predict(coords)
    return df_copy, kmeans

df_quality, kmeans = assign_geo_clusters_improved(df_quality, n_clusters=50)
print(f"Created geographic clusters: {df_quality['geo_cluster'].nunique()}")

def calculate_enhanced_score(row):
    score = 0
    categories = row['categories_list']
    name = str(row['name']).lower()

    premium_keywords = {
        'restaurant': 15, 'steakhouse': 18, 'bistro': 16, 'grill': 14,
        'bar': 12, 'wine bar': 17, 'cocktail bar': 16, 'pub': 10,
        'cafe': 8, 'coffee': 7, 'гастропаб': 19, 'авторск': 20
    }

    if categories:
        categories_str = ' '.join(categories).lower()
        for keyword, points in premium_keywords.items():
            if keyword in categories_str:
                score += points
                break

    contact_score = 0
    if pd.notna(row['website']) and row['website']:
        contact_score += 8
    if pd.notna(row['tel']) and row['tel']:
        contact_score += 5
    if pd.notna(row['instagram']) and row['instagram']:
        contact_score += 6

    score += contact_score

    unique_indicators = ['авторск', 'гастропаб', 'винный', 'крафтов', 'craft', 'гастроном', 'уникальн']
    premium_indicators = ['премиум', 'premium', 'люкс', 'luxury', 'высок', 'gourmet']

    for indicator in unique_indicators:
        if indicator in name:
            score += 8

    for indicator in premium_indicators:
        if indicator in name:
            score += 10

    chain_penalties = ['сеть', 'chain', 'филиал', '№1', '№2', '№3']
    for penalty in chain_penalties:
        if penalty in name:
            score -= 5

    return score

def categorize_quality(row):
    categories = row['categories_list']
    if not categories:
        return 'Other'

    categories_str = ' '.join(categories).lower()

    if 'restaurant' in categories_str or 'steakhouse' in categories_str or 'bistro' in categories_str:
        return 'Restaurant'
    elif 'bar' in categories_str or 'pub' in categories_str or 'wine bar' in categories_str:
        return 'Bar'
    elif 'cafe' in categories_str or 'coffee' in categories_str:
        return 'Coffee'
    return 'Other'

df_quality['score'] = df_quality.apply(calculate_enhanced_score, axis=1)
df_quality['type'] = df_quality.apply(categorize_quality, axis=1)

def select_1500_balanced(df, total_target=1500):
    target_counts = {'Restaurant': 500, 'Bar': 500, 'Coffee': 500}
    max_per_cluster = {'Restaurant': 25, 'Bar': 20, 'Coffee': 15}

    selected_indices = []

    for establishment_type, type_target in target_counts.items():
        type_df = df[df['type'] == establishment_type].copy()

        if len(type_df) == 0:
            continue

        type_df = type_df.sort_values('score', ascending=False)
        cluster_counts = {}

        for idx, row in type_df.iterrows():
            if len([i for i in selected_indices if df.loc[i, 'type'] == establishment_type]) >= type_target:
                continue

            cluster = row['geo_cluster']

            if cluster not in cluster_counts:
                cluster_counts[cluster] = 0

            if cluster_counts[cluster] < max_per_cluster[establishment_type]:
                selected_indices.append(idx)
                cluster_counts[cluster] += 1

    result = df.loc[selected_indices].copy()

    final_result = []
    type_counts = {'Restaurant': 0, 'Bar': 0, 'Coffee': 0}

    for idx in selected_indices:
        row_type = df.loc[idx, 'type']
        if type_counts[row_type] < target_counts[row_type]:
            final_result.append(idx)
            type_counts[row_type] += 1

    result = df.loc[final_result].copy()

    return result

df_top_1500 = select_1500_balanced(df_quality)
print(f"Selected balanced establishments: {len(df_top_1500)}")
print("Distribution by types:")
print(df_top_1500['type'].value_counts())

print("\nGeographic distribution (top-15 clusters):")
cluster_distribution = df_top_1500['geo_cluster'].value_counts()
print(cluster_distribution.head(15))

print(f"\nTotal clusters represented: {df_top_1500['geo_cluster'].nunique()}")
print(f"Maximum establishments in one cluster: {cluster_distribution.max()}")
print(f"Minimum establishments in one cluster: {cluster_distribution.min()}")

print("\nScoring statistics:")
print(f"Average score: {df_top_1500['score'].mean():.2f}")
print(f"Maximum score: {df_top_1500['score'].max()}")
print(f"Minimum score: {df_top_1500['score'].min()}")

print("\nAverage scoring by type:")
print(df_top_1500.groupby('type')['score'].mean())

print("\nContact information in selected establishments:")
print(f"Websites: {df_top_1500['website'].notna().sum()} ({df_top_1500['website'].notna().sum()/len(df_top_1500)*100:.1f}%)")
print(f"Phones: {df_top_1500['tel'].notna().sum()} ({df_top_1500['tel'].notna().sum()/len(df_top_1500)*100:.1f}%)")
print(f"Instagram: {df_top_1500['instagram'].notna().sum()} ({df_top_1500['instagram'].notna().sum()/len(df_top_1500)*100:.1f}%)")

print("\nTop-15 establishments by score:")
top_15 = df_top_1500.nlargest(15, 'score')[['name', 'type', 'score']]
for i, (_, row) in enumerate(top_15.iterrows(), 1):
    print(f"  {i}. {row['name']} ({row['type']}) - {row['score']} points")

df_top_1500[['fsq_place_id', 'name', 'address', 'latitude', 'longitude',
             'tel', 'website', 'instagram', 'type', 'score', 'geo_cluster']].to_csv(
    'moscow_quality_1500_enhanced.csv', index=False, encoding='utf-8')

cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['latitude', 'longitude'])
cluster_centers['geo_cluster'] = cluster_centers.index
cluster_centers.to_csv('moscow_geo_clusters_50.csv', index=False)

print(f"\nDataset of 1500 establishments saved")
print(f"Information about 50 geographic clusters saved")


# In[4]:


restaurants = df_top_1500[df_top_1500['type'] == 'Restaurant'].head(10)

print("Complete information about first 10 restaurants:")
print("=" * 80)

for i, (idx, row) in enumerate(restaurants.iterrows(), 1):
    print(f"\n{i}. {row['name']}")
    print(f"   ID: {row['fsq_place_id']}")
    print(f"   Address: {row['address']}")
    print(f"   Coordinates: {row['latitude']:.6f}, {row['longitude']:.6f}")
    print(f"   Phone: {row['tel']}")
    print(f"   Website: {row['website']}")
    print(f"   Instagram: {row['instagram']}")
    print(f"   Score: {row['score']}")
    print(f"   Geo-cluster: {row['geo_cluster']}")
    print(f"   Categories: {row['categories_list']}")

print("\n" + "=" * 80)
print(f"Total restaurants in dataset: {len(df_top_1500[df_top_1500['type'] == 'Restaurant'])}")
print(f"Total dataset size: {len(df_top_1500)} establishments")


# In[5]:


data_lens_export = df_top_1500.copy()


data_lens_export['geo_cluster_name'] = 'Cluster ' + data_lens_export['geo_cluster'].astype(str)


data_lens_export['point_size'] = data_lens_export['score'] / data_lens_export['score'].max() * 10


color_mapping = {'Restaurant': '#FF6B6B', 'Bar': '#4ECDC4', 'Coffee': '#45B7D1'}
data_lens_export['color'] = data_lens_export['type'].map(color_mapping)


data_lens_export[[
    'fsq_place_id', 'name', 'address', 'latitude', 'longitude',
    'type', 'score', 'geo_cluster', 'geo_cluster_name', 'point_size', 'color',
    'tel', 'website', 'instagram'
]].to_csv('data/processed/moscow_data_lens_export.csv', index=False, encoding='utf-8')

print("Данные для Data Lens сохранены в файл: moscow_data_lens_export.csv")
print("\nСтруктура данных для визуализации:")
print(f"- Всего точек: {len(data_lens_export)}")
print(f"- Типы заведений: {dict(data_lens_export['type'].value_counts())}")
print(f"- Гео-кластеры: {data_lens_export['geo_cluster'].nunique()}")
print(f"- Диапазон скоринга: {data_lens_export['score'].min()} - {data_lens_export['score'].max()}")


# In[6]:


df_top_100 = df_top_1500.nlargest(100, 'score').copy()

print(f"Топ-100 заведений Москвы:")
print("=" * 50)


for i, (idx, row) in enumerate(df_top_100.iterrows(), 1):
    print(f"\n{i}. {row['name']}")
    print(f"   Тип: {row['type']}")
    print(f"   Score: {row['score']}")
    print(f"   Адрес: {row['address']}")
    print(f"   Телефон: {row['tel'] if pd.notna(row['tel']) else 'нет'}")
    print(f"   Сайт: {row['website'] if pd.notna(row['website']) else 'нет'}")
    print(f"   Instagram: {row['instagram'] if pd.notna(row['instagram']) else 'нет'}")
    print(f"   Гео-кластер: {row['geo_cluster']}")


df_top_100_for_enrichment = df_top_100[[
    'fsq_place_id', 'name', 'address', 'tel', 'website', 'instagram',
    'type', 'score', 'geo_cluster'
]].copy()


df_top_100_for_enrichment['price_level'] = ''
df_top_100_for_enrichment['atmosphere'] = ''
df_top_100_for_enrichment['terrace'] = ''
df_top_100_for_enrichment['parking'] = ''
df_top_100_for_enrichment['features'] = ''
df_top_100_for_enrichment['cuisine_details'] = ''
df_top_100_for_enrichment['notes'] = ''


df_top_100_for_enrichment.to_csv('top_100_moscow_restaurants_for_manual_enrichment.csv',
                                index=False, encoding='utf-8')

print(f"\nСохранено 100 заведений в 'top_100_moscow_restaurants_for_manual_enrichment.csv'")
print(f"Распределение по типам:")
print(df_top_100['type'].value_counts())
print(f" Диапазон score: {df_top_100['score'].min()} - {df_top_100['score'].max()}")


# In[7]:


import os


os.makedirs('top_100_restaurants', exist_ok=True)

for i, (idx, row) in enumerate(df_top_100.iterrows(), 1):
    restaurant_data = {
        'fsq_place_id': row['fsq_place_id'],
        'name': row['name'],
        'address': row['address'],
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'tel': row['tel'] if pd.notna(row['tel']) else '',
        'website': row['website'] if pd.notna(row['website']) else '',
        'instagram': row['instagram'] if pd.notna(row['instagram']) else '',
        'type': row['type'],
        'score': row['score'],
        'geo_cluster': row['geo_cluster'],
        'price_level': '',
        'atmosphere': '',
        'terrace': '',
        'parking': '',
        'features': '',
        'cuisine_details': '',
        'notes': ''
    }


    safe_name = "".join(c for c in row['name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"{i:03d}_{safe_name}.csv"


    pd.DataFrame([restaurant_data]).to_csv(
        f'top_100_restaurants/{filename}',
        index=False,
        encoding='utf-8'
    )

print(f" Сохранено 100 отдельных файлов в папку 'top_100_restaurants/'")


# In[ ]:


df_top_100 = df_top_1500.nlargest(100, 'score').copy()

print(f"Топ-100 заведений Москвы:")
print("=" * 80)

for i, (idx, row) in enumerate(df_top_100.iterrows(), 1):
    print(f"\n{i}. {row['name']}")
    print(f"   Тип: {row['type']}")
    print(f"   Score: {row['score']}")
    print(f"   Адрес: {row['address']}")
    print(f"   Широта: {row['latitude']:.6f}")
    print(f"   Долгота: {row['longitude']:.6f}")
    print(f"   Телефон: {row['tel'] if pd.notna(row['tel']) else 'нет'}")
    print(f"   Сайт: {row['website'] if pd.notna(row['website']) else 'нет'}")
    print(f"   Instagram: {row['instagram'] if pd.notna(row['instagram']) else 'нет'}")
    print(f"   Гео-кластер: {row['geo_cluster']}")

# Сохранение с координатами
df_top_100_for_enrichment = df_top_100[[
    'fsq_place_id', 'name', 'address', 'latitude', 'longitude',
    'tel', 'website', 'instagram', 'type', 'score', 'geo_cluster'
]].copy()

# Добавляем поля для ручного обогащения
df_top_100_for_enrichment['price_level'] = ''
df_top_100_for_enrichment['atmosphere'] = ''
df_top_100_for_enrichment['terrace'] = ''
df_top_100_for_enrichment['parking'] = ''
df_top_100_for_enrichment['features'] = ''
df_top_100_for_enrichment['cuisine_details'] = ''
df_top_100_for_enrichment['notes'] = ''

# Сохраняем файл
df_top_100_for_enrichment.to_csv('top_100_moscow_restaurants_with_coordinates.csv',
                                index=False, encoding='utf-8')

print(f"\n=== СВОДНАЯ ИНФОРМАЦИЯ ===")
print(f"Сохранено 100 заведений в 'top_100_moscow_restaurants_with_coordinates.csv'")
print(f"Распределение по типам:")
print(df_top_100['type'].value_counts())
print(f"Диапазон score: {df_top_100['score'].min()} - {df_top_100['score'].max()}")
print(f"Диапазон широт: {df_top_100['latitude'].min():.6f} - {df_top_100['latitude'].max():.6f}")
print(f"Диапазон долгот: {df_top_100['longitude'].min():.6f} - {df_top_100['longitude'].max():.6f}")

# Дополнительная статистика по координатам
print(f"\n=== ГЕОГРАФИЧЕСКАЯ СТАТИСТИКА ===")
print(f"Средняя широта: {df_top_100['latitude'].mean():.6f}")
print(f"Средняя долгота: {df_top_100['longitude'].mean():.6f}")
print(f"Уникальных гео-кластеров в топ-100: {df_top_100['geo_cluster'].nunique()}")


# In[9]:


# Сохраняем топ-100 заведений со всей информацией
df_top_100 = df_top_1500.nlargest(100, 'score').copy()

# Выбираем нужные колонки включая широту и долготу
df_top_100[['fsq_place_id', 'name', 'address', 'latitude', 'longitude',
           'tel', 'website', 'instagram', 'type', 'score', 'geo_cluster']].to_csv(
    'moscow_top_100_restaurants.csv', index=False, encoding='utf-8')

print("Сохранено 100 заведений в moscow_top_100_restaurants.csv")
print(f"Файл содержит: широту, долготу, адреса, контакты и scoring")

