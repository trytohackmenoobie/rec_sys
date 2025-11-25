"""
Generate Interactive Visualizations for Knowledge Graph
- Interactive Knowledge Graph Visualization (pyvis)
- Interactive Restaurant Map (100 restaurants, folium)
"""

import pandas as pd
import networkx as nx
import folium
from pyvis.network import Network
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_and_preprocess_data():
    """Load and preprocess the restaurant dataset"""
    data_file = project_root / "data" / "processed" / "workingrest.csv"
    
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file, delimiter=';', on_bad_lines='skip')
    print(f"Loaded {len(df)} restaurants")
    
    # Basic preprocessing
    critical_columns = ['fsq_place_id', 'name', 'address', 'latitude', 'longitude']
    df.dropna(subset=critical_columns, inplace=True)
    
    # Numerical imputation
    numerical_cols_to_impute = ['score', 'geo_cluster', 'rating', 'min_price', 'max_price', 'atmosphere_score', 'avg_price_calculated']
    for col in numerical_cols_to_impute:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # String imputation
    na_impute_cols = ['tel', 'website', 'instagram', 'working_hours', 'open_time', 'close_time']
    for col in na_impute_cols:
        if col in df.columns:
            df[col] = df[col].fillna('N/A')
    
    # Mode imputation
    mode_impute_cols = ['type', 'is_central', 'price_level']
    for col in mode_impute_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'N/A')
    
    # Convert boolean columns
    if 'is_central' in df.columns:
        df['is_central'] = df['is_central'].astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)
    
    # Convert boolean feature columns
    boolean_cols = ['terrace', 'parking', 'menu_vegan', 'menu_seasonal', 'menu_grill',
                    'menu_kids', 'menu_diet', 'menu_exotic', 'menu_hot_dogs']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Process atmosphere column
    if 'atmosphere' in df.columns:
        df['atmosphere'] = df['atmosphere'].astype(str).str.replace('%', '').str.replace('\xa0', '').str.strip()
        df['atmosphere'] = pd.to_numeric(df['atmosphere'], errors='coerce')
    
    print(f"After preprocessing: {len(df)} restaurants")
    return df

def build_knowledge_graph(df):
    """Build the knowledge graph from the dataframe"""
    print("Building knowledge graph...")
    G = nx.DiGraph()
    
    for index, row in df.iterrows():
        restaurant_id = row['fsq_place_id']
        if pd.isna(restaurant_id):
            continue
        
        # Add restaurant node
        G.add_node(restaurant_id,
                   type='Restaurant',
                   name=row['name'],
                   address=row['address'],
                   latitude=row['latitude'],
                   longitude=row['longitude'],
                   tel=row.get('tel', 'N/A'),
                   website=row.get('website', 'N/A'),
                   instagram=row.get('instagram', 'N/A'),
                   score=row.get('score', 0),
                   rating=row.get('rating', 0),
                   working_hours=row.get('working_hours', 'N/A'),
                   open_time=row.get('open_time', 'N/A'),
                   close_time=row.get('close_time', 'N/A'),
                   atmosphere_score=row.get('atmosphere_score', 0),
                   avg_price_calculated=row.get('avg_price_calculated', 0),
                   hours_of_operation=row.get('hours_of_operation', 0),
                   is_central=row.get('is_central', False)
                  )
        
        # Add price level node and edge
        price_level = row.get('price_level')
        if pd.notna(price_level) and price_level != 'N/A':
            price_level_node_id = f"PriceLevel_{price_level}"
            if not G.has_node(price_level_node_id):
                G.add_node(price_level_node_id, type='PriceLevel', name=price_level)
            G.add_edge(restaurant_id, price_level_node_id, relation='has_price_level')
        
        # Add area (metro) node and edge
        metro_area = row.get('metro')
        if pd.notna(metro_area) and metro_area != 'N/A':
            area_node_id = f"Area_{metro_area}"
            if not G.has_node(area_node_id):
                G.add_node(area_node_id, type='Area', name=metro_area)
            G.add_edge(restaurant_id, area_node_id, relation='located_in')
        
        # Add atmosphere node and edge
        atmosphere_val = row.get('atmosphere')
        if pd.notna(atmosphere_val):
            try:
                atmosphere_score_int = int(float(atmosphere_val))
                atmosphere_node_id = f"Atmosphere_{atmosphere_score_int}"
                if not G.has_node(atmosphere_node_id):
                    G.add_node(atmosphere_node_id, type='Atmosphere', score=atmosphere_score_int)
                G.add_edge(restaurant_id, atmosphere_node_id, relation='has_atmosphere')
            except (ValueError, TypeError):
                pass
        
        # Add feature nodes and edges
        feature_cols = ['terrace', 'parking', 'menu_vegan', 'menu_seasonal', 'menu_grill',
                        'menu_kids', 'menu_diet', 'menu_exotic', 'menu_hot_dogs']
        for col in feature_cols:
            if col in row and row[col]:
                feature_name = col.replace('_', ' ').title()
                feature_node_id = f"Feature_{feature_name}"
                if not G.has_node(feature_node_id):
                    G.add_node(feature_node_id, type='Feature', name=feature_name)
                G.add_edge(restaurant_id, feature_node_id, relation='offers_feature')
    
    print(f"Knowledge graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def create_knowledge_graph_visualization(G, output_path):
    """Create interactive knowledge graph visualization using pyvis"""
    print("Creating knowledge graph visualization...")
    
    net = Network(height="800px", width="100%",
                  bgcolor="#ffffff", font_color="black",
                  notebook=False, cdn_resources='in_line')
    
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 500},
        "solver": "forceAtlas2Based"
      }
    }
    """)
    
    # Add nodes
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'Unknown')
        label = G.nodes[node].get('name', str(node))
        
        if node_type == 'Restaurant':
            net.add_node(node, label=label[:20] if len(label) > 20 else label, color='#97c2fc', size=20)
        elif node_type == 'PriceLevel':
            net.add_node(node, label=f"${label}", color='#90EE90', size=12)
        elif node_type == 'Area':
            net.add_node(node, label=label, color='#FFFFE0', size=15)
        elif node_type == 'Atmosphere':
            net.add_node(node, label=f"A:{label}", color='#FFA07A', size=12)
        else:
            label_clean = label.replace('Feature_', '') if 'Feature_' in str(label) else label
            net.add_node(node, label=label_clean, color='#FFB6C1', size=8)
    
    # Add edges
    for edge in G.edges():
        net.add_edge(edge[0], edge[1], color='gray', width=0.3)
    
    # Save to HTML file
    net.save_graph(str(output_path))
    print(f"Knowledge graph visualization saved to {output_path}")

def create_restaurant_map_visualization(G, output_path):
    """Create interactive map of 100 restaurants using folium"""
    print("Creating restaurant map visualization...")
    
    # Find restaurants with coordinates
    restaurants_with_coords = []
    for node in G.nodes():
        if (G.nodes[node].get('type') == 'Restaurant' and
            'latitude' in G.nodes[node] and 'longitude' in G.nodes[node] and
            G.nodes[node].get('latitude') and G.nodes[node].get('longitude')):
            restaurants_with_coords.append(node)
    
    if not restaurants_with_coords:
        print("No coordinate data available")
        return None
    
    # Calculate map center - average coordinates
    avg_lat = sum(G.nodes[n]['latitude'] for n in restaurants_with_coords) / len(restaurants_with_coords)
    avg_lon = sum(G.nodes[n]['longitude'] for n in restaurants_with_coords) / len(restaurants_with_coords)
    
    # Create map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
    
    # Add markers for first 100 restaurants
    restaurants_to_show = restaurants_with_coords[:100]
    for restaurant in restaurants_to_show:
        name = G.nodes[restaurant].get('name', 'Unknown')
        rating = G.nodes[restaurant].get('rating', 0)
        price = G.nodes[restaurant].get('avg_price_calculated', 'N/A')
        address = G.nodes[restaurant].get('address', 'No address')
        
        # Determine icon color based on rating
        if rating >= 4.5:
            icon_color = 'green'
        elif rating >= 4.0:
            icon_color = 'blue'
        elif rating >= 3.5:
            icon_color = 'orange'
        else:
            icon_color = 'red'
        
        # Create popup content
        popup_content = f"""
        <b>{name}</b><br>
        <b>Rating:</b> {rating}/5<br>
        <b>Average Price:</b> {price}<br>
        <b>Address:</b> {address}<br>
        """
        
        folium.Marker(
            [G.nodes[restaurant]['latitude'], G.nodes[restaurant]['longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=name,
            icon=folium.Icon(color=icon_color, icon='cutlery', prefix='fa')
        ).add_to(m)
    
    # Save map to HTML file
    m.save(str(output_path))
    print(f"Restaurant map visualization saved to {output_path}")
    print(f"Showing {len(restaurants_to_show)} restaurants on the map")

def main():
    """Main function to generate both visualizations"""
    print("=" * 60)
    print("Generating Knowledge Graph Visualizations")
    print("=" * 60)
    
    # Setup output directory
    output_dir = project_root / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Build knowledge graph
    G = build_knowledge_graph(df)
    
    # Generate visualizations
    kg_viz_path = output_dir / "knowledge_graph_interactive.html"
    map_viz_path = output_dir / "restaurant_map_100_interactive.html"
    
    create_knowledge_graph_visualization(G, kg_viz_path)
    create_restaurant_map_visualization(G, map_viz_path)
    
    print("\n" + "=" * 60)
    print("Visualizations generated successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. Knowledge Graph: {kg_viz_path}")
    print(f"  2. Restaurant Map (100): {map_viz_path}")
    print(f"\nOpen these HTML files in a web browser to view the interactive visualizations.")

if __name__ == "__main__":
    main()

