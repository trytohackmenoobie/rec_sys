"""
Visualization Generation Script
Academic Implementation

This script generates all visualizations for the POI Recommender System
based on experimental results. Creates publication-ready figures for
academic presentations and reports.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Professional color palette
PROFESSIONAL_COLORS = {
    'primary_red': '#8C1515',
    'cool_grey': '#4D4F53', 
    'light_sand': '#F4F4F4',
    'stone': '#DAD7CB',
    'black': '#2E2D29',
    'bright_red': '#B1040E',
    'success_green': '#008566'
}

# Set professional plotting style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#F0F0F0',
    'grid.alpha': 0.7
})

def load_experimental_results():
    """Load processed experimental results"""
    results_file = project_root / "results" / "metrics" / "experimental_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Experimental results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def create_accuracy_comparison_chart(results_data):
    """Create accuracy comparison bar chart"""
    
    successful_models = {
        k: v for k, v in results_data["models"].items() 
        if v["status"] == "success"
    }
    
    if not successful_models:
        print("No successful models to visualize")
        return None
    
    # Prepare data
    model_names = []
    accuracies = []
    representativeness_scores = []
    
    for model_key, model_data in successful_models.items():
        model_names.append(model_data.get("model_name", model_key))
        accuracies.append(model_data.get("accuracy", 0.0))
        representativeness_scores.append(model_data.get("representativeness", 0.0))
    
    # Create figure with professional styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Accuracy chart
    bars1 = ax1.bar(model_names, accuracies, alpha=0.8, 
                   color=PROFESSIONAL_COLORS['primary_red'], 
                   edgecolor=PROFESSIONAL_COLORS['black'], 
                   linewidth=1.5)
    ax1.set_title('Model Accuracy Comparison', fontweight='bold', color=PROFESSIONAL_COLORS['black'])
    ax1.set_ylabel('Accuracy', color=PROFESSIONAL_COLORS['cool_grey'])
    ax1.set_ylim(0, max(accuracies) * 1.1)
    ax1.tick_params(axis='x', rotation=45, colors=PROFESSIONAL_COLORS['cool_grey'])
    ax1.tick_params(axis='y', colors=PROFESSIONAL_COLORS['cool_grey'])
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', 
                fontweight='bold', color=PROFESSIONAL_COLORS['black'])
    
    # Representativeness chart
    bars2 = ax2.bar(model_names, representativeness_scores, alpha=0.8, 
                   color=PROFESSIONAL_COLORS['bright_red'], 
                   edgecolor=PROFESSIONAL_COLORS['black'], 
                   linewidth=1.5)
    ax2.set_title('Model Representativeness Comparison', fontweight='bold', color=PROFESSIONAL_COLORS['black'])
    ax2.set_ylabel('Representativeness Score', color=PROFESSIONAL_COLORS['cool_grey'])
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='x', rotation=45, colors=PROFESSIONAL_COLORS['cool_grey'])
    ax2.tick_params(axis='y', colors=PROFESSIONAL_COLORS['cool_grey'])
    
    # Add value labels on bars
    for bar, repr_score in zip(bars2, representativeness_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{repr_score:.3f}', ha='center', va='bottom', 
                fontweight='bold', color=PROFESSIONAL_COLORS['black'])
    
    plt.tight_layout()
    return fig

def create_performance_heatmap(results_data):
    """Create performance metrics heatmap"""
    
    successful_models = {
        k: v for k, v in results_data["models"].items() 
        if v["status"] == "success"
    }
    
    if not successful_models:
        print("No successful models to visualize")
        return None
    
    # Prepare data matrix
    model_names = [model_data["model_name"] for model_data in successful_models.values()]
    metrics = ['Accuracy', 'Representativeness', 'Hits@3', 'Hits@5']
    
    data_matrix = []
    for model_data in successful_models.values():
        data_matrix.append([
            model_data.get("accuracy", 0.0),
            model_data.get("representativeness", 0.0),
            model_data.get("hits_at_3", 0.0),
            model_data.get("hits_at_5", 0.0)
        ])
    
    # Create DataFrame
    df = pd.DataFrame(data_matrix, index=model_names, columns=metrics)
    
    # Create heatmap with professional styling
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create custom colormap from stone to primary red
    from matplotlib.colors import LinearSegmentedColormap
    colors = [PROFESSIONAL_COLORS['stone'], PROFESSIONAL_COLORS['primary_red']]
    cmap = LinearSegmentedColormap.from_list('professional', colors, N=256)
    
    sns.heatmap(df, annot=True, cmap=cmap, fmt='.3f', 
                cbar_kws={'label': 'Performance Score'}, ax=ax,
                linewidths=1, linecolor=PROFESSIONAL_COLORS['light_sand'])
    
    ax.set_title('Model Performance Heatmap', fontweight='bold', 
                color=PROFESSIONAL_COLORS['black'], pad=20)
    ax.set_xlabel('Performance Metrics', color=PROFESSIONAL_COLORS['cool_grey'])
    ax.set_ylabel('Models', color=PROFESSIONAL_COLORS['cool_grey'])
    
    plt.tight_layout()
    return fig

def create_model_comparison_scatter(results_data):
    """Create scatter plot comparing accuracy vs representativeness"""
    
    successful_models = {
        k: v for k, v in results_data["models"].items() 
        if v["status"] == "success"
    }
    
    if not successful_models:
        print("No successful models to visualize")
        return None
    
    # Prepare data
    x_values = []
    y_values = []
    labels = []
    colors = []
    
    color_map = {
        'baseline_cluster': PROFESSIONAL_COLORS['cool_grey'],
        'baseline_hybrid': PROFESSIONAL_COLORS['stone'], 
        'improved_cluster': PROFESSIONAL_COLORS['primary_red'],
        'improved_hybrid': PROFESSIONAL_COLORS['bright_red']
    }
    
    for model_key, model_data in successful_models.items():
        x_values.append(model_data["accuracy"])
        y_values.append(model_data["representativeness"])
        labels.append(model_data["model_name"])
        colors.append(color_map.get(model_key, 'gray'))
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, (x, y, label, color) in enumerate(zip(x_values, y_values, labels, colors)):
        ax.scatter(x, y, c=color, s=200, alpha=0.8, 
                  edgecolors=PROFESSIONAL_COLORS['black'], linewidth=2)
        ax.annotate(label, (x, y), xytext=(8, 8), textcoords='offset points',
                   fontsize=11, fontweight='bold', color=PROFESSIONAL_COLORS['black'])
    
    # Add reference lines
    ax.axhline(y=0.8, color=PROFESSIONAL_COLORS['success_green'], 
              linestyle='--', alpha=0.7, linewidth=2, 
              label='High Representativeness (0.8)')
    ax.axvline(x=0.2, color=PROFESSIONAL_COLORS['cool_grey'], 
              linestyle='--', alpha=0.7, linewidth=2, 
              label='Good Accuracy (0.2)')
    
    ax.set_xlabel('Accuracy', fontsize=12, color=PROFESSIONAL_COLORS['cool_grey'])
    ax.set_ylabel('Representativeness Score', fontsize=12, color=PROFESSIONAL_COLORS['cool_grey'])
    ax.set_title('Accuracy vs Representativeness Comparison', fontweight='bold', 
                color=PROFESSIONAL_COLORS['black'])
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=PROFESSIONAL_COLORS['light_sand'])
    
    # Set axis limits
    ax.set_xlim(0, max(x_values) * 1.1)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    return fig

def create_performance_radar_chart(results_data):
    """Create radar chart comparing all performance metrics"""
    
    successful_models = {
        k: v for k, v in results_data["models"].items() 
        if v["status"] == "success"
    }
    
    if not successful_models:
        print("No successful models to visualize")
        return None
    
    # Define metrics
    metrics = ['Accuracy', 'Representativeness', 'Hits@3', 'Hits@5']
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    color_map = {
        'baseline_cluster': PROFESSIONAL_COLORS['cool_grey'],
        'baseline_hybrid': PROFESSIONAL_COLORS['stone'],
        'improved_cluster': PROFESSIONAL_COLORS['primary_red'], 
        'improved_hybrid': PROFESSIONAL_COLORS['bright_red']
    }
    
    for model_key, model_data in successful_models.items():
        values = [
            model_data.get("accuracy", 0.0),
            model_data.get("representativeness", 0.0),
            model_data.get("hits_at_3", 0.0),
            model_data.get("hits_at_5", 0.0)
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=3, label=model_data["model_name"],
               color=color_map.get(model_key, PROFESSIONAL_COLORS['cool_grey']))
        ax.fill(angles, values, alpha=0.3, color=color_map.get(model_key, PROFESSIONAL_COLORS['cool_grey']))
    
    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color=PROFESSIONAL_COLORS['cool_grey'])
    ax.set_ylim(0, 1)
    
    ax.set_title('Model Performance Radar Chart', fontweight='bold', 
                color=PROFESSIONAL_COLORS['black'], pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
             frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=PROFESSIONAL_COLORS['light_sand'])
    
    return fig

def save_visualizations(figures):
    """Save all generated visualizations"""
    
    visualizations_dir = project_root / "results" / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    figure_names = [
        'accuracy_comparison',
        'performance_heatmap', 
        'model_comparison_scatter',
        'performance_radar_chart'
    ]
    
    for fig, name in zip(figures, figure_names):
        if fig is not None:
            file_path = visualizations_dir / f"{name}.png"
            fig.savefig(file_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            saved_files[name] = file_path
            plt.close(fig)
    
    return saved_files

def main():
    """Main visualization generation function"""
    
    print("POI RECOMMENDER SYSTEM - VISUALIZATION GENERATION")
    print("=" * 55)
    
    try:
        # Load experimental results
        print("Loading experimental results...")
        results_data = load_experimental_results()
        
        successful_count = sum(1 for model in results_data["models"].values() if model["status"] == "success")
        print(f"Found {successful_count} successful experiments")
        
        if successful_count == 0:
            print("No successful experiments found. Cannot generate visualizations.")
            return
        
        # Generate visualizations
        print("Generating visualizations...")
        
        figures = []
        
        print("  Creating accuracy comparison chart...")
        figures.append(create_accuracy_comparison_chart(results_data))
        
        print("  Creating performance heatmap...")
        figures.append(create_performance_heatmap(results_data))
        
        print("  Creating model comparison scatter plot...")
        figures.append(create_model_comparison_scatter(results_data))
        
        print("  Creating performance radar chart...")
        figures.append(create_performance_radar_chart(results_data))
        
        # Save visualizations
        print("Saving visualizations...")
        saved_files = save_visualizations(figures)
        
        print("VISUALIZATION GENERATION COMPLETED")
        print("=" * 55)
        print("Generated files:")
        for name, file_path in saved_files.items():
            print(f"  {name}: {file_path}")
        
        print(f"\nTotal visualizations generated: {len(saved_files)}")
        
    except Exception as e:
        print(f"ERROR: Visualization generation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
