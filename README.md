# DualPOI: A Dual-Modal Framework Integrating Deep Learning and Knowledge Graphs for Personalized Point-of-Interest Recommendation

## Abstract

This repository presents an evaluation of four POI (Point of Interest) recommendation models using real user features extracted from the FourSquare Moscow dataset. The study compares baseline and improved implementations of cluster-based and hybrid recommendation approaches, demonstrating significant improvements in both accuracy and representativeness when using authentic user behavioral patterns versus synthetic features.

**Additional Study:** This repository also includes a knowledge graph-based restaurant recommendation system using a manually curated dataset of 98 Moscow restaurants, implementing three recommendation schemes (Content-Based, Geographically Weighted, and Hybrid) without requiring user interaction data. See `docs/Knowledge_Graph_Methodology.md` and `docs/Dataset_Collection_Methodology.md` for details.

## Key Results

| Model | Accuracy | Representativeness | Status |
|-------|----------|-------------------|---------|
| Baseline Cluster | 18.7% | 0.921 | Highly Representative |
| Baseline Hybrid | 16.8% | 0.946 | Highly Representative |
| Improved Cluster | 24.11% | 0.804 | Highly Representative |
| Improved Hybrid | 19.59% | 0.855 | Highly Representative |

## Dataset

- **Source**: FourSquare Moscow POI Dataset (w11wo/FourSquare-Moscow-POI)
- **POI Coverage**: 600 top POIs (26.1% of total visits)
- **Clusters**: 12 semantic categories
- **User Features**: Real behavioral patterns extracted from visit sequences
- **Split**: 80% training, 20% validation (no user overlap)

## Model Architecture

### Baseline Models
- **Cluster Model**: Traditional cluster-based recommendation with synthetic user features
- **Hybrid Model**: Combined cluster and temporal features with synthetic user profiles

### Improved Models
- **Improved Cluster**: Enhanced cluster model with real user features and neighbor-aware semantic clustering
- **Improved Hybrid**: Advanced hybrid model with authentic user behavioral patterns

## Methodology

### Data Preprocessing
1. Semantic clustering of POIs into 12 balanced categories
2. Real user feature extraction from visit patterns
3. Sequence-based training example generation
4. User-based train/validation split

### Representativeness Analysis
- Data distribution analysis across clusters
- User separation validation (0% overlap)
- Temporal and spatial pattern analysis
- Model behavior and generalization assessment

### Evaluation Metrics
- Accuracy (top-1 recommendation)
- Hits@3 and Hits@5
- Representativeness score (0-1 scale)
- Representativeness analysis

## Installation

```bash
git clone https://github.com/trytohackmenoobie/rec_sys
cd rec_sys
pip install -r requirements.txt
```

**Note:** If you encounter build errors related to `pyarrow` (cmake required), install it via conda:
```bash
conda install pyarrow
```
Alternatively, install cmake:
```bash
brew install cmake  # macOS
```

## Usage

### Quick Start (Recommended)
```bash
# Interactive training guide
python scripts/quick_start.py

# Train specific model (uses exact parameters from our experiments)
python scripts/train_model.py --model improved_cluster
python scripts/train_model.py --model improved_hybrid
```

### IMPORTANT: Reproducibility
**For exact results reproduction, use default parameters:**
- Baseline models: 25-30 epochs (as configured)
- Improved Cluster: 10 epochs @ 0.001 learning rate
- Improved Hybrid: 20 epochs @ 0.0008 learning rate (48 batch, StepLR)
- **DO NOT** change epochs unless you want different results
- See `docs/training_parameters.md` for exact configuration

### Complete Pipeline
```bash
# Run everything: experiments + results + visualizations
python scripts/run_complete_pipeline.py
```

### Individual Training
```bash
# Train all models and collect results
python scripts/run_all_experiments.py

# Train specific models directly
python experiments/improved_cluster_experiment.py
python experiments/improved_hybrid_experiment.py
python scripts/baseline_cluster_model.py
python scripts/baseline_hybrid_model.py
```

### Results and Visualizations
```bash
# Collect and process results
python scripts/collect_all_results.py

# Generate publication-ready visualizations
python scripts/generate_all_visualizations.py

# Generate interactive knowledge graph visualizations
python scripts/generate_knowledge_graph_visualizations.py
```

### Viewing Interactive Visualizations


After generating the visualizations, you can view them in your web browser:

**Knowledge Graph Visualization:**
```bash
# Open in default browser (macOS)
open results/visualizations/knowledge_graph_interactive.html

# Open in default browser (Linux)
xdg-open results/visualizations/knowledge_graph_interactive.html

# Or simply double-click the file in your file manager
```

**Restaurant Map Visualization:**
```bash
# Open in default browser (macOS)
open results/visualizations/restaurant_map_100_interactive.html

# Open in default browser (Linux)
xdg-open results/visualizations/restaurant_map_100_interactive.html

# Or simply double-click the file in your file manager
```

Alternatively, you can manually open the HTML files:
- Navigate to `results/visualizations/` folder
- Double-click `knowledge_graph_interactive.html` or `restaurant_map_100_interactive.html`
- The visualizations will open in your default web browser

## Results

### Performance Analysis
- **Best Accuracy**: Improved Cluster Model (24.11%)
- **Best Representativeness**: Improved Hybrid Model (85.5%)
- **Average Improvement**: 14.9% over baseline models
- **Representativeness**: All models exceed 80% threshold

### Key Findings
1. Real user features significantly improve model performance
2. Improved models demonstrate superior generalization
3. All models achieve high representativeness scores
4. User-based evaluation prevents data leakage

## File Structure

```
├── README.md            # Project documentation and usage guide
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
│
├── experiments/         # Model experiments and analyzers
│   ├── improved_cluster_experiment.py    # Improved cluster model training
│   ├── improved_hybrid_experiment.py     # Improved hybrid model training
│   ├── baseline_cluster_analyzer.py      # Baseline cluster analysis
│   ├── baseline_hybrid_analyzer.py       # Baseline hybrid analysis
│   └── comprehensive_representativeness_analysis.py  # Representativeness evaluation
│
├── scripts/            # Automation and execution scripts
│   ├── run_complete_pipeline.py          # Full pipeline execution
│   ├── run_all_experiments.py            # Run all model experiments
│   ├── collect_all_results.py            # Collect experimental results
│   ├── generate_all_visualizations.py    # Generate all visualizations
│   ├── generate_knowledge_graph_visualizations.py  # KG interactive visualizations
│   ├── train_model.py                    # Train specific model
│   ├── quick_start.py                    # Interactive training guide
│   ├── run_notebook.py                   # Execute Jupyter notebooks
│   ├── baseline_cluster_model.py         # Baseline cluster training script
│   ├── baseline_hybrid_model.py          # Baseline hybrid training script
│   └── hybrid_model.py                   # Hybrid model training script
│
├── models/             # Trained model weights and metadata
│   ├── cluster_personalized_model.pth    # Baseline cluster model weights
│   ├── cluster_metadata.pkl              # Baseline cluster metadata
│   ├── cluster_info.json                 # Cluster information
│   ├── hybrid_personalized_model.pth     # Baseline hybrid model weights
│   ├── hybrid_metadata.pkl               # Baseline hybrid metadata
│   ├── improved_cluster_personalized_model.pth  # Improved cluster weights
│   ├── improved_cluster_metadata.pkl     # Improved cluster metadata
│   ├── hybrid/                           # Hybrid model code
│   │   └── hybrid_model.py
│   └── improved_cluster/                 # Improved cluster model code
│       └── cluster_model.py
│
├── results/            # Experimental results and visualizations
│   ├── metrics/        # Performance data (JSON, CSV)
│   │   ├── experimental_results.json     # All experiment results
│   │   ├── performance_summary.json      # Performance summary
│   │   ├── model_comparison.csv          # Model comparison table
│   │   └── raw_experiment_results.json   # Raw experiment data
│   ├── visualizations/ # Publication-ready figures
│   │   ├── knowledge_graph_interactive.html     # Interactive KG (98 restaurants)
│   │   ├── restaurant_map_100_interactive.html  # Interactive map (100 restaurants)
│   │   ├── accuracy_comparison.png              # Accuracy comparison chart
│   │   ├── performance_heatmap.png              # Performance heatmap
│   │   ├── performance_radar_chart.png          # Performance radar chart
│   │   └── model_comparison_scatter.png         # Model comparison scatter
│   ├── notebook_executions/              # Converted notebook scripts
│   │   └── frsqr_2025.py
│   └── pipeline_execution_summary.txt    # Pipeline execution log
│
├── notebooks/          # Jupyter notebooks for knowledge graph study
│   ├── frsqr_2025.ipynb                  # FourSquare S3 data parsing
│   └── recsys_knowledge_graph.ipynb      # Knowledge graph construction
│
├── data/               # Datasets
│   ├── raw/            # Raw data sources documentation
│   │   └── DATA_SOURCES.md               # Reference to original data sources
│   └── processed/      # Processed datasets
│       ├── workingrest.csv                       # Final dataset (98 establishments)
│       ├── moscow_top_100_restaurants.csv        # Top 100 for enrichment
│       ├── moscow_data_lens_export.csv           # DataLens export (1,355 establishments)
│       └── moscow_data_lens_small.csv            # Small export (1,000 establishments)
│       # DataLens Dashboard: https://datalens.yandex/w3048zrqxgtyh
│
├── docs/               # Documentation
│   ├── Abstract.md                         # Project abstract
│   ├── Methodology.md                      # Main methodology
│   ├── Results_Summary.md                  # Results summary
│   ├── Dataset_Collection_Methodology.md   # Data collection process
│   ├── Knowledge_Graph_Methodology.md      # Knowledge graph study
│   ├── Research_Integration_Summary.md     # Integration of both studies
│   └── training_parameters.md              # Training hyperparameters reference
│
└── dualpoi/            # Core model implementations and utilities
    ├── config.py                           # Configuration settings
    ├── models/                             # Model weights (duplicate for compatibility)
    │   ├── cluster_personalized_model.pth
    │   ├── cluster_metadata.pkl
    │   ├── hybrid_personalized_model.pth
    │   ├── hybrid_metadata.pkl
    │   ├── improved_cluster_personalized_model.pth
    │   └── improved_cluster_metadata.pkl
    └── utils/                              # Utility modules
        └── model.py                        # Model utilities
```

## Additional Research: Knowledge Graph-Based Recommendation

This repository also contains a parallel study on knowledge graph-based restaurant recommendation:

### Knowledge Graph Dataset
- **Source:** FourSquare Open Street Places (S3, September 2025)
- **Collection Process:** Multi-stage filtering from 60,964 → 98 establishments
- **Enrichment:** Manual data collection from [Яндекс Карты](https://yandex.ru/maps) (32 attributes per establishment)
- **Graph Structure:** 176 nodes, 623 edges (NetworkX)

### Recommendation Schemes
1. **Content-Based:** Precision@5 = 1.0000, Personalization = 0.7045
2. **Geographically Weighted:** Precision@5 = 0.7163, Geo Relevance = 0.4145
3. **Hybrid:** Precision@5 = 0.7755, NDCG@5 = 0.9188

**Documentation:**
- `docs/Dataset_Collection_Methodology.md` - Complete data collection process
- `docs/Knowledge_Graph_Methodology.md` - Knowledge graph construction and evaluation
- `docs/Research_Integration_Summary.md` - Integration of both studies

**Interactive Visualizations:**
- **Knowledge Graph**: `results/visualizations/knowledge_graph_interactive.html` - Interactive network graph (176 nodes, 623 edges)
  - Shows all restaurants, price levels, areas, atmosphere scores, and features
  - Interactive features: zoom, pan, drag nodes, click to inspect
- **Restaurant Map**: `results/visualizations/restaurant_map_100_interactive.html` - Interactive map of 98 restaurants in Moscow
  - Color-coded markers by rating (green ≥4.5, blue ≥4.0, orange ≥3.5, red <3.5)
  - Click markers to see restaurant details
- **Generate visualizations**: `python scripts/generate_knowledge_graph_visualizations.py`
- **View visualizations**: See "Viewing Interactive Visualizations" section above

## Reproducibility

All experiments are fully reproducible with:
- Fixed random seeds (42)
- Consistent data splits
- Automated pipeline execution
- Automated logging

## Documentation

### Main Studies
- **POI Recommendation:** See `docs/Methodology.md` and `docs/Results_Summary.md`
- **Knowledge Graph:** See `docs/Knowledge_Graph_Methodology.md`
- **Data Collection:** See `docs/Dataset_Collection_Methodology.md`
- **Integration:** See `docs/Research_Integration_Summary.md`

### Key Methodological Declarations
- **Data Collection Process:** Fully documented S3 extraction, filtering, scoring, clustering
- **Manual Enrichment:** Detailed process from [Яндекс Карты](https://yandex.ru/maps) (32 attributes, 98 establishments)
- **Knowledge Graph Construction:** NetworkX-based implementation (176 nodes, 623 edges)
- **Evaluation Metrics:** Comprehensive metrics for both studies

## Authors

**Student:** [Askhabaliev Gadzhi]  
**Supervisor:** Patochenko Evgeny


## Citation

```bibtex
@article{dual_poi_2025,
  title={DualPOI: A Dual-Modal Framework Integrating Deep Learning and Knowledge Graphs for Personalized Point-of-Interest Recommendation},
  author={Askhabaliev, Gadzhi},
  supervisor={Patochenko, Evgeny},
  year={2025},
  note={Includes two complementary studies: deep learning-based POI recommendation and knowledge graph-based restaurant recommendation}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.