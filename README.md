# POI Recommender System: Evaluation with Real User Features

## Abstract

This repository presents an evaluation of four POI (Point of Interest) recommendation models using real user features extracted from the FourSquare Moscow dataset. The study compares baseline and improved implementations of cluster-based and hybrid recommendation approaches, demonstrating significant improvements in both accuracy and representativeness when using authentic user behavioral patterns versus synthetic features.

## Key Results

| Model | Accuracy | Representativeness | Status |
|-------|----------|-------------------|---------|
| Baseline Cluster | 18.7% | 0.921 | Highly Representative |
| Baseline Hybrid | 16.8% | 0.946 | Highly Representative |
| Improved Cluster | 21.56% | 0.947 | Highly Representative |
| Improved Hybrid | 19.21% | 0.953 | Highly Representative |

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
- **Improved Cluster**: Enhanced cluster model with real user features from FourSquare
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
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
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
- Improved models: 10 epochs (as configured)
- **DO NOT** change epochs unless you want different results
- See `config/training_parameters.md` for exact configuration

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
```

## Results

### Performance Analysis
- **Best Accuracy**: Improved Cluster Model (21.56%)
- **Best Representativeness**: Improved Hybrid Model (95.3%)
- **Average Improvement**: 14.9% over baseline models
- **Representativeness**: All models exceed 92% threshold

### Key Findings
1. Real user features significantly improve model performance
2. Improved models demonstrate superior generalization
3. All models achieve high representativeness scores
4. User-based evaluation prevents data leakage

## File Structure

```
├── experiments/          # Model experiments and analyzers
├── scripts/             # Automation and execution scripts
├── models/              # Trained model weights and metadata
├── results/             # Experimental results and visualizations
│   ├── metrics/         # Performance data (JSON, CSV)
│   └── visualizations/  # Publication-ready figures
└── POI_RECOMMENDER/     # Core model implementations
```

## Reproducibility

All experiments are fully reproducible with:
- Fixed random seeds (42)
- Consistent data splits
- Automated pipeline execution
- Automated logging

## Citation

```bibtex
@article{poi_recommender_2025,
  title={POI Recommender System: Evaluation with Real User Features},
  author={[Askhabaliev Gadzhi]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.