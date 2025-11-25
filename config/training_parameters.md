# Training Parameters - Exact Configuration

## CRITICAL: Use these exact parameters for reproducible results

### Baseline Models

#### Baseline Cluster Model
- **Epochs**: 25
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Optimizer**: Adam
- **Weight Decay**: 1e-4
- **Scheduler**: StepLR (step_size=8, gamma=0.9)
- **User Features**: Synthetic (generated)

#### Baseline Hybrid Model
- **Epochs**: 30
- **Learning Rate**: 0.0008
- **Batch Size**: 32
- **Optimizer**: Adam
- **Weight Decay**: 1e-4
- **Scheduler**: StepLR (step_size=8, gamma=0.9)
- **User Features**: Synthetic (generated)

### Improved Models

#### Improved Cluster Model
- **Epochs**: 10
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Optimizer**: Adam
- **Weight Decay**: 1e-4
- **User Features**: Real (extracted from FourSquare)

#### Improved Hybrid Model
- **Epochs**: 10
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Optimizer**: Adam
- **User Features**: Real (extracted from FourSquare)

## Dataset Configuration

### FourSquare Moscow Dataset
- **Total POIs**: 600 (top by frequency)
- **Clusters**: 12 semantic categories
- **Train/Val Split**: 80%/20% (user-based, no overlap)
- **Random Seed**: 42 (fixed for reproducibility)

### Model Architecture
- **Hidden Dimensions**: 64
- **Dropout**: 0.3-0.4
- **Embedding Dimensions**: 32 (clusters), 16 (users)
- **Sequence Length**: Variable (2+ POIs)

## Expected Results (with these parameters)

| Model | Accuracy | Representativeness | Epochs Used |
|-------|----------|-------------------|-------------|
| Baseline Cluster | 18.7% | 0.921 | 25 |
| Baseline Hybrid | 16.8% | 0.946 | 30 |
| Improved Cluster | 24.11% | 0.804 | 10 |
| Improved Hybrid | 19.21% | 0.953 | 10 |

## IMPORTANT NOTES

1. **DO NOT CHANGE** these parameters if you want to reproduce our results
2. **Baseline models** use more epochs because they start with synthetic features
3. **Improved models** use fewer epochs because they have real user features
4. **All models** use the same random seed (42) for reproducibility
5. **User features** are the key difference between baseline and improved models

## Verification Commands

```bash
# Verify baseline parameters
python scripts/train_model.py --model baseline_cluster  # Should use 25 epochs
python scripts/train_model.py --model baseline_hybrid   # Should use 30 epochs

# Verify improved parameters  
python scripts/train_model.py --model improved_cluster  # Should use 10 epochs
python scripts/train_model.py --model improved_hybrid   # Should use 10 epochs
```
