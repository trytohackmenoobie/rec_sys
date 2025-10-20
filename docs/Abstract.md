# Abstract

## Point of Interest Recommendation with Real User Behavioral Features: An Evaluation

### Background

Point of Interest (POI) recommendation systems face significant challenges in accurately modeling user preferences and ensuring representative evaluation methodologies. Traditional approaches often rely on synthetic or simplified user features, potentially limiting model generalizability and evaluation reliability.

### Objective

This study presents an evaluation of four POI recommendation models using authentic user behavioral features extracted from real-world check-in data, comparing baseline implementations against improved approaches that leverage genuine user interaction patterns.

### Methods

We evaluated four recommendation models using the FourSquare Moscow POI dataset: two baseline models employing synthetic user features (Baseline Cluster and Baseline Hybrid) and two improved models utilizing real user behavioral patterns (Improved Cluster and Improved Hybrid). The dataset comprised 600 top POIs across 12 semantic clusters, with user features extracted from actual visit sequences. Models were evaluated using accuracy, Hits@3, Hits@5, and a representativeness score incorporating data distribution, user separation, temporal patterns, spatial patterns, model behavior, and generalization analysis.

### Results

The improved models demonstrated superior performance across all metrics. The Improved Cluster Model achieved the highest accuracy (21.56%), while the Improved Hybrid Model attained the highest representativeness score (0.953). All models exceeded the 92% representativeness threshold, indicating reliable and reproducible results. The baseline models achieved accuracies of 18.7% (Cluster) and 16.8% (Hybrid), with representativeness scores of 0.921 and 0.946, respectively.

### Conclusions

Real user behavioral features significantly enhance POI recommendation performance compared to synthetic alternatives. The improved models demonstrate superior generalization capabilities while maintaining high representativeness, suggesting that authentic user interaction patterns are crucial for developing robust recommendation systems. These findings have important implications for POI recommendation system design and evaluation methodologies.

### Keywords

Point of Interest Recommendation, User Behavior Modeling, Representativeness Analysis, Real-World Evaluation, Machine Learning, Location-Based Services
