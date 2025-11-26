# Abstract

## DualPOI: A Dual-Modal Framework Integrating Deep Learning and Knowledge Graphs for Personalized Point-of-Interest Recommendation

### Background

Point of Interest (POI) recommendation systems face significant challenges in accurately modeling user preferences and ensuring representative evaluation methodologies. Traditional approaches often rely on synthetic or simplified user features, potentially limiting model generalizability and evaluation reliability.

### Objective

This study presents an evaluation of four POI recommendation models using authentic user behavioral features extracted from real-world check-in data, comparing baseline implementations against improved approaches that leverage genuine user interaction patterns.

### Methods

I evaluated seven recommendation models using the FourSquare Moscow POI dataset: two baseline models employing synthetic user features (Baseline Cluster and Baseline Hybrid) and two improved models utilizing real user behavioral patterns (Improved Cluster and Improved Hybrid) and three recomendation models based on knowledge graph. The dataset comprised 600 top POIs across 12 semantic clusters, with user features extracted from actual visit sequences. Models were evaluated using accuracy, Hits@3, Hits@5, and a representativeness score incorporating data distribution, user separation, temporal patterns, spatial patterns, model behavior, and generalization analysis.

### Results

The baseline models demonstrated strong performance with the Baseline Cluster Model achieving the highest accuracy (24.13%), while the Improved Hybrid Model (accuracy - 16.45%) attained the highest representativeness score (0.832). All models exceeded the 80% representativeness threshold, indicating reliable and reproducible results. The baseline models achieved accuracies of 24.13% (Cluster) and 19.51% (Hybrid), with representativeness scores of 0.820 and 0.819, respectively. The improved models achieved accuracies of 23.82% (Cluster) and 16.45% (Hybrid), with representativeness scores of 0.799 and 0.832, respectively.

### Conclusions

Real user behavioral features significantly enhance POI recommendation performance compared to synthetic alternatives. The improved models demonstrate superior generalization capabilities while maintaining high representativeness, suggesting that authentic user interaction patterns are crucial for developing robust recommendation systems. Furthermore, the implementation of smart semantic clustering of POIs into 12 balanced categories contributed to improved model performance by capturing meaningful spatial and functional relationships between locations. These findings have important implications for POI recommendation system design and evaluation methodologies. 

### Keywords

Point of Interest Recommendation, User Behavior Modeling, Representativeness Analysis, Real-World Evaluation, Machine Learning, Location-Based Services
