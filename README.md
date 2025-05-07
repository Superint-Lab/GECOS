# GECOS
Urban Mobile Data Prediction with Geospatial Clustering and Dual Residual Learning

- Project Overview: This repository provides implementations for correlation-based clustering and the Residual Convolutional Temporal Learning (RCTL) model, which are tailored for sequential data prediction tasks such as mobile network traffic forecasting.

- Dataset Description: The dataset used in clustering (correlation_matrix_0721.csv) contains a correlation matrix representing the pairwise similarities among variables (e.g., mobile network cells or sensors). Each cell value denotes the correlation strength between two variables, guiding the grouping into clusters with high internal similarity.

- Clustering Methodology: The clustering algorithm groups variables based on maximizing the average correlation within each cluster. The optimization iteratively reallocates variables to different clusters to enhance intra-cluster similarity, ultimately converging to an optimized distribution of correlated variables.

- RCTL Model Explanation: The RCTL architecture integrates convolutional, recurrent (LSTM), and residual connections, effectively capturing temporal dependencies and spatial features of time-series data. Residual connections improve gradient flow and model performance, making RCTL suitable for complex sequential prediction tasks.
