# Medical-Anomaly-Detection
(NOTE: Files are hidden due to sensitive information. Contact me directly for more information. Additional details on my website (will post link later).

## Overview

This project is a machine learning-based anomaly detection system, aimed at identifying medical anomalies from a set of extracted features. Using Python and popular data science libraries, the goal was to implement a classifier to distinguish between "Normal" and "Anomalous" cases based on medical signal data.

The workflow involved data preprocessing, feature extraction, handling imbalanced classes, model training, evaluation, and optimization of threshold settings to improve classification performance.

## Approach and Methods

### Data Loading and Initial Exploration
Started by loading extracted features from CSV files and examining the initial distribution of labels in the dataset.
Preprocessing steps included label encoding, data normalization, and outlier detection.

### Anomaly Detection with Z-score Thresholding
Set a z-score threshold to classify anomalies in the feature set. Values above the threshold were labeled as "Anomalous," while the rest were labeled as "Normal."
This initial labeling provided a baseline for training the model.

### Handling Imbalanced Data with SMOTE
To address class imbalance, implemented SMOTE (Synthetic Minority Over-sampling Technique) to increase instances of the minority "Anomalous" class, improving the model’s ability to recognize anomalies.
This resampling technique enhanced training diversity and model performance.

### Model Training and Threshold Tuning
Used logistic regression for initial modeling due to its simplicity and interpretability.
To improve anomaly detection sensitivity, performed threshold tuning on predicted probabilities, aiming to balance precision and recall.

### Evaluation and Metrics
Evaluated the model’s performance using precision, recall, and f1-score metrics.
Adjusted the threshold iteratively to enhance recall for anomaly detection without sacrificing overall accuracy.

### Integration and Testing
Created a user-friendly function to accept new data input and output anomaly detection results, making the project more practical and usable.

## Lessons Learned

### Understanding Data Imbalance and Its Impacts
Imbalanced classes required special handling to prevent the model from heavily favoring the majority class. SMOTE and careful tuning helped achieve a balanced model.

### Feature Scaling and Preprocessing
Learned the importance of preprocessing steps like normalization to ensure that features contribute equally to the model’s decision-making.

### Model Threshold Tuning
While a standard 0.5 threshold is typical, adjusting it was crucial for anomaly detection, particularly in imbalanced datasets, where capturing anomalies is more valuable.

### Git and GitHub Workflow
Encountered challenges with Git and .gitignore configurations, which led to deeper knowledge of how Git tracks files and how to manage public/private data effectively.

### Documentation and User Accessibility
Gained insight into making a project more accessible by creating clean, structured scripts and functions that allow users to easily test the anomaly detection model on new data.
