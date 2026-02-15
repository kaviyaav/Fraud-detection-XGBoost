# Fraud Detection Using Machine Learning

## Project Overview

This project implements an end-to-end machine learning pipeline to detect fraudulent financial transactions using the IEEE-CIS dataset. It focuses on robust data preprocessing, handling class imbalance, and training a scalable model suitable for real-world fraud detection scenarios.

## Key Features

* Data loading and exploration from Parquet format
* Missing value handling for numerical and categorical features
* Categorical feature encoding (Label Encoding & One-Hot Encoding)
* Class imbalance handling using SMOTE
* End-to-end ML pipeline with scaling and XGBoost
* Model evaluation using confusion matrix and classification metrics

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Matplotlib, Seaborn

## Project Structure

```
├── fraud_detection_pipeline.py
├── dataset/
│   └── fraud_transactions.parquet
├── README.md
```

## How to Run

1. Install dependencies:

   ```bash
   pip install pandas scikit-learn xgboost imbalanced-learn seaborn matplotlib pyarrow
   ```
2. Place the dataset in the `dataset/` directory.
3. Run the pipeline:

   ```bash
   python fraud_detection_pipeline.py
   ```

## Model Approach

The model uses an XGBoost classifier trained within a pipeline that includes feature scaling and SMOTE oversampling to address severe class imbalance. This approach improves fraud detection recall while maintaining overall model stability.

## Results

Model performance is evaluated using a confusion matrix and classification report, providing insights into precision, recall, and overall fraud detection effectiveness.

## Authors

Kaviyaa Vasudevan & Arun Kumar Srinivasan
