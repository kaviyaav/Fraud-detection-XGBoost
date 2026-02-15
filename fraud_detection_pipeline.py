import pandas as pd

# Load dataset
fraud_data = pd.read_parquet(
    '../dataset/fraud_transactions.parquet',
    engine='pyarrow'
)

# Display settings
pd.set_option("display.max_columns", 300)
pd.set_option("display.max_rows", 1000)

# Preview data
fraud_data.head()

# Set index
fraud_data.set_index('TransactionID', inplace=True)

# Dataset overview
fraud_data.shape
fraud_data.dtypes.value_counts()
fraud_data.info()

def analyze_missing(dataframe, dtypes):
    row_count = dataframe.shape[0]
    null_count = dataframe.select_dtypes(include=dtypes).isnull().sum()
    null_pct = (null_count / row_count) * 100

    result = pd.DataFrame({
        'null_count': null_count,
        'null_percentage': null_pct.round(2)
    })

    return result[result['null_count'] > 0].sort_values(
        by='null_percentage', ascending=False
    )

# Missing values in numeric fields
analyze_missing(fraud_data, dtypes=['float32', 'int32'])

# Fill numeric features using median
numeric_features = fraud_data.select_dtypes(include='number').columns
for feature in numeric_features:
    fraud_data[feature].fillna(fraud_data[feature].median(), inplace=True)

# Fill categorical features using mode
categorical_features = fraud_data.select_dtypes(include=['object', 'category']).columns
for feature in categorical_features:
    most_common = fraud_data[feature].mode(dropna=True)
    if not most_common.empty:
        fraud_data[feature].fillna(most_common[0], inplace=True)

# Confirm no missing values
fraud_data.isnull().sum().sum()

"""
features_to_remove = set()
for col_a, col_b, _ in correlated_features:
    if col_a not in features_to_remove:
        features_to_remove.add(col_b)

fraud_data.drop(columns=list(features_to_remove), inplace=True)
"""

# Cardinality analysis
feature_cardinality = fraud_data[categorical_features].nunique().sort_values(ascending=False)
print(feature_cardinality)

from sklearn.preprocessing import LabelEncoder

# Encode high-cardinality column
email_encoder = LabelEncoder()
fraud_data['P_emaildomain'] = email_encoder.fit_transform(
    fraud_data['P_emaildomain']
)

# One-hot encode low-cardinality columns
fraud_data = pd.get_dummies(
    fraud_data,
    columns=['ProductCD', 'card4', 'card6', 'M6'],
    drop_first=True
)

fraud_data.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Class distribution
sns.countplot(x='isFraud', data=fraud_data)
plt.title('Fraud vs Non-Fraud Transactions')
plt.xlabel('Fraud Label')
plt.ylabel('Transaction Count')
plt.xticks([0, 1], ['Legitimate', 'Fraud'])
plt.show()

fraud_distribution = fraud_data['isFraud'].value_counts(normalize=True) * 100
print(fraud_distribution)

from sklearn.model_selection import train_test_split

features = fraud_data.drop(columns=['isFraud'])
target = fraud_data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

ml_pipeline = ImbPipeline(steps=[
    ('scaling', StandardScaler()),
    ('oversample', SMOTE(random_state=42)),
    ('model', xgb_model)
])

# Train model
ml_pipeline.fit(X_train, y_train)

# Predictions
predictions = ml_pipeline.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, predictions)
labels = ['Legitimate', 'Fraud']

plt.figure(figsize=(8, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))
