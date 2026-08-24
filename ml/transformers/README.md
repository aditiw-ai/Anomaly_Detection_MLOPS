# Fraud Feature Engineer

## Overview

`FraudFeatureEngineer` is a scikit-learn compatible transformer for fraud detection feature engineering.

## Features Created

### Time Features
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `is_weekend`: Binary flag for weekend
- `is_night`: Binary flag for night hours (22:00-06:00)
- `day_of_month`: Day of month (1-31)

### Velocity Features (requires cache)
- `user_txn_count_1h`: Transaction count in last 1 hour
- `user_txn_count_24h`: Transaction count in last 24 hours

### Derived Features
- `amount_log`: Log transformation of amount
- `amount_sqrt`: Square root of amount
- `amount_vs_user_avg`: Ratio of amount to user average
- `merchant_risk_score`: Merchant fraud rate (learned during fit)

### Categorical Encoding
- `merchant_category_code`: Label-encoded merchant category
- `device_type_code`: Label-encoded device type
- `payment_method_code`: Label-encoded payment method

## Usage

### Basic Usage

```python
from ml.transformers.fraud_feature_engineer import FraudFeatureEngineer
import pandas as pd

# Load data
df = pd.read_csv('transactions.csv')

# Initialize transformer
transformer = FraudFeatureEngineer()

# Fit and transform
transformer.fit(df)
X_transformed = transformer.transform(df)
```

### With sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

pipeline = Pipeline([
    ('features', FraudFeatureEngineer()),
    ('model', XGBClassifier())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### With Redis Cache (for velocity features)

```python
import redis

redis_client = redis.Redis(host='localhost', port=6379)
transformer = FraudFeatureEngineer(cache=redis_client)
```

## Input Requirements

### Required Columns
- `amount` (float): Transaction amount
- `timestamp` (datetime): Transaction timestamp

### Optional Columns
- `user_id` (str): User identifier (for aggregations)
- `merchant_id` (str): Merchant identifier (for risk scores)
- `merchant_category` (str): Merchant category
- `device_type` (str): Device type
- `payment_method` (str): Payment method

## Output

All output columns are numeric (float32) and ready for tree-based models.

## Testing

```bash
# Run unit tests
pytest backend/tests/test_fraud_feature_engineer.py -v

# Run manual test
python backend/tests/manual_test_transformer.py
```

## Limitations

⚠️ **Current implementation is fraud-specific** with hardcoded column names.

For generic, dataset-agnostic feature engineering, see:
- Design doc: `generic_feature_engineering_design.md`
- Planned in Phase 0.5 (after ADR-008 completion)

## Test Results

✅ All 8 unit tests passing
✅ Manual test validates transformer with sample data
✅ All outputs are numeric (float32)
✅ Feature order is consistent across transforms
