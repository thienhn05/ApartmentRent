"""
Training Script: Logistic Regression Model for Cheap vs Expensive Apartment Classification
Saves the trained model as a pickle file for use in Streamlit
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 60)

# --- 1. Load Your Real Dataset ---
print("\n1️⃣  Loading dataset...")

# Load your clean apartment data
df = pd.read_csv('clean_apartment_data.csv')

print(f"   ✓ Dataset loaded: {len(df)} samples")
print(f"   ✓ Columns: {list(df.columns)}")
print(f"   ✓ Shape: {df.shape}")

# Display basic info
print(f"\n   Dataset Preview:")
print(f"   - Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
print(f"   - Average price: ${df['price'].mean():.2f}")
print(f"   - Records with missing values: {df.isnull().sum().sum()}")

# --- 2. Use Existing Target Variable ---
print("\n2️⃣  Checking classification target...")

# Your dataset already has 'is_expensive' column
if 'is_expensive' not in df.columns:
    print("   ⚠️  'is_expensive' column not found. Creating from price threshold...")
    threshold = df['price'].quantile(0.80)
    df['is_expensive'] = (df['price'] > threshold).astype(int)
    print(f"   ✓ Threshold set to: ${threshold:,.2f}")
else:
    threshold = df[df['is_expensive'] == 1]['price'].min()
    print(f"   ✓ Using existing 'is_expensive' column")
    print(f"   ✓ Approximate threshold: ${threshold:,.2f}")

y = df['is_expensive']
cheap_count = (y == 0).sum()
expensive_count = (y == 1).sum()

print(f"   ✓ Class distribution:")
print(f"      Cheap (0): {cheap_count} samples ({cheap_count/len(y)*100:.1f}%)")
print(f"      Expensive (1): {expensive_count} samples ({expensive_count/len(y)*100:.1f}%)")

# --- 3. Feature Selection ---
print("\n3️⃣  Preparing features...")

# Select features from your dataset
feature_cols = [
    'bathrooms', 
    'bedrooms', 
    'square_feet',
    'housing/rent/apartment',
    'housing/rent/condo',
    'fee_yes',
    'city_avg_price',
    'state'  # Already numeric in your dataset
]

# Check which features exist and handle missing ones
available_features = [col for col in feature_cols if col in df.columns]
missing_features = [col for col in feature_cols if col not in df.columns]

if missing_features:
    print(f"   ⚠️  Missing features: {missing_features}")
    print(f"   ✓ Using available features: {available_features}")
    feature_cols = available_features

# Handle missing values - fill with median/mode
for col in feature_cols:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"   ✓ Filled {col} missing values with median")
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"   ✓ Filled {col} missing values with mode")

X = df[feature_cols]
y = df['is_expensive']

print(f"   ✓ Features selected: {len(feature_cols)} features")
print(f"   ✓ Feature names: {feature_cols}")
print(f"   ✓ Feature matrix shape: {X.shape}")
print(f"   ✓ Target variable shape: {y.shape}")

# --- 4. Train/Test Split ---
print("\n4️⃣  Splitting data (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training set: {len(X_train)} samples")
print(f"   ✓ Test set: {len(X_test)} samples")

# --- 5. Create Pipeline ---
print("\n5️⃣  Building pipeline...")

# Pipeline with scaling, polynomial features, and logistic regression
pipeline = Pipeline([
    ('scaler_init', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler_final', StandardScaler()),
    ('logreg', LogisticRegression(solver='lbfgs', max_iter=5000, random_state=42))
])

print(f"   ✓ Pipeline steps:")
print(f"      • StandardScaler (initial)")
print(f"      • PolynomialFeatures (degree=2, interactions)")
print(f"      • StandardScaler (final)")
print(f"      • LogisticRegression")

# --- 6. Train Model ---
print("\n6️⃣  Training model...")

pipeline.fit(X_train, y_train)

print(f"   ✓ Model trained successfully!")

# --- 7. Evaluate Model ---
print("\n7️⃣  Evaluating model...")

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, 
                                      target_names=['Cheap', 'Expensive'],
                                      digits=3)

print(f"\n   📊 PERFORMANCE METRICS:")
print(f"   ✓ Accuracy: {accuracy * 100:.2f}%")
print(f"   ✓ AUC-ROC Score: {auc_score:.4f}")
print(f"\n   📋 Confusion Matrix:")
print(f"      [[TN={cm[0,0]}  FP={cm[0,1]}]")
print(f"       [FN={cm[1,0]}  TP={cm[1,1]}]]")
print(f"\n   📋 Classification Report:")
print(class_report)

# --- 8. Save Model ---
print("\n8️⃣  Saving model...")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

model_path = 'model/logistic_regression_model.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"   ✓ Model saved to: {model_path}")
print(f"   ✓ File size: {os.path.getsize(model_path) / 1024:.2f} KB")

# --- 9. Save Feature Names & Metadata ---
print("\n9️⃣  Saving metadata...")

# Get all feature names (including polynomial interactions)
poly_features = pipeline.named_steps['poly'].get_feature_names_out(feature_cols).tolist()

# Calculate ROC curve for visualization
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Get feature coefficients
logreg_model = pipeline.named_steps['logreg']
coefficients = logreg_model.coef_[0]

# Get confusion matrix
confusion_matrix_data = cm.tolist()

metadata = {
    'original_feature_columns': feature_cols,
    'polynomial_feature_names': poly_features,
    'threshold': threshold,
    'accuracy': accuracy,
    'auc_score': auc_score,
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'class_names': ['Cheap', 'Expensive'],
    'n_original_features': len(feature_cols),
    'n_polynomial_features': len(poly_features),
    'confusion_matrix': confusion_matrix_data,
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist(),
    'feature_coefficients': coefficients.tolist()
}

metadata_path = 'model/logistic_regression_metadata.pkl'

with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"   ✓ Metadata saved to: {metadata_path}")

# --- 10. Test Loading ---
print("\n🔟 Testing model loading...")

with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

# Test prediction
test_sample = X_test.iloc[0:1]
test_prediction = loaded_model.predict(test_sample)[0]
test_proba = loaded_model.predict_proba(test_sample)[0]

print(f"   ✓ Model loaded successfully!")
print(f"   ✓ Test prediction: {['Cheap', 'Expensive'][test_prediction]}")
print(f"   ✓ Confidence: {max(test_proba)*100:.2f}%")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nYou can now use this model in Streamlit:")
print(f"  with open('model/logistic_regression_model.pkl', 'rb') as f:")
print(f"      model = pickle.load(f)")
print("=" * 60)
