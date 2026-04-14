"""
Training Script: Decision Tree Model for Cheap vs Expensive Apartment Classification
Saves the trained model as a pickle file for use in Streamlit
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING DECISION TREE MODEL")
print("=" * 60)

# --- 1. Load Your Real Dataset ---
print("\n1️⃣  Loading dataset...")

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

if 'is_expensive' not in df.columns:
    print("   ⚠️  'is_expensive' column not found. Creating from price threshold...")
    threshold = df['price'].quantile(0.80)
    df['is_expensive'] = (df['price'] > threshold).astype(int)
    print(f"   ✓ Threshold set to: ${threshold:,.2f}")
else:
    threshold = df[df['is_expensive'] == 1]['price'].min()
    print(f"   ✓ Using existing 'is_expensive' column")

print(f"   ✓ Class distribution: {df['is_expensive'].value_counts().to_dict()}")

# --- 3. Feature Engineering ---
print("\n3️⃣  Preparing features...")

# Create city average price
city_map = df.groupby('cityname')['price'].mean()
df['city_avg_price'] = df['cityname'].map(city_map)

features = [
    'bathrooms', 'bedrooms', 'square_feet',
    'city_avg_price', 'state',
    'housing/rent/apartment',
    'housing/rent/condo',
    'fee_yes'
]

print(f"   ✓ Features selected: {features}")
print(f"   ✓ Total features: {len(features)}")

# --- 4. Prepare Data ---
print("\n4️⃣  Splitting data...")

X = df[features]
y = df['is_expensive']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   ✓ Training set: {len(X_train):,} samples")
print(f"   ✓ Test set: {len(X_test):,} samples")

# --- 5. Train Decision Tree Model ---
print("\n5️⃣  Training Decision Tree Model...")

dt_model = DecisionTreeClassifier(
    max_depth=8,
    random_state=42,
    min_samples_split=10,
    min_samples_leaf=5
)

dt_model.fit(X_train, y_train)

print("   ✓ Model trained successfully")

# --- 6. Evaluate Model ---
print("\n6️⃣  Evaluating model...")

y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
cm = confusion_matrix(y_test, y_pred)

print(f"\n   📊 Model Performance:")
print(f"   - Accuracy: {accuracy*100:.2f}%")
print(f"   - AUC-ROC: {auc_score:.4f}")

print(f"\n   🔢 Confusion Matrix:")
print(f"   - True Negatives: {cm[0, 0]:,}")
print(f"   - False Positives: {cm[0, 1]:,}")
print(f"   - False Negatives: {cm[1, 0]:,}")
print(f"   - True Positives: {cm[1, 1]:,}")

print(f"\n   📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Cheap', 'Expensive']))

# --- 7. Save Model and Metadata ---
print("\n7️⃣  Saving model...")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model
with open('model/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
print("   ✓ Model saved to: model/decision_tree_model.pkl")

# Save metadata
metadata = {
    'accuracy': accuracy,
    'auc_score': auc_score,
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'original_feature_columns': features,
    'threshold': threshold,
    'class_names': ['Cheap', 'Expensive'],
    'confusion_matrix': cm.tolist(),
    'max_depth': 8
}

with open('model/decision_tree_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("   ✓ Metadata saved to: model/decision_tree_metadata.pkl")

# --- 8. Feature Importance ---
print("\n8️⃣  Feature Importance:")
feature_importance = pd.Series(dt_model.feature_importances_, index=features)
feature_importance = feature_importance.sort_values(ascending=False)
print(feature_importance)

print("\n" + "=" * 60)
print("✅ DECISION TREE TRAINING COMPLETE!")
print("=" * 60)
