"""
Training Script: Random Forest Model for Cheap vs Expensive Apartment Classification
Saves the trained model as a pickle file for use in Streamlit
Based on the Random Forest notebook approach
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING RANDOM FOREST MODEL")
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

# --- 3. Feature Selection ---
print("\n3️⃣  Selecting features...")

# Use the same features as Decision Tree for consistency
features = [
    'bathrooms', 'bedrooms', 'square_feet',
    'city_avg_price', 'state',
    'housing/rent/apartment',
    'housing/rent/condo',
    'fee_yes'
]

# Create city average price if needed
if 'city_avg_price' not in df.columns:
    city_map = df.groupby('cityname')['price'].mean()
    df['city_avg_price'] = df['cityname'].map(city_map)

print(f"   ✓ Features selected: {features}")
print(f"   ✓ Total features: {len(features)}")

# --- 4. Prepare Data ---
print("\n4️⃣  Splitting data (80/20)...")

X = df[features]
y = df['is_expensive']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training set: {len(X_train):,} samples")
print(f"   ✓ Test set: {len(X_test):,} samples")

# --- 5. Train Random Forest Model ---
print("\n5️⃣  Training Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("   ✓ Model trained successfully")

# --- 6. Evaluate Model ---
print("\n6️⃣  Evaluating model...")

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n   📊 Model Performance:")
print(f"   - Accuracy: {accuracy*100:.2f}%")
print(f"   - Precision: {precision*100:.2f}%")
print(f"   - Recall: {recall*100:.2f}%")
print(f"   - F1-Score: {f1*100:.2f}%")

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
with open('model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("   ✓ Model saved to: model/random_forest_model.pkl")

# Save metadata
metadata = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'original_feature_columns': features,
    'threshold': threshold,
    'class_names': ['Cheap', 'Expensive'],
    'confusion_matrix': cm.tolist(),
    'n_estimators': 100,
    'max_depth': 15
}

with open('model/random_forest_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("   ✓ Metadata saved to: model/random_forest_metadata.pkl")

# --- 8. Feature Importance ---
print("\n8️⃣  Feature Importance:")
feature_importance = pd.Series(rf_model.feature_importances_, index=features)
feature_importance = feature_importance.sort_values(ascending=False)
print(feature_importance)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nYou can now use this model in Streamlit:")
print(f"  with open('model/random_forest_model.pkl', 'rb') as f:")
print(f"      model = pickle.load(f)")
print("=" * 60)
