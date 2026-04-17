"""
Training Script: K-Nearest Neighbors (KNN) Model for Cheap vs Expensive Apartment Classification
Saves the trained model as a pickle file for use in Streamlit
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING K-NEAREST NEIGHBORS (KNN) MODEL")
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
    'latitude', 'longitude',
    'housing/rent/apartment',
    'housing/rent/condo', 'fee_yes'
]

print(f"   ✓ Features selected: {features}")
print(f"   ✓ Total features: {len(features)}")

# --- 4. Prepare Data ---
print("\n4️⃣  Splitting data (80/20)...")

X = df[features]
y = df['is_expensive']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

print(f"   ✓ Training set: {len(X_train):,} samples")
print(f"   ✓ Test set: {len(X_test):,} samples")

# --- 5. Train KNN Model ---
print("\n5️⃣  Training K-Nearest Neighbors Model...")
print(f"   Parameters: n_neighbors=5")

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

print(f"   ✓ Model trained successfully")

# --- 6. Make Predictions ---
print("\n6️⃣  Making predictions on test set...")

y_pred = knn_model.predict(X_test)
y_pred_proba = knn_model.predict_proba(X_test)

print(f"   ✓ Predictions generated")

# --- 7. Evaluate Model ---
print("\n7️⃣  Evaluating model performance...")

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n   📊 Model Metrics:")
print(f"   - Accuracy:  {accuracy*100:.2f}%")
print(f"   - Precision: {precision*100:.2f}%")
print(f"   - Recall:    {recall*100:.2f}%")
print(f"   - F1-Score:  {f1*100:.2f}%")

print(f"\n   Confusion Matrix:")
print(f"   {cm}")

print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Cheap', 'Expensive']))

# --- 8. Create Model Directory if not exists ---
print("\n8️⃣  Saving model and metadata...")

if not os.path.exists('model'):
    os.makedirs('model')
    print(f"   ✓ Created 'model' directory")

# --- 9. Save Model ---
with open('model/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
print(f"   ✓ Model saved to 'model/knn_model.pkl'")

# --- 10. Save Metadata ---
metadata = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': cm.tolist(),
    'original_feature_columns': features,
    'n_neighbors': 5,
    'metric': 'euclidean',
    'threshold': threshold
}

with open('model/knn_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✓ Metadata saved to 'model/knn_metadata.pkl'")

print("\n" + "=" * 60)
print("✅ KNN MODEL TRAINING COMPLETE!")
print("=" * 60)
