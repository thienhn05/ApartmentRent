"""
Training Script: Support Vector Machine (SVM) Model for Cheap vs Expensive Apartment Classification
Saves the trained model as a pickle file for use in Streamlit
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRAINING SUPPORT VECTOR MACHINE (SVM) MODEL")
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
    'bathrooms', 'bedrooms', 'square_feet', 'city_avg_price',
    'state', 'housing/rent/apartment',
    'housing/rent/condo', 'fee_yes'
]

print(f"   ✓ Features selected: {features}")
print(f"   ✓ Total features: {len(features)}")

# --- 4. Prepare Data ---
print("\n4️⃣  Splitting data (80/20)...")

X = df[features]
y = df['is_expensive']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   ✓ Training set: {len(X_train):,} samples")
print(f"   ✓ Test set: {len(X_test):,} samples")

# --- 5. Feature Engineering with Polynomial Features & Scaling (Match Module 3 notebook) ---
print("\n5️⃣  Applying Polynomial Features (Degree 2, Interaction Only) and Scaling...")

# Create polynomial features (degree 2, interaction only) - matches Module 3 notebook
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(f"   ✓ Original features: {len(features)}")
print(f"   ✓ After polynomial features: {X_train_poly.shape[1]}")

# Scale the features (crucial for SVM) - matches Module 3 notebook
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

print(f"   ✓ Features scaled using StandardScaler")

# --- 6. Train SVM Model (Match Module 3 notebook parameters) ---
print("\n6️⃣  Training Support Vector Machine Model...")
print(f"   Parameters: random_state=42, max_iter=10000, dual=False, C=1.0")

svm_model = LinearSVC(random_state=42, max_iter=10000, dual=False, C=1.0)
svm_model.fit(X_train_scaled, y_train)

print("   ✓ Model trained successfully")

# --- 7. Evaluate Model ---
print("\n7️⃣  Evaluating model...")

y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n   📊 Model Performance:")
print(f"   - Accuracy: {accuracy*100:.2f}%")

print(f"\n   🔢 Confusion Matrix:")
print(f"   - True Negatives: {cm[0, 0]:,}")
print(f"   - False Positives: {cm[0, 1]:,}")
print(f"   - False Negatives: {cm[1, 0]:,}")
print(f"   - True Positives: {cm[1, 1]:,}")

print(f"\n   📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Cheap', 'Expensive']))

# --- 8. Save Model and Metadata ---
print("\n8️⃣  Saving model...")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model and preprocessors
model_data = {
    'svm_model': svm_model,
    'poly': poly,
    'scaler': scaler
}

with open('model/svm_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("   ✓ Model saved to: model/svm_model.pkl")

# Save metadata
metadata = {
    'accuracy': accuracy,
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'original_feature_columns': features,
    'threshold': threshold,
    'class_names': ['Cheap', 'Expensive'],
    'confusion_matrix': cm.tolist(),
    'model_type': 'LinearSVC',
    'max_iter': 10000,
    'C': 1.0,
    'poly_degree': 2
}

with open('model/svm_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("   ✓ Metadata saved to: model/svm_metadata.pkl")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nYou can now use this model in Streamlit:")
print(f"  with open('model/svm_model.pkl', 'rb') as f:")
print(f"      model_data = pickle.load(f)")
print(f"      svm_model = model_data['svm_model']")
print(f"      poly = model_data['poly']")
print(f"      scaler = model_data['scaler']")
print("=" * 60)
