import pandas as pd
import pickle
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load the cleaned Apartment Rent dataset [1]
df = pd.read_csv('dataset/clean_apartment_data.csv')

# 2. Feature Selection (Based on previous analysis) [2]
features = ['square_feet', 'bedrooms', 'bathrooms', 'pets_allowed_encoded', 'state_encoded']
X = df[features]
y = df['price']

# 3. Model Training (Support Vector Regression - Best performer) [3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVR(kernel='linear', C=0.15, epsilon=0.06)
model.fit(X_train, y_train)

# 4. Model Serialization (Saving as .pkl) [4]
with open('model/best_rent_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model successfully serialized and saved to model/best_rent_model.pkl")
