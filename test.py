import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="RentWise: Apartment Rent Predictor", layout="wide")
st.title("🏙️ Apartment Rental Market Dashboard")
st.markdown("A standalone predictive framework utilizing **Support Vector Regression**.")

# 2. Loading the Serialized Model [5]
@st.cache_resource
def load_model():
    try:
        with open("model/best_rent_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# 3. Sidebar for Interative "What-if" Scenarios [6]
st.sidebar.header("🔍 Apartment Specifications")

def user_input_features():
    sqft = st.sidebar.slider("Square Footage (sq_ft)", 100, 5000, 950)
    beds = st.sidebar.selectbox("Number of Bedrooms", [7-11])
    baths = st.sidebar.selectbox("Number of Bathrooms", [1.0, 1.5, 2.0, 2.5, 3.0])
    pets = st.sidebar.selectbox("Pets Allowed? (0=No, 1=Yes)", [7])
    state = st.sidebar.selectbox("State Code", [7-14]) # Encodings from training

    data = {
        'square_feet': sqft,
        'bedrooms': beds,
        'bathrooms': baths,
        'pets_allowed_encoded': pets,
        'state_encoded': state
    }
    # FIXED: Index= ensures proper single-row DataFrame creation
    return pd.DataFrame(data, index=)

input_df = user_input_features()

# 4. Overview KPIs [15]
st.subheader("📌 Market Overview KPIs")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Avg. National Rent", "$1,527 USD") # Based on dataset mean [16]
kpi2.metric("Median Size", "900 sq.ft")
kpi3.metric("Model R² Accuracy", "0.65")

# 5. Visual Insights (Exploratory Data Analysis) [8]
st.subheader("📊 Rental Market Trends")
state_data = pd.DataFrame({
    'State': ["TX", "CA", "VA", "NC", "CO", "FL", "NY", "WA"],
    'Avg_Rent': 
})
fig = px.bar(state_data, x='State', y='Avg_Rent', color='Avg_Rent', title="Average Rent by State")
st.plotly_chart(fig, use_container_width=True)

# 6. Prediction Logic (Deployment) [5]
st.subheader("🔮 Predictive Analytics Engine")
if st.button("Calculate Estimated Rent"):
    if model:
        prediction = model.predict(input_df)
        st.success(f"### The predicted monthly rent is: **${prediction:,.2f} USD**")
    else:
        st.error("⚠️ Error: Model file 'best_rent_model.pkl' not found. Please ensure it is in the /model folder.")

# 7. Model Technicalities [17]
with st.expander("🛠️ View Model Performance Details"):
    st.write("Results from 10-fold Cross-Validation:")
    st.write("- **RMSE:** 0.0189")
    st.write("- **MAE:** 0.0924")
    st.write("- **Baseline Comparison:** Outperforms Decision Tree (R²=0.50)")
