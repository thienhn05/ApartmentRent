import streamlit as st
import pandas as pd
import pickle
from price_fairness import classify_price_fairness

st.set_page_config(page_title="Price Fairness Module", layout="wide")
st.title("💡 Price Fairness Analysis Module")
st.markdown("This module uses **Random Forest Regression** to estimate fair apartment rent and classify listings as underpriced, fairly priced, or overpriced.")

@st.cache_resource
def load_model():
    try:
        with open("model/RandomForest_rent_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

st.sidebar.header("Apartment Specifications")

sqft = st.sidebar.slider("Square Footage", 100, 5000, 950)
beds = st.sidebar.selectbox("Bedrooms", list(range(0, 11)))
baths = st.sidebar.selectbox("Bathrooms", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
pets = st.sidebar.selectbox("Pets Allowed? (0=No, 1=Yes)", [0, 1])
state = st.sidebar.selectbox("State Code", list(range(1, 15)))
actual_rent = st.sidebar.number_input(
    "Actual Listing Rent (USD)",
    min_value=100.0,
    max_value=20000.0,
    value=1500.0,
    step=50.0
)

input_df = pd.DataFrame({
    'square_feet': [sqft],
    'bedrooms': [beds],
    'bathrooms': [baths],
    'pets_allowed_encoded': [pets],
    'state_encoded': [state]
})

st.write("App loaded successfully.")

if st.button("Analyze Price Fairness"):
    if model is not None:
        try:
            predicted_rent = float(model.predict(input_df)[0])
            result = classify_price_fairness(actual_rent, predicted_rent)

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Fair Rent", f"${predicted_rent:,.2f}")
            col2.metric("Difference", f"${result['difference']:,.2f}")
            col3.metric("Percentage Difference", f"{result['percentage_difference']:.2f}%")

            st.markdown(f"### Status: **{result['status']}**")

            if result["status"] == "Overpriced":
                st.warning("This apartment appears to be listed above the predicted fair rent.")
            elif result["status"] == "Underpriced":
                st.success("This apartment appears to be listed below the predicted fair rent.")
            else:
                st.info("This apartment appears to be fairly priced.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model file not found. Please train and save RandomForest_rent_model.pkl first.")
