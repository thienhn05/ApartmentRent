import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ===============================================
# CONSTANTS & MAPPINGS
# ===============================================
STATE_CODE_TO_NAME = {
    1: "Alabama", 2: "Alaska", 3: "Arizona", 4: "Arkansas", 5: "California",
    6: "Colorado", 7: "Connecticut", 8: "Delaware", 9: "Florida", 10: "Georgia",
    11: "Hawaii", 12: "Idaho", 13: "Illinois", 14: "Indiana", 15: "Iowa",
    16: "Kansas", 17: "Kentucky", 18: "Louisiana", 19: "Maine", 20: "Maryland",
    21: "Massachusetts", 22: "Michigan", 23: "Minnesota", 24: "Mississippi", 25: "Missouri",
    26: "Montana", 27: "Nebraska", 28: "Nevada", 29: "New Hampshire", 30: "New Jersey",
    31: "New Mexico", 32: "New York", 33: "North Carolina", 34: "North Dakota", 35: "Ohio",
    36: "Oklahoma", 37: "Oregon", 38: "Pennsylvania", 39: "Rhode Island", 40: "South Carolina",
    41: "South Dakota", 42: "Tennessee", 43: "Texas", 44: "Utah", 45: "Vermont",
    46: "Virginia", 47: "Washington", 48: "West Virginia", 49: "Wisconsin", 50: "Wyoming"
}
STATE_NAME_TO_CODE = {v: k for k, v in STATE_CODE_TO_NAME.items()}

# ===============================================
# PAGE CONFIGURATION
# ===============================================
st.set_page_config(
    page_title="Apartment Rent", 
    layout="wide",
    page_icon="🏙️"
)

# ===============================================
# LOAD REAL DATASET
# ===============================================
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('clean_apartment_data.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found!")
        return None

df_main = load_dataset()

# ===============================================
# LOAD DECISION TREE MODEL & METADATA
# ===============================================
@st.cache_resource
def load_dt_model():
    try:
        with open('model/decision_tree_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_dt_metadata():
    try:
        with open('model/decision_tree_metadata.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

dt_model = load_dt_model()
dt_metadata = load_dt_metadata()

# ===============================================
# LOAD SVM MODEL & METADATA
# ===============================================
@st.cache_resource
def load_svm_model():
    try:
        with open('model/svm_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            return model_data['svm_model'], model_data['poly'], model_data['scaler']
    except FileNotFoundError:
        return None, None, None

@st.cache_resource
def load_svm_metadata():
    try:
        with open('model/svm_metadata.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

svm_model, svm_poly, svm_scaler = load_svm_model()
svm_metadata = load_svm_metadata()

# ===============================================
# LOAD RANDOM FOREST MODEL & METADATA
# ===============================================
@st.cache_resource
def load_rf_model():
    try:
        with open('model/random_forest_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_rf_metadata():
    try:
        with open('model/random_forest_metadata.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

rf_model = load_rf_model()
rf_metadata = load_rf_metadata()

# ===============================================
# LOAD KNN MODEL & METADATA
# ===============================================
@st.cache_resource
def load_knn_model():
    try:
        with open('model/knn_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_knn_metadata():
    try:
        with open('model/knn_metadata.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

knn_model = load_knn_model()
knn_metadata = load_knn_metadata()


# ===============================================
# SIDEBAR NAVIGATION
# ===============================================
with st.sidebar:
    st.title("🏙️ Apartment Rent")
    st.markdown("---")
    
    # Custom CSS for the radio buttons to look like a menu
    st.markdown("""
        <style>
        .stRadio [data-testid="stMarkdownContainer"] p {
            font-size: 18px;
            font-weight: 500;
        }
        div[data-testid="stSidebarNav"] {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Select:",
        ["🏠 Project Overview", "⚖️ Model Comparison", "📊 Model Predictions"],
        index=0,
        help="Switch between overview, model comparison, and analytics"
    )

# ===============================================
# PAGE 1: PROJECT OVERVIEW
# ===============================================
if "Project Overview" in page:
    st.title("🏙️ Apartment Rent Prices")
    st.caption("A Machine Learning approach to predicting apartment rent price categories.")
    
    st.markdown("---")
    
    # Hero/Problem Statement Section
    st.markdown("""
        <div style="background: linear-gradient(to right, #f8f9fa, #e9ecef); padding: 20px; border-radius: 10px; border-left: 5px solid #0068c9; margin-bottom: 25px;">
            <h3 style="margin-top: 0; color: #1e1e1e;">📖 Problem Statement</h3>
            <p style="font-size: 16px; color: #424242; margin-bottom: 0;">
                There is a significant <b>lack of transparency in rental pricing</b>. Renters often struggle to determine if an apartment is fairly priced, while landlords need help pricing realistically to stay competitive in the market.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Objectives")
        st.markdown("""
        - 🥇 **Primary Objective:** Classify apartments as **Cheap or Expensive** based on their physical attributes and location.
        - 🥈 **Secondary Objective:** Identify the **key influencing features** (e.g., size vs. location) that determine apartment pricing.
        - 📈 **Analytical Goal:** Compare multiple classification models to find the most accurate and robust approach.
        """)
        
    with col2:
        st.subheader("📊 Dataset Overview")
        st.markdown("""
        Our analysis relies on a cleaned dataset of apartment listings across the US. Key features include:
        - 🛏️ **Property Size:** Bedrooms, Bathrooms, Square footage
        - 🏢 **Property Type:** Apartment, Condo, House
        - 📍 **Location:** State, City pricing averages, Latitude, Longitude
        - 💰 **Additional Costs:** Monthly fees included
        """)
            
    st.markdown("---")
    
    st.subheader("🤖 Machine Learning Models")
    st.write("We evaluated the following four classification models (replacing Logistic Regression with Random Forest for higher ensemble accuracy):")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.info("**Decision Tree**")
    with m_col2:
        st.info("**Random Forest**")
    with m_col3:
        st.info("**K-Nearest Neighbors (KNN)**")
    with m_col4:
        st.info("**Support Vector Machine (SVM)**")

# ===============================================
# PAGE 2: MODEL COMPARISON
# ===============================================
elif "Model Comparison" in page:
    st.title("⚖️ Compare Machine Learning Models")
    st.caption("Side-by-side performance evaluation of all trained models.")
    st.markdown("---")
    
    st.write("To determine the best performing algorithm for predicting apartment prices, we compared the four trained models across key evaluation metrics: Accuracy, Precision, Recall, and F1-Score.")
    
    # Helper to safely extract metrics for comparison
    def get_metrics_dict(metadata, name):
        if not metadata or 'confusion_matrix' not in metadata:
            return {"Model": name, "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0}
            
        cm = np.array(metadata['confusion_matrix'])
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        
        acc = metadata.get('accuracy', (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0)
        prec = metadata.get('precision', tp / (tp + fp) if (tp + fp) > 0 else 0)
        rec = metadata.get('recall', tp / (tp + fn) if (tp + fn) > 0 else 0)
        f1 = metadata.get('f1_score', 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0)
        
        return {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        }
        
    metrics_data = [
        get_metrics_dict(dt_metadata, "Decision Tree"),
        get_metrics_dict(rf_metadata, "Random Forest"),
        get_metrics_dict(knn_metadata, "K-Nearest Neighbors (KNN)"),
        get_metrics_dict(svm_metadata, "Support Vector Machine (SVM)")
    ]
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Sort models by accuracy
    df_metrics_sorted = df_metrics.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    # --- Display Table ---
    st.subheader("📋 Performance Metrics Table")
    
    # Create formatted copy for display
    def format_pct(x): return f"{x*100:.2f}%"
    df_display = df_metrics_sorted.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        df_display[col] = df_display[col].apply(format_pct)
        
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    st.caption("*Note: Higher percentages represent better model performance.*")
    
    st.markdown("---")
    
    # --- Display Visualization ---
    st.subheader("📊 Accuracy Comparison Chart")
    
    fig = px.bar(
        df_metrics_sorted, 
        x="Model", 
        y="Accuracy",
        color="Model",
        text=df_metrics_sorted["Accuracy"].apply(format_pct),
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        yaxis_title="Accuracy", 
        xaxis_title="",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=500,
        showlegend=False
    )
    fig.update_traces(textposition='outside', textfont_size=14)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # --- Compare Confusion Matrices ---
    st.subheader("🔢 Confusion Matrices Comparison")
    st.write("Compare the True Positives, True Negatives, False Positives, and False Negatives across all four models to understand where each model excels or struggles.")
    
    def plot_cm_for_comparison(metadata, name, colorscale):
        if not metadata or 'confusion_matrix' not in metadata:
            return None
        cm = np.array(metadata['confusion_matrix'])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Pred Cheap', 'Pred Exp'],
            y=['Actual Cheap', 'Actual Exp'],
            text=cm,
            texttemplate='%{text}',
            colorscale=colorscale,
            showscale=False
        ))
        fig_cm.update_layout(
            title=dict(text=name, font=dict(size=18)),
            height=300, 
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="", 
            yaxis_title=""
        )
        return fig_cm
        
    cm_col1, cm_col2 = st.columns(2)
    
    with cm_col1:
        if dt_metadata:
            st.plotly_chart(plot_cm_for_comparison(dt_metadata, "Decision Tree", "Greens"), use_container_width=True)
        if knn_metadata:
            st.plotly_chart(plot_cm_for_comparison(knn_metadata, "K-Nearest Neighbors", "Purples"), use_container_width=True)
            
    with cm_col2:
        if rf_metadata:
            st.plotly_chart(plot_cm_for_comparison(rf_metadata, "Random Forest", "Oranges"), use_container_width=True)
        if svm_metadata:
            st.plotly_chart(plot_cm_for_comparison(svm_metadata, "Support Vector Machine", "Blues"), use_container_width=True)

    st.markdown("---")
    
    # --- Display Winner ---
    best_model = df_metrics_sorted.iloc[0]
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                    border-left: 6px solid #28a745; padding: 20px; border-radius: 8px; margin: 20px 0;">
            <div style="display: flex; align-items: flex-start; gap: 20px;">
                <span style="font-size: 36px; padding-top: 5px; color: #28a745;"></span>
                <div>
                    <h3 style="margin: 0 0 10px 0; color: #155724;">Best Performing Model: {best_model['Model']}</h3>
                    <p style="margin: 0; color: #155724; font-size: 16px; line-height: 1.5;">
                        Based on the comparative analysis, the <b>{best_model['Model']}</b> performs the best overall 
                        with an outstanding accuracy of <b>{best_model['Accuracy']*100:.2f}%</b> and an F1-Score 
                        of <b>{best_model['F1-Score']*100:.2f}%</b>. This model is highly recommended for predicting apartment price categories in deployment.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================================
# PAGE 3: MODEL INSIGHTS & PREDICTIONS
# ===============================================
elif "Model Predictions" in page:
    st.title("📊 Model Insights & Predictions")
    st.markdown("---")
    
    # Select Model
    selected_model = st.selectbox(
        "Select Machine Learning Model:", 
        ["Decision Tree", "Support Vector Machine (SVM)", "Random Forest", "K-Nearest Neighbors (KNN)"]
    )
    
    st.markdown("---")
    
    if selected_model == "Decision Tree":
        st.header("Decision Tree")
        
        tab1, tab2 = st.tabs(["🔮 Make a Prediction", "📈 Model Insights"])
        
        # ---------------------------------------------------------
        # TAB 1: PREDICTION
        # ---------------------------------------------------------
        with tab1:
            st.subheader("Interactive Prediction")
            st.write("Adjust the apartment features below to predict whether it would be classified as Cheap or Expensive.")
            
            with st.form("dt_prediction_form"):
                # Row 1: Property Type & State
                col1, col2 = st.columns(2)
                with col1:
                    property_type = st.selectbox("Property Type", ["Apartment", "Condo", "House"])
                with col2:
                    state = st.selectbox("State", options=sorted(list(STATE_NAME_TO_CODE.keys())), index=sorted(list(STATE_NAME_TO_CODE.keys())).index("Texas"))
                
                # Row 2: Square Feet & Monthly Fee
                col1, col2 = st.columns(2)
                with col1:
                    sqft = st.number_input("Square Feet", min_value=200, max_value=5000, value=1000, step=50)
                with col2:
                    fee = st.selectbox("Monthly Fee", ["No", "Yes"])
                
                # Row 3: Bedrooms & Bathrooms
                col1, col2 = st.columns(2)
                with col1:
                    beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
                with col2:
                    baths = st.number_input("Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5)
                
                # Row 4: City Average Rent
                city_avg = st.number_input("City Avg. Rent ($)", min_value=500.0, max_value=5000.0, value=1500.0, step=50.0)
                
                submit = st.form_submit_button("Predict Rent Price Category", use_container_width=True)
                
            if submit:
                if dt_model:
                    apt_val = 1 if property_type == "Apartment" else 0
                    condo_val = 1 if property_type == "Condo" else 0
                    state_code = STATE_NAME_TO_CODE[state]
                    fee_val = 1 if fee == "Yes" else 0
                    
                    input_data = pd.DataFrame({
                        'bathrooms': [float(baths)],
                        'bedrooms': [int(beds)],
                        'square_feet': [float(sqft)],
                        'city_avg_price': [float(city_avg)],
                        'state': [state_code],
                        'housing/rent/apartment': [apt_val],
                        'housing/rent/condo': [condo_val],
                        'fee_yes': [fee_val]
                    })
                    
                    try:
                        prediction = dt_model.predict(input_data)[0]
                        probabilities = dt_model.predict_proba(input_data)[0]
                        prob_expensive = probabilities[1]
                        prob_cheap = probabilities[0]
                        
                        # Modern result display
                        if prediction == 0:
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                                            border-left: 6px solid #28a745; padding: 20px; border-radius: 8px; margin: 20px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center; gap: 15px;">
                                            <span style="font-size: 32px; color: #28a745;">🟢</span>
                                            <div>
                                                <p style="margin: 0; color: #155724; font-weight: bold; font-size: 24px;">CHEAP</p>
                                                <p style="margin: 0; color: #155724; font-size: 14px;">Predicted Price Category</p>
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 30px; text-align: center;">
                                            <div>
                                                <p style="margin: 0; color: #155724; font-size: 28px; font-weight: bold;">{prob_cheap*100:.1f}%</p>
                                                <p style="margin: 0; color: #155724; font-size: 12px;">Cheap Confidence</p>
                                            </div>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-size: 28px; font-weight: bold;">{prob_expensive*100:.1f}%</p>
                                                <p style="margin: 0; color: #721c24; font-size: 12px;">Expensive Confidence</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
                                            border-left: 6px solid #dc3545; padding: 20px; border-radius: 8px; margin: 20px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center; gap: 15px;">
                                            <span style="font-size: 32px; color: #dc3545;">🔴</span>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-weight: bold; font-size: 24px;">EXPENSIVE</p>
                                                <p style="margin: 0; color: #721c24; font-size: 14px;">Predicted Price Category</p>
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 30px; text-align: center;">
                                            <div>
                                                <p style="margin: 0; color: #155724; font-size: 28px; font-weight: bold;">{prob_cheap*100:.1f}%</p>
                                                <p style="margin: 0; color: #155724; font-size: 12px;">Cheap Confidence</p>
                                            </div>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-size: 28px; font-weight: bold;">{prob_expensive*100:.1f}%</p>
                                                <p style="margin: 0; color: #721c24; font-size: 12px;">Expensive Confidence</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
                else:
                    st.error("Decision Tree model not loaded. Please train the model and generate the .pkl files.")

        # ---------------------------------------------------------
        # TAB 2: MODEL INSIGHTS
        # ---------------------------------------------------------
        with tab2:
            st.write("This section breaks down the performance and behavior of the **Decision Tree** model.")
            
            if dt_model and dt_metadata:
                # Performance Overview Table
                st.subheader("🏆 Model Performance")
                
                cm = np.array(dt_metadata['confusion_matrix'])
                tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{precision*100:.2f}%")
                with col3:
                    st.metric("Recall", f"{recall*100:.2f}%")
                with col4:
                    st.metric("F1-Score", f"{f1_score*100:.2f}%")
                    
                st.markdown("---")
                
                # Confusion Matrix
                colLeft, colRight = st.columns([1, 2])
                with colLeft:
                    st.subheader("🔢 Confusion Matrix")
                    st.write(f"**True Negatives (TN): {tn:,}**")
                    
                    st.write(f"**False Positives (FP): {fp:,}**")
                    
                    st.write(f"**False Negatives (FN): {fn:,}**")
                    
                    st.write(f"**True Positives (TP): {tp:,}**")
                    
                with colRight:
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Cheap', 'Predicted Expensive'],
                        y=['Actually Cheap', 'Actually Expensive'],
                        text=cm,
                        texttemplate='%{text}',
                        colorscale='Greens',
                        showscale=False
                    ))
                    fig_cm.update_layout(height=500, xaxis_title="Predicted Label", yaxis_title="True Label")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                st.markdown("---")
                
                # Feature Importance
                st.subheader("🔑 Feature Importance")
                st.write("Which features influenced the price classification the most?")
                
                feature_importance = pd.Series(
                    dt_model.feature_importances_,
                    index=dt_metadata['original_feature_columns']
                ).sort_values(ascending=True)
                
                fig_importance = px.bar(
                    x=feature_importance.values,
                    y=feature_importance.index,
                    orientation='h',
                    title="Importance of Apartment Features",
                    labels={'x': 'Importance Score', 'y': 'Feature'},
                    color=feature_importance.values,
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.error("Decision Tree model or metadata not loaded. Please ensure the model files exist.")
    
    elif selected_model == "Support Vector Machine (SVM)":
        st.header("Support Vector Machine (SVM)")
        
        tab1, tab2 = st.tabs(["🔮 Make a Prediction", "📈 Model Insights"])
        
        # ---------------------------------------------------------
        # TAB 1: PREDICTION
        # ---------------------------------------------------------
        with tab1:
            st.subheader("Interactive Prediction")
            st.write("Adjust the apartment features below to predict whether it would be classified as Cheap or Expensive.")
            
            with st.form("svm_prediction_form"):
                # Row 1: Property Type & State
                col1, col2 = st.columns(2)
                with col1:
                    property_type = st.selectbox("Property Type", ["Apartment", "Condo", "House"], key="svm_property")
                with col2:
                    state = st.selectbox("State", options=sorted(list(STATE_NAME_TO_CODE.keys())), index=sorted(list(STATE_NAME_TO_CODE.keys())).index("Texas"), key="svm_state")
                
                # Row 2: Square Feet & Monthly Fee
                col1, col2 = st.columns(2)
                with col1:
                    sqft = st.number_input("Square Feet", min_value=200, max_value=5000, value=1000, step=50, key="svm_sqft")
                with col2:
                    fee = st.selectbox("Monthly Fee", ["No", "Yes"], key="svm_fee")
                
                # Row 3: Bedrooms & Bathrooms
                col1, col2 = st.columns(2)
                with col1:
                    beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1, key="svm_beds")
                with col2:
                    baths = st.number_input("Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5, key="svm_baths")
                
                # Row 4: City Average Rent
                city_avg = st.number_input("City Avg. Rent ($)", min_value=500.0, max_value=5000.0, value=1500.0, step=50.0, key="svm_city_avg")
                
                submit = st.form_submit_button("Predict Rent Price Category", use_container_width=True)
                
            if submit:
                if svm_model and svm_poly and svm_scaler:
                    apt_val = 1 if property_type == "Apartment" else 0
                    condo_val = 1 if property_type == "Condo" else 0
                    state_code = STATE_NAME_TO_CODE[state]
                    fee_val = 1 if fee == "Yes" else 0
                    
                    input_data = pd.DataFrame({
                        'bathrooms': [float(baths)],
                        'bedrooms': [int(beds)],
                        'square_feet': [float(sqft)],
                        'city_avg_price': [float(city_avg)],
                        'state': [state_code],
                        'housing/rent/apartment': [apt_val],
                        'housing/rent/condo': [condo_val],
                        'fee_yes': [fee_val]
                    })
                    
                    try:
                        # Transform features: polynomial then scale
                        input_poly = svm_poly.transform(input_data)
                        input_scaled = svm_scaler.transform(input_poly)
                        
                        prediction = svm_model.predict(input_scaled)[0]
                        
                        # Modern result display matching Decision Tree & Random Forest format
                        if prediction == 0:
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                                            border-left: 6px solid #28a745; padding: 20px; border-radius: 8px; margin: 20px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center; gap: 15px;">
                                            <span style="font-size: 32px; color: #28a745;">🟢</span>
                                            <div>
                                                <p style="margin: 0; color: #155724; font-weight: bold; font-size: 24px;">CHEAP</p>
                                                <p style="margin: 0; color: #155724; font-size: 14px;">Predicted Price Category</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
                                            border-left: 6px solid #dc3545; padding: 20px; border-radius: 8px; margin: 20px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center; gap: 15px;">
                                            <span style="font-size: 32px; color: #dc3545;">🔴</span>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-weight: bold; font-size: 24px;">EXPENSIVE</p>
                                                <p style="margin: 0; color: #721c24; font-size: 14px;">Predicted Price Category</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
                else:
                    st.error("SVM model not loaded. Please run train_svm.py to train and generate the model files.")

        # ---------------------------------------------------------
        # TAB 2: MODEL INSIGHTS
        # ---------------------------------------------------------
        with tab2:
            st.write("This section breaks down the performance and behavior of the **Support Vector Machine (SVM)** model.")
            
            if svm_model and svm_metadata:
                # Performance Overview Table
                st.subheader("🏆 Model Performance")
                
                cm = np.array(svm_metadata['confusion_matrix'])
                tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{precision*100:.2f}%")
                with col3:
                    st.metric("Recall", f"{recall*100:.2f}%")
                with col4:
                    st.metric("F1-Score", f"{f1_score*100:.2f}%")
                    
                st.markdown("---")
                
                # Confusion Matrix
                colLeft, colRight = st.columns([1, 2])
                with colLeft:
                    st.subheader("🔢 Confusion Matrix")
                    st.write(f"**True Negatives (TN): {tn:,}**")
                    
                    st.write(f"**False Positives (FP): {fp:,}**")
                    
                    st.write(f"**False Negatives (FN): {fn:,}**")
                    
                    st.write(f"**True Positives (TP): {tp:,}**")
                    
                with colRight:
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Cheap', 'Predicted Expensive'],
                        y=['Actually Cheap', 'Actually Expensive'],
                        text=cm,
                        texttemplate='%{text}',
                        colorscale='Blues',
                        showscale=False
                    ))
                    fig_cm.update_layout(height=500, xaxis_title="Predicted Label", yaxis_title="True Label")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
            else:
                st.error("SVM model or metadata not loaded. Please ensure the model files exist.")
    
    elif selected_model == "Random Forest":
        st.header("Random Forest")
        
        tab1, tab2 = st.tabs(["🔮 Make a Prediction", "📈 Model Insights"])
        
        # ---------------------------------------------------------
        # TAB 1: PREDICTION
        # ---------------------------------------------------------
        with tab1:
            st.subheader("Interactive Prediction")
            st.write("Adjust the apartment features below to predict whether it would be classified as Cheap or Expensive.")
            
            with st.form("rf_prediction_form"):
                # Row 1: Property Type & State
                col1, col2 = st.columns(2)
                with col1:
                    property_type = st.selectbox("Property Type", ["Apartment", "Condo", "House"], key="rf_property")
                with col2:
                    state = st.selectbox("State", options=sorted(list(STATE_NAME_TO_CODE.keys())), index=sorted(list(STATE_NAME_TO_CODE.keys())).index("Texas"), key="rf_state")
                
                # Row 2: Square Feet & Monthly Fee
                col1, col2 = st.columns(2)
                with col1:
                    sqft = st.number_input("Square Feet", min_value=200, max_value=5000, value=1000, step=50, key="rf_sqft")
                with col2:
                    fee = st.selectbox("Monthly Fee", ["No", "Yes"], key="rf_fee")
                
                # Row 3: Bedrooms & Bathrooms
                col1, col2 = st.columns(2)
                with col1:
                    beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1, key="rf_beds")
                with col2:
                    baths = st.number_input("Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5, key="rf_baths")
                
                # Row 4: City Average Rent
                city_avg = st.number_input("City Avg. Rent ($)", min_value=500.0, max_value=5000.0, value=1500.0, step=50.0, key="rf_city_avg")
                
                submit = st.form_submit_button("Predict Rent Price Category", use_container_width=True)
                
            if submit:
                if rf_model:
                    apt_val = 1 if property_type == "Apartment" else 0
                    condo_val = 1 if property_type == "Condo" else 0
                    state_code = STATE_NAME_TO_CODE[state]
                    fee_val = 1 if fee == "Yes" else 0
                    
                    input_data = pd.DataFrame({
                        'bathrooms': [float(baths)],
                        'bedrooms': [int(beds)],
                        'square_feet': [float(sqft)],
                        'city_avg_price': [float(city_avg)],
                        'state': [state_code],
                        'housing/rent/apartment': [apt_val],
                        'housing/rent/condo': [condo_val],
                        'fee_yes': [fee_val]
                    })
                    
                    try:
                        prediction = rf_model.predict(input_data)[0]
                        probabilities = rf_model.predict_proba(input_data)[0]
                        prob_expensive = probabilities[1]
                        prob_cheap = probabilities[0]
                        
                        # Modern result display
                        if prediction == 0:
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                                            border-left: 6px solid #28a745; padding: 20px; border-radius: 8px; margin: 20px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center; gap: 15px;">
                                            <span style="font-size: 32px; color: #28a745;">🟢</span>
                                            <div>
                                                <p style="margin: 0; color: #155724; font-weight: bold; font-size: 24px;">CHEAP</p>
                                                <p style="margin: 0; color: #155724; font-size: 14px;">Predicted Price Category</p>
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 30px; text-align: center;">
                                            <div>
                                                <p style="margin: 0; color: #155724; font-size: 28px; font-weight: bold;">{prob_cheap*100:.1f}%</p>
                                                <p style="margin: 0; color: #155724; font-size: 12px;">Cheap Confidence</p>
                                            </div>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-size: 28px; font-weight: bold;">{prob_expensive*100:.1f}%</p>
                                                <p style="margin: 0; color: #721c24; font-size: 12px;">Expensive Confidence</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
                                            border-left: 6px solid #dc3545; padding: 20px; border-radius: 8px; margin: 20px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center; gap: 15px;">
                                            <span style="font-size: 32px; color: #dc3545;">🔴</span>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-weight: bold; font-size: 24px;">EXPENSIVE</p>
                                                <p style="margin: 0; color: #721c24; font-size: 14px;">Predicted Price Category</p>
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 30px; text-align: center;">
                                            <div>
                                                <p style="margin: 0; color: #155724; font-size: 28px; font-weight: bold;">{prob_cheap*100:.1f}%</p>
                                                <p style="margin: 0; color: #155724; font-size: 12px;">Cheap Confidence</p>
                                            </div>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-size: 28px; font-weight: bold;">{prob_expensive*100:.1f}%</p>
                                                <p style="margin: 0; color: #721c24; font-size: 12px;">Expensive Confidence</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
                else:
                    st.error("Random Forest model not loaded. Please run train_random_forest.py to train and generate the model files.")

        # ---------------------------------------------------------
        # TAB 2: MODEL INSIGHTS
        # ---------------------------------------------------------
        with tab2:
            st.write("This section breaks down the performance and behavior of the **Random Forest** model.")
            
            if rf_model and rf_metadata:
                # Performance Overview Table
                st.subheader("🏆 Model Performance")
                
                cm = np.array(rf_metadata['confusion_matrix'])
                tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                
                accuracy = rf_metadata.get('accuracy', (tp + tn) / (tp + tn + fp + fn))
                precision = rf_metadata.get('precision', tp / (tp + fp) if (tp + fp) > 0 else 0)
                recall = rf_metadata.get('recall', tp / (tp + fn) if (tp + fn) > 0 else 0)
                f1_score = rf_metadata.get('f1_score', 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{precision*100:.2f}%")
                with col3:
                    st.metric("Recall", f"{recall*100:.2f}%")
                with col4:
                    st.metric("F1-Score", f"{f1_score*100:.2f}%")
                    
                st.markdown("---")
                
                # Confusion Matrix
                colLeft, colRight = st.columns([1, 2])
                with colLeft:
                    st.subheader("🔢 Confusion Matrix")
                    st.write(f"**True Negatives (TN): {tn:,}**")
                    
                    st.write(f"**False Positives (FP): {fp:,}**")
                    
                    st.write(f"**False Negatives (FN): {fn:,}**")
                    
                    st.write(f"**True Positives (TP): {tp:,}**")
                    
                with colRight:
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Cheap', 'Predicted Expensive'],
                        y=['Actually Cheap', 'Actually Expensive'],
                        text=cm,
                        texttemplate='%{text}',
                        colorscale='Oranges',
                        showscale=False
                    ))
                    fig_cm.update_layout(height=500, xaxis_title="Predicted Label", yaxis_title="True Label")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                st.markdown("---")
                
                # Feature Importance
                st.subheader("🔑 Feature Importance")
                st.write("Which features influenced the price classification the most?")
                
                feature_importance = pd.Series(
                    rf_model.feature_importances_,
                    index=rf_metadata['original_feature_columns']
                ).sort_values(ascending=True)
                
                fig_importance = px.bar(
                    x=feature_importance.values,
                    y=feature_importance.index,
                    orientation='h',
                    title="Importance of Apartment Features",
                    labels={'x': 'Importance Score', 'y': 'Feature'},
                    color=feature_importance.values,
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                    
            else:
                st.error("Random Forest model or metadata not loaded. Please ensure the model files exist.")
    
    elif selected_model == "K-Nearest Neighbors (KNN)":
        st.header("K-Nearest Neighbors (KNN)")
        
        tab1, tab2 = st.tabs(["🔮 Make a Prediction", "📈 Model Insights"])
        
        # ---------------------------------------------------------
        # TAB 1: PREDICTION
        # ---------------------------------------------------------
        with tab1:
            st.subheader("Interactive Prediction")
            st.write("Adjust the apartment features below to predict whether it would be classified as Cheap or Expensive.")
            
            with st.form("knn_prediction_form"):
                # Row 1: Property Type & State
                col1, col2 = st.columns(2)
                with col1:
                    property_type = st.selectbox("Property Type", ["Apartment", "Condo", "House"], key="knn_property")
                with col2:
                    state = st.selectbox("State", options=sorted(list(STATE_NAME_TO_CODE.keys())), index=sorted(list(STATE_NAME_TO_CODE.keys())).index("Texas"), key="knn_state")
                
                # Row 2: Square Feet & Monthly Fee
                col1, col2 = st.columns(2)
                with col1:
                    sqft = st.number_input("Square Feet", min_value=200, max_value=5000, value=1000, step=50, key="knn_sqft")
                with col2:
                    fee = st.selectbox("Monthly Fee", ["No", "Yes"], key="knn_fee")
                
                # Row 3: Bedrooms & Bathrooms
                col1, col2 = st.columns(2)
                with col1:
                    beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1, key="knn_beds")
                with col2:
                    baths = st.number_input("Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5, key="knn_baths")
                
                # Row 4: City Average Rent
                city_avg = st.number_input("City Avg. Rent ($)", min_value=500.0, max_value=5000.0, value=1500.0, step=50.0, key="knn_city_avg")
                
                submit = st.form_submit_button("Predict Rent Price Category", use_container_width=True)
                
            if submit:
                if knn_model:
                    apt_val = 1 if property_type == "Apartment" else 0
                    condo_val = 1 if property_type == "Condo" else 0
                    state_code = STATE_NAME_TO_CODE[state]
                    fee_val = 1 if fee == "Yes" else 0
                    
                    input_data = pd.DataFrame({
                        'bathrooms': [float(baths)],
                        'bedrooms': [int(beds)],
                        'square_feet': [float(sqft)],
                        'city_avg_price': [float(city_avg)],
                        'state': [state_code],
                        'housing/rent/apartment': [apt_val],
                        'housing/rent/condo': [condo_val],
                        'fee_yes': [fee_val]
                    })
                    
                    try:
                        prediction = knn_model.predict(input_data)[0]
                        probabilities = knn_model.predict_proba(input_data)[0]
                        prob_expensive = probabilities[1]
                        prob_cheap = probabilities[0]
                        
                        # Modern result display
                        if prediction == 0:
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                                            border-left: 6px solid #28a745; padding: 20px; border-radius: 8px; margin: 20px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center; gap: 15px;">
                                            <span style="font-size: 32px; color: #28a745;">🟢</span>
                                            <div>
                                                <p style="margin: 0; color: #155724; font-weight: bold; font-size: 24px;">CHEAP</p>
                                                <p style="margin: 0; color: #155724; font-size: 14px;">Predicted Price Category</p>
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 30px; text-align: center;">
                                            <div>
                                                <p style="margin: 0; color: #155724; font-size: 28px; font-weight: bold;">{prob_cheap*100:.1f}%</p>
                                                <p style="margin: 0; color: #155724; font-size: 12px;">Cheap Confidence</p>
                                            </div>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-size: 28px; font-weight: bold;">{prob_expensive*100:.1f}%</p>
                                                <p style="margin: 0; color: #721c24; font-size: 12px;">Expensive Confidence</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
                                            border-left: 6px solid #dc3545; padding: 20px; border-radius: 8px; margin: 20px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center; gap: 15px;">
                                            <span style="font-size: 32px; color: #dc3545;">🔴</span>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-weight: bold; font-size: 24px;">EXPENSIVE</p>
                                                <p style="margin: 0; color: #721c24; font-size: 14px;">Predicted Price Category</p>
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 30px; text-align: center;">
                                            <div>
                                                <p style="margin: 0; color: #155724; font-size: 28px; font-weight: bold;">{prob_cheap*100:.1f}%</p>
                                                <p style="margin: 0; color: #155724; font-size: 12px;">Cheap Confidence</p>
                                            </div>
                                            <div>
                                                <p style="margin: 0; color: #721c24; font-size: 28px; font-weight: bold;">{prob_expensive*100:.1f}%</p>
                                                <p style="margin: 0; color: #721c24; font-size: 12px;">Expensive Confidence</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
                else:
                    st.error("KNN model not loaded. Please run train_knn.py to train and generate the model files.")

        # ---------------------------------------------------------
        # TAB 2: MODEL INSIGHTS
        # ---------------------------------------------------------
        with tab2:
            st.write("This section breaks down the performance and behavior of the **K-Nearest Neighbors (KNN)** model.")
            
            if knn_model and knn_metadata:
                # Performance Overview Table
                st.subheader("🏆 Model Performance")
                
                cm = np.array(knn_metadata['confusion_matrix'])
                tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                
                accuracy = knn_metadata.get('accuracy', (tp + tn) / (tp + tn + fp + fn))
                precision = knn_metadata.get('precision', tp / (tp + fp) if (tp + fp) > 0 else 0)
                recall = knn_metadata.get('recall', tp / (tp + fn) if (tp + fn) > 0 else 0)
                f1_score = knn_metadata.get('f1_score', 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{precision*100:.2f}%")
                with col3:
                    st.metric("Recall", f"{recall*100:.2f}%")
                with col4:
                    st.metric("F1-Score", f"{f1_score*100:.2f}%")
                    
                st.markdown("---")
                
                # Confusion Matrix
                colLeft, colRight = st.columns([1, 2])
                with colLeft:
                    st.subheader("🔢 Confusion Matrix")
                    st.write(f"**True Negatives (TN): {tn:,}**")
                    
                    st.write(f"**False Positives (FP): {fp:,}**")
                    
                    st.write(f"**False Negatives (FN): {fn:,}**")
                    
                    st.write(f"**True Positives (TP): {tp:,}**")
                    
                with colRight:
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Cheap', 'Predicted Expensive'],
                        y=['Actually Cheap', 'Actually Expensive'],
                        text=cm,
                        texttemplate='%{text}',
                        colorscale='Purples',
                        showscale=False
                    ))
                    fig_cm.update_layout(height=500, xaxis_title="Predicted Label", yaxis_title="True Label")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
            else:
                st.error("KNN model or metadata not loaded. Please ensure the model files exist.")
