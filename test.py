import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from price_fairness import classify_price_fairness
import warnings
warnings.filterwarnings('ignore')

# ===============================================
# STATE MAPPING & CONSTANTS
# ===============================================
# Mapping of state codes to state names (1-50)
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

# Property type mapping
PROPERTY_TYPE_MAPPING = {
    "Apartment": (1, 0),
    "Condo": (0, 1)
}

# ===============================================
# PAGE CONFIGURATION
# ===============================================
st.set_page_config(
    page_title="Apartment Rental Market - ML Model Showcase", 
    layout="wide",
    page_icon="🏙️"
)

st.title("🏙️ Apartment Rental Market - ML Model Showcase")
st.markdown("""
**Data Science Assignment**: Comparison of multiple machine learning models for apartment price prediction and classification.
""")

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
# LOAD LOGISTIC REGRESSION MODEL & METADATA
# ===============================================
@st.cache_resource
def load_logreg_model():
    try:
        with open('model/logistic_regression_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_logreg_metadata():
    try:
        with open('model/logistic_regression_metadata.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

logreg_model = load_logreg_model()
logreg_metadata = load_logreg_metadata()

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
# TABS FOR DIFFERENT SECTIONS
# ===============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Logistic Regression",
    "Decision Tree",
    "Association Rules",
    "Price Fairness"
])

# ===============================================
# TAB 1: LOGISTIC REGRESSION
# ===============================================
with tab1:
    st.header("Logistic Regression Classifier")
    
    if logreg_model and logreg_metadata:
        st.markdown("""
        **Task:** Classify apartments as "Cheap" (≤80th percentile) or "Expensive" (>80th percentile)
        
        **Algorithm:** Logistic Regression with Polynomial Features (degree=2, interactions only)
        
        **Key Strength:** Shows probability of classification and feature impact on predictions.
        """)
        
        # Model metrics
        st.subheader("📈 Performance Metrics")
        
        metric1, metric2 = st.columns(2)
        with metric1:
            st.metric("Accuracy", f"{logreg_metadata['accuracy']*100:.2f}%")
        with metric2:
            st.metric("AUC-ROC", f"{logreg_metadata['auc_score']:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        
        col1, col2 = st.columns(2)
        with col1:
            cm = np.array(logreg_metadata['confusion_matrix'])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Cheap', 'Predicted Expensive'],
                y=['Actually Cheap', 'Actually Expensive'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues',
                showscale=False
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                height=400,
                xaxis_title="Predicted Label",
                yaxis_title="True Label"
            )
            st.plotly_chart(fig_cm, width='stretch')
        
        with col2:
            st.write("**Interpretation:**")
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            
            st.write(f"**True Negatives (TN): {tn:,}**")
            st.write("✅ Correctly identified cheap")
            
            st.write(f"**False Positives (FP): {fp:,}**")
            st.write("❌ Cheap wrongly classified as expensive")
            
            st.write(f"**False Negatives (FN): {fn:,}**")
            st.write("❌ Expensive wrongly classified as cheap")
            
            st.write(f"**True Positives (TP): {tp:,}**")
            st.write("✅ Correctly identified expensive")
        
        st.markdown("---")
        
        # ROC Curve
        st.subheader("📊 ROC Curve - Model Discrimination Ability")
        
        fpr = logreg_metadata['fpr']
        tpr = logreg_metadata['tpr']
        auc_score = logreg_metadata['auc_score']
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Logistic Regression (AUC = {auc_score:.4f})',
            line=dict(color='#0066CC', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier (AUC = 0.5)',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            hovermode='closest',
            showlegend=True
        )
        st.plotly_chart(fig_roc, width='stretch')
        
        
        st.markdown("---")
        
        # Simple Probability Calculator
        st.subheader("🔮 Quick Prediction: 2-Factor Calculator")
        st.write("*Predict apartment classification using Square Feet and City Average Price*")
        
        if 'logreg_sqft' not in st.session_state:
            st.session_state.logreg_sqft = 1000
        if 'logreg_city_avg' not in st.session_state:
            st.session_state.logreg_city_avg = 1500
        
        col1, col2 = st.columns(2)
        
        with col1:
            sqft = st.slider(
                "Square Feet",
                min_value=300,
                max_value=4000,
                step=100,
                value=st.session_state.logreg_sqft,
                key='logreg_sqft'
            )
        
        with col2:
            city_avg = st.slider(
                "City Average Price ($)",
                min_value=500,
                max_value=5000,
                step=100,
                value=st.session_state.logreg_city_avg,
                key='logreg_city_avg'
            )
        
        # Default values for non-slider features
        input_data = pd.DataFrame({
            'bathrooms': [2.0],
            'bedrooms': [2],
            'square_feet': [float(sqft)],
            'housing/rent/apartment': [1],
            'housing/rent/condo': [0],
            'fee_yes': [0],
            'city_avg_price': [float(city_avg)],
            'state': [4]  # Arkansas
        })
        
        try:
            prediction = logreg_model.predict(input_data)[0]
            probabilities = logreg_model.predict_proba(input_data)[0]
            
            prob_cheap = probabilities[0] * 100
            prob_expensive = probabilities[1] * 100
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 0:
                    st.success(f"### 🟢 CHEAP")
                else:
                    st.error(f"### 🔴 EXPENSIVE")
            
            with col2:
                st.metric("Probability - Cheap", f"{prob_cheap:.1f}%")
            
            with col3:
                st.metric("Probability - Expensive", f"{prob_expensive:.1f}%")
            
            # Show input summary
            st.write("**Input Values:**")
            st.write(f"- Square Feet: {sqft:,} sqft")
            st.write(f"- City Average Price: ${city_avg:,}")
            st.write(f"- Other features set to typical apartment (2 bed, 2 bath, no fee)")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    else:
        st.error("❌ Logistic Regression model not loaded. Please run training script first.")

# ===============================================
# TAB 2: DECISION TREE
# ===============================================
with tab2:
    st.header("Decision Tree Classifier")
    
    if dt_model and dt_metadata:
        st.markdown("""
        **Task:** Understanding the importance of features that influence apartment price 
        
        **Algorithm:** Decision Tree Classifier (max_depth=8)
        """)
        
        # Model metrics
        st.subheader("📈 Performance Metrics")
        
        metric1, metric2, metric3, metric4 = st.columns(4)
        with metric1:
            st.metric("Accuracy", f"{dt_metadata['accuracy']*100:.2f}%")
        with metric2:
            st.metric("AUC-ROC", f"{dt_metadata['auc_score']:.4f}")
        with metric3:
            st.metric("Training Samples", f"{dt_metadata['n_samples_train']:,}")
        with metric4:
            st.metric("Test Samples", f"{dt_metadata['n_samples_test']:,}")
        
        # Features used
        st.subheader("🎯 Features Used")
        feature_list = dt_metadata['original_feature_columns']
        features_cols = st.columns(len(feature_list))
        for i, feat in enumerate(feature_list):
            with features_cols[i]:
                st.write(f"• {feat}")
        
        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix (from test set)")
        
        col1, col2 = st.columns(2)
        with col1:
            cm = np.array(dt_metadata['confusion_matrix'])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Cheap', 'Predicted Expensive'],
                y=['Actually Cheap', 'Actually Expensive'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Greens',
                showscale=False
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                height=400,
                xaxis_title="Predicted Label",
                yaxis_title="True Label"
            )
            st.plotly_chart(fig_cm, width='stretch')
        
        with col2:
            st.write("**Interpretation:**")
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            
            st.write(f"**True Negatives (TN): {tn:,}**")
            st.write("✅ Correctly identified cheap apartments")
            
            st.write(f"**False Positives (FP): {fp:,}**")
            st.write("❌ Cheap apartments wrongly classified as expensive")
            
            st.write(f"**False Negatives (FN): {fn:,}**")
            st.write("❌ Expensive apartments wrongly classified as cheap")
            
            st.write(f"**True Positives (TP): {tp:,}**")
            st.write("✅ Correctly identified expensive apartments")
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            st.metric("Sensitivity (Recall)", f"{sensitivity:.2%}")
            st.metric("Specificity", f"{specificity:.2%}")
            st.metric("Precision", f"{precision:.2%}")
        
        # Feature Importance
        st.subheader("📊 Feature Importance")
        
        feature_importance = pd.Series(
            dt_model.feature_importances_,
            index=dt_metadata['original_feature_columns']
        ).sort_values(ascending=True)
        
        fig_importance = px.bar(
            x=feature_importance.values,
            y=feature_importance.index,
            orientation='h',
            title="Feature Importance in Decision Tree",
            labels={'x': 'Importance Score', 'y': 'Feature'},
            color=feature_importance.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_importance, width='stretch')
        
        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        
        # Calculate metrics for classification report
        cm = np.array(dt_metadata['confusion_matrix'])
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        
        # Cheap class metrics
        cheap_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        cheap_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        cheap_f1 = 2 * (cheap_precision * cheap_recall) / (cheap_precision + cheap_recall) if (cheap_precision + cheap_recall) > 0 else 0
        cheap_support = tn + fn
        
        # Expensive class metrics
        exp_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        exp_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        exp_f1 = 2 * (exp_precision * exp_recall) / (exp_precision + exp_recall) if (exp_precision + exp_recall) > 0 else 0
        exp_support = tp + fn
        
        # Weighted average
        total = tn + tp + fn + fp
        wavg_precision = (cheap_precision * cheap_support + exp_precision * exp_support) / total if total > 0 else 0
        wavg_recall = (cheap_recall * cheap_support + exp_recall * exp_support) / total if total > 0 else 0
        wavg_f1 = (cheap_f1 * cheap_support + exp_f1 * exp_support) / total if total > 0 else 0
        
        report_data = {
            'Class': ['Cheap', 'Expensive', 'Weighted Average'],
            'Precision': [f'{cheap_precision:.3f}', f'{exp_precision:.3f}', f'{wavg_precision:.3f}'],
            'Recall': [f'{cheap_recall:.3f}', f'{exp_recall:.3f}', f'{wavg_recall:.3f}'],
            'F1-Score': [f'{cheap_f1:.3f}', f'{exp_f1:.3f}', f'{wavg_f1:.3f}'],
            'Support': [f'{cheap_support:,}', f'{exp_support:,}', f'{total:,}']
        }
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, width='stretch', hide_index=True)
        
        # ===============================================
        # INTERACTIVE FEATURE EXPLORER SECTION
        # ===============================================
        st.markdown("---")
        st.subheader("🎚️ Interactive Feature Explorer")
        st.write("""
        Adjust the sliders below to see how each feature affects the model's prediction in **real-time**.
        Notice how features with higher importance (like Square Feet and City Average Price) have more impact on the classification.
        """)
        
        # Initialize session state for dt interactive features
        if 'dt_interactive_bathrooms' not in st.session_state:
            st.session_state.dt_interactive_bathrooms = 2.0
        if 'dt_interactive_bedrooms' not in st.session_state:
            st.session_state.dt_interactive_bedrooms = 2
        if 'dt_interactive_sqft' not in st.session_state:
            st.session_state.dt_interactive_sqft = 1000
        if 'dt_interactive_apartment' not in st.session_state:
            st.session_state.dt_interactive_apartment = "Apartment"
        if 'dt_interactive_fee' not in st.session_state:
            st.session_state.dt_interactive_fee = "No"
        if 'dt_interactive_city_avg' not in st.session_state:
            st.session_state.dt_interactive_city_avg = 1500
        if 'dt_interactive_state' not in st.session_state:
            st.session_state.dt_interactive_state = "Arkansas"
        
        # Get feature importance for highlighting
        feature_importance = pd.Series(
            dt_model.feature_importances_,
            index=dt_metadata['original_feature_columns']
        ).sort_values(ascending=False)
        
        # Create two columns for interactive controls
        feature_col1, feature_col2 = st.columns(2)
        
        # ===== LEFT COLUMN: PROPERTY FEATURES =====
        with feature_col1:
            st.write("### 🏠 Property Features")
            
            st.session_state.dt_interactive_bathrooms = st.slider(
                "🛁 Bathrooms",
                min_value=0.0,
                max_value=5.0,
                step=0.5,
                value=st.session_state.dt_interactive_bathrooms,
                key='dt_bath_slider'
            )
            
            st.session_state.dt_interactive_bedrooms = st.slider(
                "🛏️ Bedrooms",
                min_value=0,
                max_value=6,
                step=1,
                value=st.session_state.dt_interactive_bedrooms,
                key='dt_bed_slider'
            )
            
            st.session_state.dt_interactive_sqft = st.slider(
                "📐 Square Feet (IMPORTANT)",
                min_value=300,
                max_value=4000,
                step=100,
                value=st.session_state.dt_interactive_sqft,
                key='dt_sqft_slider',
                help=f"High importance score: {feature_importance.get('square_feet', 0):.4f}"
            )
            
            st.session_state.dt_interactive_apartment = st.selectbox(
                "🏢 Property Type",
                ["Apartment", "Condo"],
                index=0 if st.session_state.dt_interactive_apartment == "Apartment" else 1,
                key='dt_prop_select'
            )
        
        # ===== RIGHT COLUMN: MARKET FEATURES =====
        with feature_col2:
            st.write("### 💰 Market Information")
            
            st.session_state.dt_interactive_fee = st.selectbox(
                "💳 Monthly Fee?",
                ["No", "Yes"],
                index=0 if st.session_state.dt_interactive_fee == "No" else 1,
                key='dt_fee_select'
            )
            
            st.session_state.dt_interactive_city_avg = st.slider(
                "📊 City Average Price (IMPORTANT)",
                min_value=500,
                max_value=5000,
                step=100,
                value=st.session_state.dt_interactive_city_avg,
                key='dt_city_slider',
                help=f"High importance score: {feature_importance.get('city_avg_price', 0):.4f}"
            )
            
            st.session_state.dt_interactive_state = st.selectbox(
                "📍 State",
                sorted(STATE_NAME_TO_CODE.keys()),
                index=sorted(STATE_NAME_TO_CODE.keys()).index(st.session_state.dt_interactive_state),
                key='dt_state_select'
            )
        
        # ===== REAL-TIME PREDICTION =====
        st.markdown("---")
        
        # Prepare input data based on slider values
        apartment_val, condo_val = PROPERTY_TYPE_MAPPING[st.session_state.dt_interactive_apartment]
        state_code = STATE_NAME_TO_CODE[st.session_state.dt_interactive_state]
        
        # IMPORTANT: Column order must match training data exactly
        input_data = pd.DataFrame({
            'bathrooms': [float(st.session_state.dt_interactive_bathrooms)],
            'bedrooms': [int(st.session_state.dt_interactive_bedrooms)],
            'square_feet': [float(st.session_state.dt_interactive_sqft)],
            'city_avg_price': [float(st.session_state.dt_interactive_city_avg)],
            'state': [state_code],
            'housing/rent/apartment': [apartment_val],
            'housing/rent/condo': [condo_val],
            'fee_yes': [1 if st.session_state.dt_interactive_fee == "Yes" else 0]
        })
        
        try:
            # Get prediction and probabilities
            prediction = dt_model.predict(input_data)[0]
            probabilities = dt_model.predict_proba(input_data)[0]
            
            prob_expensive = probabilities[1]  # Probability of being in higher price category
            
            # ===== ESTIMATED PRICE DISPLAY =====
            st.markdown("---")
            st.subheader("💰 Estimated Price Range")
            
            # Get similar apartments from dataset based on prediction
            df_sample = df_main[df_main['is_expensive'] == prediction]
            if len(df_sample) > 0:
                estimated_price = df_sample['price'].median()
                price_std = df_sample['price'].std()
                price_low = estimated_price - (price_std * 0.5)
                price_high = estimated_price + (price_std * 0.5)
                
                # Display as prominent price cards
                price_col1, price_col2, price_col3 = st.columns(3)
                
                with price_col1:
                    st.metric(
                        "Low End",
                        f"${price_low:,.0f}",
                        delta=f"Based on {len(df_sample):,} similar apartments"
                    )
                
                with price_col2:
                    st.metric(
                        "Expected Price",
                        f"${estimated_price:,.0f}",
                        delta=f"Median of your category"
                    )
                
                with price_col3:
                    st.metric(
                        "High End",
                        f"${price_high:,.0f}",
                        delta=f"Price range ±50%"
                    )
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    else:
        st.error("❌ Decision Tree model not loaded. Please run training script first.")

# ===============================================
# TAB 3: ASSOCIATION RULES & UNSUPERVISED LEARNING
# ===============================================
with tab3:
    st.header("🔗 Association Rules Mining - Unsupervised Learning")
    
    st.markdown("""
    **Machine Learning Type:** Unsupervised Learning - Association Analysis
    
    This model discovers **hidden patterns and associations** between apartment features without a predefined target variable.
    We analyze the **frequency and co-occurrence** of price categories across different states to understand market segmentation and price patterns.
    
    **Key Insight:** Unlike supervised models (Logistic Regression, Decision Tree), this unsupervised approach finds natural groupings 
    and associations in the data, revealing which price segments dominate specific markets.
    """)
    
    st.info("""
    **📊 Methodology:**
    - **Price Segmentation (Clustering):** Using quantile-based clustering (quintiles), we create 5 natural price segments
    - **Association Detection:** Find relationships between state location and price segment (antecedent → consequent)
    - **Pattern Extraction:** Calculate support (frequency), confidence (co-occurrence rate), and market dominance
    - **No Target Variable:** This is unsupervised - we don't predict a specific outcome, but discover inherent market structures
    """)
    
    # Prepare data for association analysis
    eda_data = df_main.dropna(subset=['state', 'price']).copy()
    eda_data['price_cat_intervals'] = pd.qcut(eda_data['price'], q=5, duplicates='drop')
    
    # Create mapping from interval to formatted range string
    price_range_mapping = {}
    for interval in eda_data['price_cat_intervals'].unique():
        if pd.notna(interval):
            price_range_mapping[interval] = f"({interval.left:.1f}, {interval.right:.1f}]"
    
    # Create labeled version for display
    eda_data['price_cat'] = eda_data['price_cat_intervals'].map(price_range_mapping)
    
    # ===== SECTION 1: PRICE SEGMENTS (UNSUPERVISED CLUSTERING) =====
    st.subheader("💰 Price Segmentation Analysis")
    st.write("*Unsupervised clustering of apartments into 5 natural price segments (quintiles)*")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        st.write("**Market Price Distribution**")
        fig_hist = px.histogram(
            eda_data,
            x='price',
            nbins=50,
            title='Distribution of Apartment Prices',
            labels={'price': 'Monthly Rent ($)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, width='stretch', use_container_width=True)
    
    with dist_col2:
        st.write("**Price Statistics**")
        stats = {
            'Mean Price': f"${eda_data['price'].mean():,.2f}",
            'Median Price': f"${eda_data['price'].median():,.2f}",
            'Std Dev': f"${eda_data['price'].std():,.2f}",
            'Min Price': f"${eda_data['price'].min():,.2f}",
            'Max Price': f"${eda_data['price'].max():,.2f}",
            'Q1 (25%)': f"${eda_data['price'].quantile(0.25):,.2f}",
            'Q3 (75%)': f"${eda_data['price'].quantile(0.75):,.2f}"
        }
        for label, value in stats.items():
            st.metric(label, value)
    
    # ===== SECTION 2: ASSOCIATION STRENGTH - SEGMENT FREQUENCIES =====
    st.markdown("---")
    st.subheader("📊 Price Segment Frequencies")
    st.write("*How apartments distribute across unsupervised price clusters*")
    
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        st.write("**Count by Price Category**")
        cat_counts = eda_data['price_cat'].value_counts().sort_index()
        fig_cat = px.bar(
            x=cat_counts.index,
            y=cat_counts.values,
            title='Apartments by Price Category',
            labels={'x': 'Price Category', 'y': 'Count'},
            color=cat_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_cat.update_layout(height=400)
        st.plotly_chart(fig_cat, width='stretch', use_container_width=True)
    
    with cat_col2:
        st.write("**Price Ranges by Category**")
        price_categories_sorted = sorted(eda_data['price_cat'].dropna().unique(), key=lambda x: float(x.split(',')[0].replace('(', '')))
        for cat in price_categories_sorted:
            cat_data = eda_data[eda_data['price_cat'] == cat]['price']
            if len(cat_data) > 0:
                st.metric(
                    cat,
                    f"${cat_data.min():,.0f} - ${cat_data.max():,.0f}",
                    delta=f"{len(cat_data):,} apartments"
                )
    
    # ===== SECTION 3: ASSOCIATION RULES - STATE × PRICE PATTERNS =====
    st.markdown("---")
    st.subheader("🗺️ State-Price Association Analysis")
    st.write("*Association rules mining: Which price segments associate with which states? Discover market patterns.*")
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Filter: Number of top states
        top_n = st.slider(
            "Show Top N States by Apartment Count",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
    
    with filter_col2:
        # Filter: Price category selection
        price_categories = sorted(eda_data['price_cat'].dropna().unique())
        selected_categories = st.multiselect(
            "Filter by Price Category",
            options=price_categories,
            default=price_categories
        )
    
    # Apply filters
    filtered_eda = eda_data[eda_data['price_cat'].isin(selected_categories)]
    
    # Get top states
    state_counts = filtered_eda.groupby('state').size().sort_values(ascending=False).head(top_n).index
    grouped_data = filtered_eda[filtered_eda['state'].isin(state_counts)].groupby(['state', 'price_cat']).size().unstack(fill_value=0)
    
    if len(grouped_data) > 0:
        # Convert state codes to state names
        grouped_data.index = grouped_data.index.map(lambda x: STATE_CODE_TO_NAME.get(int(x), str(x)) if pd.notna(x) else str(x))
        
        # Reorder columns by price range (numerically)
        col_order = sorted(grouped_data.columns, key=lambda x: float(x.split(',')[0].replace('(', '')))
        grouped_data = grouped_data[col_order]
        
        # Convert to long format for proper labeling
        grouped_data_long = grouped_data.reset_index().melt(
            id_vars='state',
            var_name='Price Category',
            value_name='Number of Apartments'
        )
        
        # Order by price range
        grouped_data_long['Price Category'] = pd.Categorical(
            grouped_data_long['Price Category'],
            categories=col_order,
            ordered=True
        )
        
        fig_state = px.bar(
            grouped_data_long,
            x='state',
            y='Number of Apartments',
            color='Price Category',
            barmode='group',
            title=f'Price Category Distribution - Top {top_n} States',
            height=600,
            category_orders={'Price Category': col_order}
        )
        fig_state.update_layout(
            xaxis_tickangle=-45,
            margin=dict(b=150),
            xaxis=dict(title='State', tickfont=dict(size=12)),
            yaxis=dict(title='Number of Apartments'),
            legend=dict(title="Price Categories")
        )
        st.plotly_chart(fig_state, width='stretch', use_container_width=True)
        
        # Show summary table
        st.write("**Association Rules Summary Table**")
        st.write("*Support values: How often each State × Price Segment association occurs in the dataset*")
        summary_table = grouped_data.copy()
        summary_table.index.name = 'State'
        summary_table['Total'] = summary_table.sum(axis=1)
        summary_table = summary_table.sort_values('Total', ascending=False)
        st.dataframe(summary_table, width='stretch')
    else:
        st.warning("No data available for selected filters. Try adjusting the filters.")

# ===============================================
# TAB 4: PRICE FAIRNESS ANALYSIS
# ===============================================
with tab4:
    st.header("Price Fairness Analysis")
    
    st.markdown("""
    Compare an apartment's actual rent with its fair market value. The model determines if a listing is **overpriced**, **underpriced**, or **fairly priced**.
    """)
    
    # Initialize session state for price fairness
    if 'actual_rent_fairness' not in st.session_state:
        st.session_state.actual_rent_fairness = 1500
    if 'fair_rent_fairness' not in st.session_state:
        st.session_state.fair_rent_fairness = 1500
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        actual_rent = st.number_input(
            "Actual Listed Rent ($)",
            min_value=100,
            max_value=10000,
            value=st.session_state.actual_rent_fairness,
            step=50,
            key="actual_rent_fairness"
        )
    
    with col2:
        fair_rent = st.number_input(
            "Fair Market Rent ($)",
            min_value=100,
            max_value=10000,
            value=st.session_state.fair_rent_fairness,
            step=50,
            key="fair_rent_fairness"
        )
    
    st.markdown("---")
    
    # Perform price fairness analysis with fixed threshold (10%)
    fairness_result = classify_price_fairness(actual_rent, fair_rent, threshold_percent=10)
    
    # Display result with color coding
    if fairness_result['status'] == "Overpriced":
        st.error(f"### 📈 {fairness_result['status'].upper()}")
        st.write("This apartment is listed **above** fair market value")
    elif fairness_result['status'] == "Underpriced":
        st.success(f"### 📉 {fairness_result['status'].upper()}")
        st.write("This apartment is listed **below** fair market value")
    else:
        st.info(f"### 📊 {fairness_result['status'].upper()}")
        st.write("This apartment is priced **according to** market standards")
    
    st.markdown("---")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Price Difference",
            f"${abs(fairness_result['difference']):,.0f}",
            delta=f"{fairness_result['percentage_difference']:+.1f}%"
        )
    
    with col2:
        st.metric(
            "Actual Rent",
            f"${actual_rent:,.0f}"
        )
    
    with col3:
        st.metric(
            "Fair Market Rent",
            f"${fair_rent:,.0f}"
        )
    
    st.markdown("---")
    
    # Comparison chart
    comparison_data = pd.DataFrame({
        'Type': ['Actual Listed Rent', 'Fair Market Rent'],
        'Price': [actual_rent, fair_rent]
    })
    
    fig_comparison = px.bar(
        comparison_data,
        x='Type',
        y='Price',
        title="Actual vs Fair Market Rent",
        labels={'Price': 'Monthly Rent ($)'},
        color='Type',
        color_discrete_map={
            'Actual Listed Rent': '#FF6B6B' if fairness_result['status'] == 'Overpriced' else ('#4ECDC4' if fairness_result['status'] == 'Underpriced' else '#95E1D3'),
            'Fair Market Rent': '#4ECDC4'
        }
    )
    st.plotly_chart(fig_comparison, width='stretch')

# ===============================================
# FOOTER
# ===============================================
st.markdown("---")
st.markdown("""
🏛️ **Data Science Assignment** | Multiple ML Models for Apartment Rent Prediction & Classification
Built with Streamlit | Dataset: Clean Apartment Data (99,492 listings)
""")
