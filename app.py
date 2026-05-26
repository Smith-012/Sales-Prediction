"""
Streamlit Interactive App - Sales Prediction
Shows model predictions with SHAP explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Sales Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3em; color: #d62728; font-weight: bold; text-align: center; margin-bottom: 0.5em; }
    .metric-card { background-color: #f0f2f6; padding: 1.5em; border-radius: 0.5em; border-left: 4px solid #d62728; }
    .prediction-box { background-color: #ffe6e6; padding: 2em; border-radius: 0.5em; text-align: center; }
    .sales-display { color: #d62728; font-weight: bold; font-size: 2.5em; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💰 Sales Prediction</div>', unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_model_and_data():
    model = joblib.load("artifacts/model.joblib")
    df_train = pd.read_csv("data/sales.csv")
    return model, df_train

try:
    model, df_train = load_model_and_data()
except FileNotFoundError:
    st.error("⚠️ Model not found! Please run `python main.py --data data/sales.csv --out artifacts` first.")
    st.stop()

# Sidebar - Input features
st.sidebar.header("📊 Sales Metrics")

col1, col2 = st.sidebar.columns(2)
with col1:
    advertising = st.number_input("Advertising Spend ($K)", min_value=0.0, max_value=300.0, value=100.0)
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])

with col2:
    customer_count = st.number_input("Customer Count", min_value=10, max_value=5000, value=500)
    product_count = st.number_input("Product Types", min_value=1, max_value=100, value=25)

# Create input dataframe
input_data = pd.DataFrame({
    'advertising': [advertising],
    'customer_count': [customer_count],
    'product_count': [product_count]
})

# Add categorical features
regions = ["North", "South", "East", "West"]
seasons = ["Spring", "Summer", "Fall", "Winter"]

for r in regions:
    input_data[f'region_{r.lower()}'] = [1 if r == region else 0]

for s in seasons:
    input_data[f'season_{s.lower()}'] = [1 if s == season else 0]

# Main prediction
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Sales Forecast")
    
    try:
        # Make prediction
        if isinstance(model, dict):
            preprocessor = model.get('preprocessor')
            regressor = model.get('model')
        else:
            if hasattr(model, 'named_steps'):
                preprocessor = model.named_steps.get('preprocessor')
                regressor = model.named_steps.get('regressor')
            else:
                preprocessor = None
                regressor = model
        
        # Transform input
        if preprocessor:
            input_transformed = preprocessor.transform(input_data)
        else:
            input_transformed = input_data
        
        # Get prediction
        prediction = regressor.predict(input_transformed)[0]
        
        # Display prediction
        st.markdown(f'<div class="prediction-box"><div class="sales-display">${prediction:,.0f}</div></div>', unsafe_allow_html=True)
        st.markdown(f"**Predicted Sales:** ${prediction:,.2f}")
        
        # Confidence interval estimate
        st.markdown("### Sales Range (Est.)")
        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15
        st.info(f"Estimated range: **${lower_bound:,.0f}** - **${upper_bound:,.0f}**")
        
        # Model metrics
        st.markdown("### Model Performance")
        metrics_file = Path("artifacts/metrics.txt")
        if metrics_file.exists():
            metrics_text = metrics_file.read_text()
            for line in metrics_text.split('\n')[:5]:
                if line.strip():
                    st.text(line)
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

with col2:
    st.subheader("🔍 SHAP Explainability")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(regressor) if hasattr(regressor, 'tree_') else shap.KernelExplainer(regressor.predict, input_transformed[:1])
        shap_values = explainer.shap_values(input_transformed)
        
        # Handle array format
        if isinstance(shap_values, np.ndarray):
            shap_vals = shap_values
        else:
            shap_vals = shap_values
        
        # Display feature importance
        fig, ax = plt.subplots(figsize=(10, 4))
        feature_names = preprocessor.get_feature_names_out() if preprocessor else input_data.columns.tolist()
        
        # Create waterfall plot data
        importance_indices = np.argsort(np.abs(shap_vals[0]))[-5:]
        top_features = [feature_names[i] if isinstance(feature_names, np.ndarray) else feature_names[i] for i in importance_indices]
        top_values = shap_vals[0][importance_indices]
        
        ax.barh(range(len(top_features)), top_values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('SHAP Value (Impact on Sales)')
        ax.set_title('Top Features Influencing Sales')
        plt.tight_layout()
        st.pyplot(fig)
    
    except Exception as e:
        st.warning(f"SHAP visualization: {str(e)}")
        st.info("📊 Feature importance loading...")

# Dataset Info
st.markdown("---")
st.subheader("📈 Dataset Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df_train))
    st.metric("Avg Sales", f"${df_train.iloc[:, -1].mean():,.0f}")

with col2:
    st.metric("Sales Range", f"${df_train.iloc[:, -1].min():,.0f} - ${df_train.iloc[:, -1].max():,.0f}")
    st.metric("Std Dev", f"${df_train.iloc[:, -1].std():,.0f}")

with col3:
    st.metric("Avg Advertising", f"${df_train['advertising'].mean():,.0f}K")
    st.metric("Avg Customers", f"{df_train['customer_count'].mean():.0f}")

st.markdown("""
---
**📚 About This App:**
- Built with Streamlit for interactive predictions
- Uses SHAP for model explainability
- Predicts sales based on multiple factors
- Model: Gradient Boosting Regressor with hyperparameter tuning
""")
