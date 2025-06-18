import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Define columns
num_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
nominal_columns = ['Item_Type', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Identifier']
ordinal_columns = ['Item_Fat_Content', 'Outlet_Size']

VALID_CATEGORIES = {
    'Item_Type': ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 
                  'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 
                  'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood'],
    'Item_Fat_Content': ['Low Fat', 'Regular'],
    'Outlet_Size': ['Small', 'Medium', 'High'],
    'Outlet_Location_Type': ['Tier 1', 'Tier 2', 'Tier 3'],
    'Outlet_Type': ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']
}

def create_feature_vector(input_data):
    df = pd.DataFrame([input_data])
    required_columns = num_columns + ordinal_columns + nominal_columns
    df = df[required_columns]
    return df

def load_model():
    try:
        model = pickle.load(open("XGBoost_GPU_best_model.pkl", "rb"))
        if hasattr(model, 'get_params'):
            params = model.get_params()
            if 'tree_method' in params and params['tree_method'] == 'gpu_hist':
                model.set_params(tree_method='hist', device='cuda')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_sales(features):
    try:
        model = load_model()
        if model is None:
            return None
        feature_df = create_feature_vector(features)
        prediction = model.predict(feature_df)
        return np.exp(prediction)[0]
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="🛒 Store Sales Predictor", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #FF6F61;'>🛍️ Smart Store Sales Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>Fill in product and outlet details to get an instant sales estimate</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.form("prediction_form"):
        with st.container():
            st.markdown("### 📦 Product Information")
            prod_col1, prod_col2 = st.columns([1, 1])

            with prod_col1:
                item_weight = st.slider('Item Weight (kg)', 0.0, 100.0, 12.0)
                item_fat_content = st.radio('Fat Content', options=VALID_CATEGORIES['Item_Fat_Content'], horizontal=True)
                item_visibility = st.slider('Visibility (0-1)', 0.0, 1.0, 0.1)

            with prod_col2:
                item_type = st.selectbox('Product Type', options=VALID_CATEGORIES['Item_Type'])
                item_mrp = st.number_input('Item MRP (₹)', min_value=0.0, step=1.0, value=100.0)

        st.markdown("---")

        with st.container():
            st.markdown("### 🏪 Outlet Information")
            store_col1, store_col2 = st.columns([1, 1])

            with store_col1:
                outlet_identifier = st.selectbox('Outlet ID', options=[
                    'OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019',
                    'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'
                ])
                outlet_establishment_year = st.number_input('Established Year', min_value=1900, max_value=2025, value=2000)

            with store_col2:
                outlet_size = st.radio('Outlet Size', options=VALID_CATEGORIES['Outlet_Size'], horizontal=True)
                outlet_location_type = st.selectbox('Location Type', options=VALID_CATEGORIES['Outlet_Location_Type'])
                outlet_type = st.selectbox('Outlet Type', options=VALID_CATEGORIES['Outlet_Type'])

        st.markdown("---")
        submit = st.form_submit_button("🚀 Predict Sales Now")

    if submit:
        input_data = {
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier': outlet_identifier,
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location_type,
            'Outlet_Type': outlet_type
        }

        sales = predict_sales(input_data)

        if sales is not None:
            st.success(f"🎯 **Estimated Sales: ₹{sales:,.2f}**")
            st.markdown("#### 🔍 Prediction Summary")
            st.markdown("""
                <ul>
                    <li>✔️ Based on real historical data</li>
                    <li>✔️ Considers both product and outlet factors</li>
                    <li>✔️ Optimized using XGBoost and log-scale transformation</li>
                </ul>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.info("""
            💡 Tip: You can adjust MRP, visibility, or outlet size to see how sales change.
            """)

if __name__ == '__main__':
    main()
