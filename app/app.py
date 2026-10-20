import streamlit as st
import pandas as pd
import sys
import os
import __main__

# 1. Add the src directory to the system path so Python can find your modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 2. Apply the critical namespace fix before loading the model
from preprocessing import MercariPreprocessor
__main__.MercariPreprocessor = MercariPreprocessor

# 3. Import your existing predictor
from predict import MercariPricePredictor

st.set_page_config(page_title="Mercari Price Predictor", layout="centered")

# Cache the model so it only loads once when the app starts
@st.cache_resource
def load_model():
    # Point the artifact directory to the data folder
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    return MercariPricePredictor(artifact_dir=base_path)

try:
    predictor = load_model()
    st.success("Machine Learning Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Marketplace Price Suggestion")
st.write("Enter the details of a product to get an AI-powered price estimate.")

# Build the user interface form
with st.form("listing_form"):
    name = st.text_input("Product Name", "Nike running shoes")
    category_name = st.text_input("Category (e.g., Women/Shoes/Athletic)", "Women/Shoes/Athletic")
    brand_name = st.text_input("Brand", "Nike")
    
    col1, col2 = st.columns(2)
    with col1:
        item_condition_id = st.selectbox("Condition (1=Mint, 5=Poor)", [1, 2, 3, 4, 5], index=1)
    with col2:
        shipping_text = st.selectbox("Shipping", ["Buyer Pays", "Seller Pays (Free Shipping)"])
        shipping = 1 if "Seller Pays" in shipping_text else 0
        
    item_description = st.text_area("Description", "Lightly used running shoes in good condition")
    
    submitted = st.form_submit_button("Predict Price")

# Execute prediction when the user clicks the button
if submitted:
    # Package inputs into a dictionary matching your sample_listing format
    input_data = {
        "name": name,
        "item_condition_id": item_condition_id,
        "category_name": category_name,
        "brand_name": brand_name,
        "shipping": shipping,
        "item_description": item_description
    }
    
    try:
        predicted_price = predictor.predict(input_data)[0]
        st.metric(label="Suggested Listing Price", value=f"${predicted_price:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")