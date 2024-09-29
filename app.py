# app.py
import streamlit as st
from PIL import Image
import webbrowser
import functions
import pandas as pd
import pickle 
from cost import crop

# Load model function
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Update your model loading calls
scaler_n = load_model('models/scaler_n_category.pkl')
scaler_K = load_model('models/scaler_K_category.pkl')
scaler_P = load_model('models/scaler_P_category.pkl')
scaler_N = load_model('models/scaler_N.pkl')

gb_classifier = load_model('models/gb_n_category_classifier.pkl')  # Load N category classifier
rf_K_classifier = load_model('models/rf_K_category_classifier.pkl')  # Load K category classifier
rf_N_regressor = load_model('models/rf_N_regressor.pkl')  # Load N regressor
rf_P_classifier = load_model('models/rf_P_category_classifier.pkl')  # Load P category classifier

# Define crop IDs for prediction
crop_ids = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
    'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
    'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
    'jute': 21, 'coffee': 22
}

# Set page configuration
st.set_page_config(
    page_title="fertiCulture",
    page_icon="assets/favicon.png",  # Ensure favicon.png exists
    layout="wide",
    initial_sidebar_state="auto"
)

# Application Title
st.title("FarmX")

# Create Tabs for Navigation
tabs = st.tabs(["Home", "Projected Yield", "Soil Nitrogen", "Resources"])

##########################
# Home Tab
##########################
with tabs[0]:
    # Display Logo
    logo = Image.open("assets/logo.png")
    st.image(logo, use_column_width=True)

    # Introductory Text
    st.markdown(
        "<h2 style='text-align: center;'>FarmX predicts by how much you can enrich your precious soil, while informing you of its potential bounty.</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center;'>A farmer's dream, manifest.</h3>",
        unsafe_allow_html=True
    )

##########################
# Projected Yield Tab
##########################
with tabs[1]:
    st.header("Projected Yield")

    # Input Fields for yield prediction
    grain_weight = st.number_input("Grain Weight (kg)", min_value=0.0, step=0.1, format="%.2f")
    grain_moisture = st.number_input("Grain Moisture (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f")
    harvested_area = st.number_input("Harvested Area (ha)", min_value=0.0, step=0.01, format="%.2f")
    
    # Crop Selection for yield prediction
    crop_selection = st.selectbox("Crop", options=list(crop_ids.keys()), key="crop_yield")

    # Predict Button
    if st.button("Predict Yield"):
        if all([grain_weight, grain_moisture, harvested_area, crop_selection]):
            crid = crop_ids.get(crop_selection, None)
            if crid:
                yield_prediction = crop(grain_weight, grain_moisture, harvested_area, crop_selection)
                st.success(f"**Projected Yield for {crop_selection}:** {yield_prediction} kg")
            else:
                st.error("Invalid crop selection.")
        else:
            st.error("Please fill in all fields.")

##########################
# Soil Nitrogen Tab
##########################
with tabs[2]:
    st.header("Predicting Nitrogen Values")

    # Input Fields
    temp = st.number_input("Temperature (°C)", min_value=-100.0, max_value=100.0, step=0.1, format="%.2f", key="temp_input")
    hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f", key="hum_input")
    pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, format="%.2f", key="ph_input")
    rain = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1, format="%.2f", key="rain_input")

    # Crop Selection
    crop_selection_npk = st.selectbox("Crop", options=list(crop_ids.keys()), key="crop_npk")

    # Predict Button
    if st.button("Predict Optimal Nitrogen"):
        if all([temp, hum, pH, rain, crop_selection_npk]):
            crid = crop_ids.get(crop_selection_npk, None)
            if crid:
                N = functions.predict_n_category(hum, temp, rain, pH, crid)
                K = functions.predict_K_category(hum, temp, rain, pH, crid)
                P = functions.predict_P_category(hum, temp, rain, pH, crid)
                Nyield_value = functions.predict_N(hum, temp, rain, pH, crid)
                
                result_text = (
                    f"**Optimal Nitrogen content for {crop_selection_npk}:** {N}\n\n"
                    f"**Recommended for this soil:** {Nyield_value}\n\n"
                    f"**Predicted K Category:** {K}\n\n"
                    f"**Predicted P Category:** {P}\n\n"
                )
                st.success(result_text)
            else:
                st.error("Invalid crop selection.")
        else:
            st.error("Please fill in all fields.")

##########################
# Resources Tab
##########################
with tabs[3]:
    st.header("Resources")

    # Define resource links
    resources = {
        "Soil Testing - MSU": "https://homesoiltest.msu.edu/get-started",
        "NPK Fertilizer Calculator": "https://aesl.ces.uga.edu/soil/fertcalc/",
        "United States Department of Agriculture": "https://www.usda.gov/",
        "Minority and Women Farmers and Ranchers": "https://www.fsa.usda.gov/programs-and-services/farm-loan-programs/minority-and-women-farmers-and-ranchers/index",
        "Soil Health Institute": "https://soilhealthinstitute.org/",
        "How much is too much for the climate?": "https://msutoday.msu.edu/news/2014/how-much-fertilizer-is-too-much-for-the-climate"
    }

    # Display buttons for each resource
    for name, link in resources.items():
        if st.button(name):
            webbrowser.open_new_tab(link)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2024 FarmX. All rights reserved.</p>", unsafe_allow_html=True)
