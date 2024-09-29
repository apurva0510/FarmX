# app.py
import streamlit as st
from PIL import Image
import webbrowser
import functions
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="FarmX",
    page_icon="assets/favicon.png",  # Ensure favicon.png exists
    layout="wide",
    initial_sidebar_state="auto"
)

# Load model function
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load all models and scalers at once for efficiency
@st.cache_resource
def load_scalers_and_models():
    scaler_n = load_model('models/scaler_n_category.pkl')
    gb_classifier = load_model('models/gb_n_category_classifier.pkl')
    scaler_K = load_model('models/scaler_K_category.pkl')
    rf_K_classifier = load_model('models/rf_K_category_classifier.pkl')
    scaler_P = load_model('models/scaler_P_category.pkl')
    rf_P_classifier = load_model('models/rf_P_category_classifier.pkl')
    scaler_N = load_model('models/scaler_N.pkl')
    rf_N_regressor = load_model('models/rf_N_regressor.pkl')
    return scaler_n, gb_classifier, scaler_K, rf_K_classifier, scaler_P, rf_P_classifier, scaler_N, rf_N_regressor

scalers_and_models = load_scalers_and_models()

# Define crop IDs for prediction
crop_ids = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
    'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
    'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
    'jute': 21, 'coffee': 22
}

# Application Title
st.title("FarmX")

# Create Tabs for Navigation
tabs = st.tabs(["Home", "Projected Yield", "Soil Nitrogen", "Resources", "FAQ"])

st.sidebar.title("Customize Your Experience")
theme = st.sidebar.selectbox("Choose Theme", ["Default", "Green"])

#Set theme based on user selection
if theme == "Green":
    st.markdown(
        """
        <style>
        body {
            background-color: #18453B;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: white;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


##########################
# Home Tab
##########################
with tabs[0]:
    # Display Logo
    logo = Image.open("assets/logo.png")
    st.image(logo, use_column_width=True)

    # Introductory Text
    st.markdown(
        "<h2 style='text-align: center;'>Predicts how you can enrich your precious soil, while informing you of its potential bounty.</h2>",
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

    # Select Comparison Type with unique key
    comparison_type = st.radio(
        "Choose Comparison Type",
        ("Single Crop Prediction", "Compare Two Crops"),
        key="comparison_type_yield"  # Unique key assigned
    )

    if comparison_type == "Single Crop Prediction":
        # Input Fields for single yield prediction
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
                    try:
                        yield_prediction = crop(grain_weight, grain_moisture, harvested_area, crop_selection)
                        st.success(f"**Projected Yield for {crop_selection}:** {yield_prediction:.2f} kg")
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")
                else:
                    st.error("Invalid crop selection.")
            else:
                st.error("Please fill in all fields.")

    elif comparison_type == "Compare Two Crops":
        # Input Fields for comparison
        with st.form(key='comparison_form'):
            # First Crop Selection and Inputs
            st.markdown("### Crop 1")
            col1, col2 = st.columns(2)
            with col1:
                crop1 = st.selectbox("Select First Crop", options=list(crop_ids.keys()), key="crop1")
            with col2:
                crid1 = crop_ids.get(crop1, None)

            grain_weight1 = st.number_input("Grain Weight (kg) - Crop 1", min_value=0.0, step=0.1, format="%.2f", key="grain_weight1")
            grain_moisture1 = st.number_input("Grain Moisture (%) - Crop 1", min_value=0.0, max_value=100.0, step=0.1, format="%.2f", key="grain_moisture1")
            harvested_area1 = st.number_input("Harvested Area (ha) - Crop 1", min_value=0.0, step=0.01, format="%.2f", key="harvested_area1")

            # Second Crop Selection and Inputs
            st.markdown("### Crop 2")
            col3, col4 = st.columns(2)
            with col3:
                crop2 = st.selectbox("Select Second Crop", options=list(crop_ids.keys()), key="crop2")
            with col4:
                crid2 = crop_ids.get(crop2, None)

            grain_weight2 = st.number_input("Grain Weight (kg) - Crop 2", min_value=0.0, step=0.1, format="%.2f", key="grain_weight2")
            grain_moisture2 = st.number_input("Grain Moisture (%) - Crop 2", min_value=0.0, max_value=100.0, step=0.1, format="%.2f", key="grain_moisture2")
            harvested_area2 = st.number_input("Harvested Area (ha) - Crop 2", min_value=0.0, step=0.01, format="%.2f", key="harvested_area2")

            submit_button = st.form_submit_button(label='Compare Yields')

        if submit_button:
            if all([grain_weight1, grain_moisture1, harvested_area1, crid1,
                    grain_weight2, grain_moisture2, harvested_area2, crid2]):
                try:
                    yield1 = crop(grain_weight1, grain_moisture1, harvested_area1, crop1)
                    yield2 = crop(grain_weight2, grain_moisture2, harvested_area2, crop2)

                    # Display side-by-side
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"### **{crop1}**")
                        st.write(f"**Grain Weight:** {grain_weight1} kg")
                        st.write(f"**Grain Moisture:** {grain_moisture1}%")
                        st.write(f"**Harvested Area:** {harvested_area1} ha")
                        st.write(f"**Projected Yield:** {yield1:.2f} kg")
                    with col_b:
                        st.markdown(f"### **{crop2}**")
                        st.write(f"**Grain Weight:** {grain_weight2} kg")
                        st.write(f"**Grain Moisture:** {grain_moisture2}%")
                        st.write(f"**Harvested Area:** {harvested_area2} ha")
                        st.write(f"**Projected Yield:** {yield2:.2f} kg")

                    # Optional: Visualization
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots()
                    crops = [crop1, crop2]
                    yields = [yield1, yield2]
                    ax.bar(crops, yields, color=['blue', 'green'])
                    ax.set_xlabel('Crop')
                    ax.set_ylabel('Projected Yield (kg)')
                    ax.set_title('Yield Comparison')
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error in prediction: {e}")
            else:
                st.error("Please fill in all fields for both crops.")

##########################
# Soil Nitrogen Tab
##########################
with tabs[2]:
    st.header("Predicting Nitrogen Values")

    # Select Comparison Type with unique key
    comparison_type_n = st.radio(
        "Choose Comparison Type",
        ("Single Crop Prediction", "Compare Two Crops"),
        key="comparison_type_nitrogen"  # Unique key assigned
    )

    if comparison_type_n == "Single Crop Prediction":
        # Input Fields for single nitrogen prediction
        temp = st.number_input("Temperature (°C)", min_value=-100.0, max_value=100.0, step=0.1, format="%.2f", key="temp_single_n")
        hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f", key="hum_single_n")
        pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, format="%.2f", key="ph_single_n")
        rain = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1, format="%.2f", key="rain_single_n")

        # Crop Selection
        crop_selection_npk = st.selectbox("Crop", options=list(crop_ids.keys()), key="crop_npk_single_n")

        # Predict Button
        if st.button("Predict Optimal Nitrogen"):
            if all([temp, hum, pH, rain, crop_selection_npk]):
                crid = crop_ids.get(crop_selection_npk, None)
                if crid:
                    try:
                        # Predict categories
                        N_category = functions.predict_n_category(
                            hum, temp, rain, pH, crid
                        )
                        K_category = functions.predict_K_category(
                            hum, temp, rain, pH, crid
                        )
                        P_category = functions.predict_P_category(
                            hum, temp, rain, pH, crid
                        )

                        # Predict N value
                        N_value = functions.predict_N(
                            hum, temp, rain, pH, crid
                        )

                        result_text = (
                            f"### **Optimal Nitrogen Category for {crop_selection_npk}:** {N_category}\n\n"
                            f"**Predicted Nitrogen Value:** {N_value:.2f} kg/ha\n\n"
                            f"**Predicted K Category:** {K_category}\n"
                            f"**Predicted P Category:** {P_category}"
                        )
                        st.success(result_text)
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")
                else:
                    st.error("Invalid crop selection.")
            else:
                st.error("Please fill in all fields.")

    elif comparison_type_n == "Compare Two Crops":
        # Input Fields for comparison
        with st.form(key='comparison_form_nitrogen'):
            # First Crop Selection and Inputs
            st.markdown("### Crop 1")
            col1_n, col2_n = st.columns(2)
            with col1_n:
                crop1_n = st.selectbox("Select First Crop", options=list(crop_ids.keys()), key="crop1_nitrogen")
            with col2_n:
                crid1_n = crop_ids.get(crop1_n, None)

            temp1_n = st.number_input("Temperature (°C) - Crop 1", min_value=-100.0, max_value=100.0, step=0.1, format="%.2f", key="temp1_nitrogen")
            hum1_n = st.number_input("Humidity (%) - Crop 1", min_value=0.0, max_value=100.0, step=0.1, format="%.2f", key="hum1_nitrogen")
            pH1_n = st.number_input("Soil pH - Crop 1", min_value=0.0, max_value=14.0, step=0.1, format="%.2f", key="ph1_nitrogen")
            rain1_n = st.number_input("Rainfall (mm) - Crop 1", min_value=0.0, step=0.1, format="%.2f", key="rain1_nitrogen")

            # Second Crop Selection and Inputs
            st.markdown("### Crop 2")
            col3_n, col4_n = st.columns(2)
            with col3_n:
                crop2_n = st.selectbox("Select Second Crop", options=list(crop_ids.keys()), key="crop2_nitrogen")
            with col4_n:
                crid2_n = crop_ids.get(crop2_n, None)

            temp2_n = st.number_input("Temperature (°C) - Crop 2", min_value=-100.0, max_value=100.0, step=0.1, format="%.2f", key="temp2_nitrogen")
            hum2_n = st.number_input("Humidity (%) - Crop 2", min_value=0.0, max_value=100.0, step=0.1, format="%.2f", key="hum2_nitrogen")
            pH2_n = st.number_input("Soil pH - Crop 2", min_value=0.0, max_value=14.0, step=0.1, format="%.2f", key="ph2_nitrogen")
            rain2_n = st.number_input("Rainfall (mm) - Crop 2", min_value=0.0, step=0.1, format="%.2f", key="rain2_nitrogen")

            submit_button_n = st.form_submit_button(label='Compare Nitrogen Predictions')

        if submit_button_n:
            if all([temp1_n, hum1_n, pH1_n, rain1_n, crid1_n,
                    temp2_n, hum2_n, pH2_n, rain2_n, crid2_n]):
                try:
                    # Crop 1 Predictions
                    N_category1 = functions.predict_n_category(
                        hum1_n, temp1_n, rain1_n, pH1_n, crid1_n
                    )
                    K_category1 = functions.predict_K_category(
                        hum1_n, temp1_n, rain1_n, pH1_n, crid1_n
                    )
                    P_category1 = functions.predict_P_category(
                        hum1_n, temp1_n, rain1_n, pH1_n, crid1_n
                    )
                    N_value1 = functions.predict_N(
                        hum1_n, temp1_n, rain1_n, pH1_n, crid1_n
                    )

                    # Crop 2 Predictions
                    N_category2 = functions.predict_n_category(
                        hum2_n, temp2_n, rain2_n, pH2_n, crid2_n
                    )
                    K_category2 = functions.predict_K_category(
                        hum2_n, temp2_n, rain2_n, pH2_n, crid2_n
                    )
                    P_category2 = functions.predict_P_category(
                        hum2_n, temp2_n, rain2_n, pH2_n, crid2_n
                    )
                    N_value2 = functions.predict_N(
                        hum2_n, temp2_n, rain2_n, pH2_n, crid2_n
                    )

                    # Display side-by-side
                    col_a_n, col_b_n = st.columns(2)
                    with col_a_n:
                        st.markdown(f"### **{crop1_n}**")
                        st.write(f"**Temperature:** {temp1_n}°C")
                        st.write(f"**Humidity:** {hum1_n}%")
                        st.write(f"**Soil pH:** {pH1_n}")
                        st.write(f"**Rainfall:** {rain1_n} mm")
                        st.write(f"**Optimal Nitrogen Category:** {N_category1}")
                        st.write(f"**Predicted Nitrogen Value:** {N_value1:.2f} kg/ha")
                        st.write(f"**Predicted K Category:** {K_category1}")
                        st.write(f"**Predicted P Category:** {P_category1}")
                    with col_b_n:
                        st.markdown(f"### **{crop2_n}**")
                        st.write(f"**Temperature:** {temp2_n}°C")
                        st.write(f"**Humidity:** {hum2_n}%")
                        st.write(f"**Soil pH:** {pH2_n}")
                        st.write(f"**Rainfall:** {rain2_n} mm")
                        st.write(f"**Optimal Nitrogen Category:** {N_category2}")
                        st.write(f"**Predicted Nitrogen Value:** {N_value2:.2f} kg/ha")
                        st.write(f"**Predicted K Category:** {K_category2}")
                        st.write(f"**Predicted P Category:** {P_category2}")

                    # Visualization: Nitrogen Comparison Bar Chart
                    fig, ax = plt.subplots()
                    crops_n = [crop1_n, crop2_n]
                    nitrogen_values = [N_value1, N_value2]
                    colors_n = ['#FF9800', '#9C27B0']  # Orange and Purple for distinction

                    ax.bar(crops_n, nitrogen_values, color=colors_n)
                    ax.set_xlabel('Crop')
                    ax.set_ylabel('Predicted Nitrogen Value (kg/ha)')
                    ax.set_title('Nitrogen Prediction Comparison')
                    ax.set_ylim(0, max(nitrogen_values) * 1.2)  # Add some space on top for labels

                    # Adding labels on top of bars
                    for i, v in enumerate(nitrogen_values):
                        ax.text(i, v + (max(nitrogen_values) * 0.05), f"{v:.2f} kg/ha", ha='center', fontweight='bold')

                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error in prediction: {e}")
            else:
                st.error("Please fill in all fields for both crops.")

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

##########################
# FAQ Tab
##########################
with tabs[4]:
    st.header("Frequently Asked Questions")

    with st.expander("How to interpret soil nutrient levels?"):
        st.write("""
            Soil nutrient levels indicate the availability of essential nutrients like Nitrogen (N), Phosphorus (P), and Potassium (K). Optimal levels ensure healthy crop growth.
        """)

    with st.expander("When should I apply fertilizers?"):
        st.write("""
            Fertilizers should be applied based on soil test results and crop requirements. Typically, application is done during planting and mid-growth stages.
        """)

    with st.expander("How to use FarmX for best results?"):
        st.write("""
            Regularly input your soil test data, monitor nutrient levels, and follow the fertilizer recommendations provided by FarmX to maintain optimal soil health.
        """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2024 FarmX. All rights reserved.</p>", unsafe_allow_html=True)