import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# Assuming the model file is named 'delivery_delay_model.sav' and is in the /content/ directory
# If you moved it, please update the path accordingly.
try:
    model = joblib.load('/content/delivery_delay_model.sav')
except FileNotFoundError:
    st.error("Error: 'delivery_delay_model.sav' not found. Please ensure the model file is uploaded to the /content/ directory.")
    st.stop()

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Define the feature names in the correct order as per x.columns
# These were: ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
# 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
# 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
# 'Warehouse_Processing_Time']

# Create input widgets for each feature
# Using reasonable min/max/default values, adjust as needed based on your data distribution
delivery_distance = st.slider('Delivery Distance (km)', min_value=0.0, max_value=100.0, value=20.0, step=0.1)
traffic_congestion = st.slider('Traffic Congestion (1=Low, 5=High)', min_value=1, max_value=5, value=3)
weather_condition = st.slider('Weather Condition (1=Good, 5=Bad)', min_value=1, max_value=5, value=3)
delivery_slot = st.slider('Delivery Slot (1=Early, 2=Mid, 3=Late)', min_value=1, max_value=3, value=2)
driver_experience = st.slider('Driver Experience (Years)', min_value=0, max_value=30, value=10)
num_stops = st.slider('Number of Stops', min_value=1, max_value=10, value=5)
vehicle_age = st.slider('Vehicle Age (Years)', min_value=0, max_value=15, value=5)
road_condition_score = st.slider('Road Condition Score (1=Poor, 5=Excellent)', min_value=1, max_value=5, value=3)
package_weight = st.slider('Package Weight (kg)', min_value=0.0, max_value=50.0, value=10.0, step=0.1)
fuel_efficiency = st.slider('Fuel Efficiency (km/L)', min_value=5.0, max_value=25.0, value=15.0, step=0.1)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', min_value=0, max_value=120, value=60)

# Create a DataFrame from the input values, ensuring correct column order
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'Predicted: **Delivery Delay** (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.success(f'Predicted: **No Delivery Delay** (Probability: {prediction_proba[0][0]:.2f})')

st.markdown("""
--- 
**Note:** To run this app, save it as `streamlit_app.py` in your Colab environment.
Then, install Streamlit: `!pip install streamlit -qq`
And run it using: `!streamlit run streamlit_app.py &>/dev/null&`
You might also need to install `localtunnel` (`npm install -g localtunnel`) and run `lt --port 8501` to get a public URL.
""")
