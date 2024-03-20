import streamlit as st
import joblib

# Load the trained Linear Regression model
model = joblib.load("linear_regression_model.joblib")

st.title('Movie Gross Revenue Prediction')

# User inputs for budget and score
budget = st.number_input('Movie Budget', min_value=0)
score = st.number_input('Movie Score', min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button('Predict Gross Revenue'):
    # Prediction
    prediction = model.predict([[budget, score]])
    st.write(f'Predicted Gross Revenue: ${prediction[0]:,.2f}')
