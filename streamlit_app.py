import streamlit as st
import joblib
import pandas as pd

# Load the trained Linear Regression model and the cleaned dataset
model = joblib.load("linear_regression_model.joblib")
data = pd.read_csv("cleaned_movies.csv")

st.title('Movie Gross Revenue Prediction')

# Movie name input
movie_name = st.selectbox('Choose a movie', data['name'].unique())

# When a movie is selected, pre-fill the budget and score fields
if movie_name:
    selected_movie = data[data['name'] == movie_name].iloc[0]
    budget = st.number_input('Movie Budget', value=int(selected_movie['budget']))
    score = st.number_input('Movie Score', value=float(selected_movie['score']))

# Predict button
if st.button('Predict Gross Revenue'):
    # Prediction
    prediction = model.predict([[budget, score]])
    actual = selected_movie['gross'] if not pd.isnull(selected_movie['gross']) else "Unknown"
    st.write(f'Predicted Gross Revenue: ${prediction[0]:,.2f}')
    st.write(f'Actual Gross Revenue: ${actual:,.2f}' if actual != "Unknown" else "Actual Gross Revenue: Unknown")
