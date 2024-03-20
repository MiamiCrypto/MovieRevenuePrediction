import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the dataset
model = joblib.load('random_forest_model.joblib')
movies_data = pd.read_csv('path_to_your/movies.csv')

st.title('Movie Gross Revenue Prediction')

# Dropdown to select the movie
selected_movie = st.selectbox('Choose a movie', movies_data['name'].unique())

# When a movie is selected, we will pre-fill budget and score
if selected_movie:
    movie_info = movies_data[movies_data['name'] == selected_movie].iloc[0]
    budget = st.number_input('Movie Budget', value=int(movie_info['budget']))
    score = st.number_input('Movie Score', value=float(movie_info['score']))

    # Predict button
    if st.button('Predict Gross Revenue'):
        prediction = model.predict([[budget, score]])
        # Preventing negative predictions
        prediction = max(0, prediction[0])
        st.success(f'Predicted Gross Revenue: ${prediction:,.2f}')

        # Display actual revenue if available
        if pd.notnull(movie_info['gross']):
            st.info(f'Actual Gross Revenue: ${movie_info["gross"]:,.2f}')
