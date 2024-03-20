import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
movies_data = pd.read_csv('cleaned_movies.csv')

st.title('Movie Gross Revenue Prediction')

# Display an image from a file
st.image(Aiface3.png', caption='Predict your favorite movie's budget')

# Dropdown to select the movie
selected_movie = st.selectbox('Choose a movie', movies_data['name'].unique())

# When a movie is selected, pre-fill the budget and score
if selected_movie:
    movie_info = movies_data[movies_data['name'] == selected_movie].iloc[0]
    budget = st.number_input('Movie Budget', value=int(movie_info['budget']))
    score = st.number_input('Movie Score', value=float(movie_info['score']))

    # Predict button
    if st.button('Predict Gross Revenue'):
        prediction = model.predict([[budget, score]])
        prediction = max(0, prediction[0])  # Ensure the prediction is not negative
        st.success(f'Predicted Gross Revenue: ${prediction:,.2f}')

        # Display actual revenue if available
        if pd.notnull(movie_info['gross']):
            st.info(f'Actual Gross Revenue: ${movie_info["gross"]:,.2f}')
