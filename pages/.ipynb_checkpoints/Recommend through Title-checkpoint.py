import streamlit as st
import time
import pandas  as pd
import pickle
import nltk
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests


st.set_page_config(
    page_title = 'Recommend through Title'
)


st.sidebar.success('Select a Feature from above')

#Downloading some dependencies
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

movies_df = pd.read_csv('preprocessed_movies_data_tmdb.csv')

# https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl



#Getting the index of the movie enetred by the user
def idx_of_movie_in_dataframe(string):
    idx = movies_df[movies_df['original_title'] == string].index.values
    idx = idx.tolist()[0]
    return idx

#Get movie id
def get_movie_id(idx):
    index = movies_df.iloc[idx, 0]
    return index



#Loading the vectos.pkl file

file_name = 'vectors.pkl'
open_file = open(file_name, "rb")
final_embeddings = pickle.load(open_file)
open_file.close()


#Getting the top five movies from the inputted movie

def recommend_movie(index):
    similarity = []
    similarity = cosine_similarity(final_embeddings[index].reshape(1, -1), final_embeddings)[0]
    movie_indexes = list(enumerate(similarity))
    movie_indexes.sort(key = lambda x: x[1], reverse=True)
    movie_indexes = movie_indexes[1:11]

    movie_names = []
    for i in movie_indexes:
        # print(movies.iloc[i[0], 1])
        movie_names.append(movies_df.iloc[i[0], 4])
        # print(movie)
    return movie_names


#Fetching the image of the movie
def fetch_image(movie_id):

    url = "https://api.themoviedb.org/3/movie/{}?language=en-US".format(movie_id)

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3ZjFlYTM2YTFjMmU0ZWM2OWI0NmFhYTg3YjVkMjRkNiIsInN1YiI6IjY0Y2UzNzFjNGQ2NzkxMDEzOWVkZjYyYiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Um8Newx2IqVQ_7vZMs4hBbZCqoz9uMuNgOX-7ERNPl4"
}

    response = requests.get(url, headers=headers)
    data = response.json()
    # print(data['poster_path'])
    image_url = 'https://image.tmdb.org/t/p/w780/' + data['poster_path']
    # print(image_url)
    return image_url
    
    
#Giving our web app a title
st.title("Movies Recommender System")


#Giving movie options to the user

movie_names_list = list(movies_df.iloc[:, 4])

name = st.selectbox("Choose the movie for which you want recommendations from below", options=movie_names_list, key='1')

# print(movies_df)

#Button to how recommendations

if st.button('Show Top 10 Similar Movies'):

    idx = idx_of_movie_in_dataframe(name)
  
    recommendations = recommend_movie(idx)

    movie_id_for_image = []
    for i in recommendations:
        idx = idx_of_movie_in_dataframe(i)
        movie_id_for_image.append(get_movie_id(idx))
        
    print(recommendations, movie_id_for_image)
    columns = st.columns(5)
    for i in range(len(columns)):
        with columns[i]:
            homepage_url = 'https://www.themoviedb.org/movie/{}'.format(movie_id_for_image[i])

            st.write(recommendations[i])
            st.image(fetch_image(movie_id_for_image[i]))
            # url = "https://www.streamlit.io"
            st.write("[Explore](%s)" % homepage_url)

    with st.expander('Next Five'):
        columns = st.columns(5)
        for i in range(len(columns)):
            with columns[i]:
                    
                homepage_url = 'https://www.themoviedb.org/movie/{}'.format(movie_id_for_image[i+5])

                st.write(recommendations[i+5])
                st.image(fetch_image(movie_id_for_image[i+5]))
                # url = "https://www.streamlit.io"
                st.write("[Explore](%s)" % homepage_url)
     
            
