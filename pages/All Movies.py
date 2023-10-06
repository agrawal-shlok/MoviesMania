import streamlit as st
import time
import pandas  as pd
import pickle
import nltk
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from streamlit_option_menu import option_menu

movies_df = pd.read_csv('preprocessed_movies_data_tmdb.csv')
movies_full_df = pd.read_csv("tmdb_full_data.csv")

# movies_full_df = movies_full_df[['original_title', 'genres', 'release_date', 'vote_average']]

st.set_page_config(
    page_title = 'All Movies'
)

st.title("All Movies")
st.divider()

genres_ls = []
def get_genres_list(ls):

    # ls = string.split()
    genres_ls.append(ls)
    return ls
    
#Getting the index of the movie enetred by the user
def idx_of_movie_in_dataframe(string):
    idx = movies_full_df[movies_full_df['original_title'] == string].index.values
    idx = idx.tolist()[0]
    return idx

#Get movie id
def get_movie_id(idx):
    index = movies_full_df.iloc[idx, 0]
    return index

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
    

movie_names_list = list(movies_full_df.iloc[:, 9])
movie_id_for_image = []

for i in movie_names_list:
        idx = idx_of_movie_in_dataframe(i)
        movie_id_for_image.append(get_movie_id(idx))
       
# col1, col2, col3 = st.columns(3)
selected =  option_menu(
        
        menu_title =  "Movies Collection",
        options=["Show all movies", "Choose specifc movies"],
        orientation='horizontal',
        # menu_icon='cast',
        default_index=0,
 )
st.divider()
if selected == "Show all movies":
    if st.button("Show"):
        k=0
        for i in range(961):
            columns = st.columns(5)
            for j in range(len(columns)):
                            with columns[j]:
                                homepage_url = 'https://www.themoviedb.org/movie/{}'.format(movie_id_for_image[k+j])
                                # print(recommendations[i])
                                
                    
                                st.image(fetch_image(movie_id_for_image[k+j]  ))
                                st.write(movie_names_list[k+j])
                                # url = "https://www.streamlit.io"
                                st.write("[Explore](%s)" % homepage_url)
            k += 6
if selected == "Choose specifc movies":
        # movies_full_df['genres'] = movies_full_df['genres'].apply(get_genres_list)
        final_genres_ls =list(set(movies_full_df['genres']))
        # final_genres_ls = " ".join(final_genres_ls)
        # print(final_genres_ls)
        selected_genres = st.selectbox("Select genres:", options=final_genres_ls)
        st.divider()
        
        st.markdown("<h3 style='text-align: center; color: white;'>Select your preferred movie rating!</h3>", unsafe_allow_html=True)
        movies_full_df['vote_average'] = movies_full_df['vote_average'].astype('str')
        movies_full_df.loc[(movies_full_df['vote_average'] >= str(1)) & (movies_full_df['vote_average'] < str(2)), 'vote_average'] = '1-2'
        movies_full_df.loc[(movies_full_df['vote_average'] >= str(2)) & (movies_full_df['vote_average'] < str(3)), 'vote_average'] = '2-3'
        movies_full_df.loc[(movies_full_df['vote_average'] >= str(3)) & (movies_full_df['vote_average'] < str(4)), 'vote_average'] = '3-4'
        movies_full_df.loc[(movies_full_df['vote_average'] >= str(4)) & (movies_full_df['vote_average'] < str(5)), 'vote_average'] = '4-5'
        movies_full_df.loc[(movies_full_df['vote_average'] >= str(5)) & (movies_full_df['vote_average'] < str(6)), 'vote_average'] = '5-6'
        movies_full_df.loc[(movies_full_df['vote_average'] >= str(6)) & (movies_full_df['vote_average'] < str(7)), 'vote_average'] = '6-7'
        movies_full_df.loc[(movies_full_df['vote_average'] >= str(7)) & (movies_full_df['vote_average'] < str(8)), 'vote_average'] = '7-8'
        movies_full_df.loc[(movies_full_df['vote_average'] >= str(8)) & (movies_full_df['vote_average'] < str(9)), 'vote_average'] = '8-9'
        movies_full_df.loc[(movies_full_df['vote_average'] >= str(9)) & (movies_full_df['vote_average'] < str(10)), 'vote_average'] = '9-10'
        
        
        movie_rating_list = sorted(list(set(list(movies_full_df.iloc[:, 21]))))
        selected_rating = st.selectbox("Here", options=movie_rating_list)
        st.divider()
        
        movies_release_date_ls =sorted(list(set(movies_full_df.iloc[:, 14])))
        st.markdown("<h3 style='text-align: center; color: white;'>Select your preferred release date too!</h3>", unsafe_allow_html=True)
        # movie_rating_list = list(set(list(movies_full_df.iloc[:, 21])))
        selected_release_date = st.selectbox("Here", options=movies_release_date_ls)
        st.divider()
        
        
        selected_movies_ls =[]
        # for index in movies_full_df.index:
            # if movies_full_df['genres'][index] == selected_genres:
                # if movies_full_df['vote_average'][index] == selected_rating:
                    # if movies_full_df['release_date'][index] == selected_release_date:
                            # selected_movies_ls.append(movies_full_df[index])
                            
        # print(type(selected_genres), type(selected_rating), type(selected_release_date))
        selected_movies_ls = movies_full_df[(movies_full_df['genres']== selected_genres) & (movies_full_df['vote_average'] == selected_rating)]                    
        print(selected_movies_ls)
        
        
# with col1:
    # for i in movie_names_list:
        # st.write(i)
        # pass
# with col2:
    # for i in range(len(movie_names_list)):
       # st.write(movie_names_list[i])
       # st.image(fetch_image(movie_id_for_image[i]))
       
# with col3:
       # homepage_url = 'https://www.themoviedb.org/movie/{}'.format(movie_id_for_image[i])
       # st.write("[Explore](%s)" % homepage_url)

    #    vfhgucftvguhcfvt
    