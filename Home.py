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
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card

movies_df = pd.read_csv('preprocessed_movies_data_tmdb.csv')

movies_full_df = pd.read_csv("tmdb_full_data.csv")

# movies_full_df = movies_full_df[['original_title', 'genres', 'release_date', 'vote_average']]

def process():

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

    while st.button("Show"):
            k=0
            for i in range(1):
                columns = st.columns(5)
                for j in range(len(columns)):
                                with columns[j]:
                                    homepage_url = 'https://www.themoviedb.org/movie/{}'.format(movie_id_for_image[k+j])
                                    # print(recommendations[i])
                                    st.image(fetch_image(movie_id_for_image[k+j]  ))
                                    st.write(movie_names_list[k+j])
                                    # url = "https://www.streamlit.io"
                                    st.write("[Explore](%s)" % homepage_url)
                                    
                                    hasClicked = card(
                                          title=st.write(movie_names_list[k+j]),
                                          text="Movie",
                                          image=fetch_image(movie_id_for_image[k+j]  ),
                                          url=homepage_url
                                        )
 
                k += 6


st.set_page_config(
    page_title = 'MoviesMania'
)

st.markdown("<h1 style='text-align: center; color: white;'>Movies Mania</h1>", unsafe_allow_html=True)
st.divider()

genres_ls = []
def get_genres_list(string):

    ls = string.split()
    genres_ls.append(ls)
    return ls
    
 
selected =  option_menu(
        
        menu_title =  "Getting Started!",
        options=["Welcome", "For starters", "Instructions for Use of features"],
        orientation='horizontal',
        # menu_icon='cast',
        default_index=0,
 )
st.divider()
if selected == "For starters":
    
    col1, col2 = st.columns([6,2])
    with col1:
        # pass
        st.markdown("<h3 style='text-align: center; color: white;'>Take a look through our movies collection!</h3>", unsafe_allow_html=True)
    with col2:
        
        
        # movies_df['genres'] = movies_df['genres'].apply(get_genres_list)
        # final_genres_ls =list(set(genres_ls))
        # selected_genres = st.multiselect("Select genres:", options=final_genres_ls)
        # st.divider()
        
        # st.markdown("<h2 style='text-align: center; color: white;'>Select five movies of your choice!</h2>", unsafe_allow_html=True)
        # movie_names_list = list(set(list(movies_df.iloc[:, 4])))
        # name = st.multiselect("Choose five movies", options=movie_names_list)
        
        # final_movies_ls =list(set(movies_ls))
        # if st.checkbox("Select through option menu?"):
            # selected_movies = st.multiselect("Select some movies!:", options=final_movies_ls)
            # st.divider()
        if st.button("Show all movies"):
            switch_page("All Movies")
    st.divider()      
    col3, col4 = st.columns([6,2])
    with col3:
        # pass
        st.markdown("<h3 style='text-align: center; color: white;'>Looking for specific movies?</h3>", unsafe_allow_html=True)
    with col4:
        if st.button("Choose"):
            switch_page("All Movies")
    
if selected ==  "Instructions for Use of features":
    
    st.markdown("<h3 style='text-align: center; font-weight:bold; font-style:italic; color: white;'>For 'Recommend through Videos' </h3>", unsafe_allow_html=True)
    st.divider()
    st.markdown('''
        1. Go to Upload a Video page and upload either through your local machine or Download through YT Shorts page'
        2. Then visit the Recommend through Video page and input the plot and genres either on your own or thorugh the drop-own menus'
        3. Once downloaded/uploaded, visit the Upload Here! page and select your particular option to proceed. If you want to use our special feature, please select so beofre proceeding with selecting your option.'
        4. Please wait for 8-10 min for whole process to be finished. Meanwhile you can make use of our special feature of viewing Reviews for the movie of your choice, if selected.'
        5. Voila! The Top 10 Recommended Movies are live!
    '''
    )
    
    st.markdown("<h3 style='text-align: center; font-weight:bold; font-style:italic; color: white;'>For 'Recommend through Plot' </h3>", unsafe_allow_html=True)
    st.divider()
    st.markdown('''
    
        1. Visit the 'Recommnd through Plot' page and enter your preferred plot and genre
        2. For better recommendations, try to atleast input 50-60 words worth of meaningful sentences of your plot and as specific as possible genres. You can also provide meaningful keywords.
        3. Voila! The Top 10 Recommended Movies are live!
    '''
    )
    
    st.markdown("<h3 style='text-align: center; font-weight:bold; font-style:italic; color: white;'>For 'Recommend through Title' </h3>", unsafe_allow_html=True)
    st.divider()
    st.markdown('''
    
        1. Visit the 'Recommnd through Title' page and select your preferred movie title from the dropdown menu, if available(More will be added!).
        2. Select 'Show Top 10 Similar Movies' button.
        3. Voila! The Top 10 Recommended Movies are live!
    '''
    )
    

if selected == "Welcome":
    
    process()