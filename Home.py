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

movies_df = pd.read_csv('preprocessed_movies_data_tmdb.csv')

st.set_page_config(
    page_title = 'MoviesX'
)

st.title("Movies Mania")
st.divider()

genres_ls = []
def get_genres_list(string):

    ls = string.split()
    genres_ls.append(ls)
    return ls
    
 
selected =  option_menu(
        
        menu_title =  "Getting Started!",
        options=["For starters", "Instructions for Use of features",  "Contribute!"],
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
    pass  


if selected == "Contribute":
    pass