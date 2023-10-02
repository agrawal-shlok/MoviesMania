import streamlit as st
import time
import pandas  as pd
import pickle
import nltk
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import cv2
import tempfile
import streamlit as st
import time
import pandas  as pd
import pickle
import nltk
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import cv2  
import tempfile
from facenet_pytorch import MTCNN
from PIL import Image
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D,MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import shutil
import pickle
import gensim   
from gensim.models import FastText, Word2Vec
import itertools
import spacy
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.utils import pad_sequences


# @st.cache_resource
nlp = spacy.load('en_core_web_sm')

#Dependencies
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


embeddings_movies = pickle.load(open('embeddings_movies_final.pkl', 'rb'))
movies_df = pd.read_csv('preprocessed_movies_data_tmdb.csv')
extracted_actor_names = pickle.load(open('extracted_actor_names.pkl', 'rb'))

metadata = pd.read_csv('preprocessed_movies_data_tmdb.csv')


        
@st.cache_resource
def load_model_autocomplete():
    
    model = tf.keras.saving.load_model("Autocomplete_movies_tmdb_bidirectional_lstm_model.keras")
    
    return model


st.set_page_config(
    page_title = 'Recommend through Plot'
)


st.markdown("<h1 style='text-align: center; color: white;'>Be Creative!</h1>", unsafe_allow_html=True)
st.divider()

def making_corpus(string):
    data = string.split()
    return data

@st.cache_resource
def load_model():
    model = pickle.load(open('model_Word2Vec_movies.pkl', 'rb'))
    return model

 
def get_vectors(ls):
    # print(ls)
    word2vec_model = load_model()
    embeddings = word2vec_model.wv.get_normed_vectors()
    all_embd = np.stack(embeddings)
    embd_mean, embd_std = all_embd.mean(), all_embd.std()   
    #Building vocabulary
    vocabulary = set(word2vec_model.wv.index_to_key)
    final_embeddings = []
    for i in ls:   
        avg_embeddings = np.random.normal(embd_mean, embd_std)
        for j in i:
            
            if j in vocabulary: 
                
                if avg_embeddings is None:
                    avg_embeddings = word2vec_model.wv[j]
                else:
                    avg_embeddings = avg_embeddings + word2vec_model .wv[j]
        if avg_embeddings is not None:
            # print(avg_embeddings)
            avg_embeddings = avg_embeddings / len(avg_embeddings)
            final_embeddings.append(avg_embeddings)
    # print(len(list(unique_words)))
    return final_embeddings

#Get movie id
def get_movie_id(idx):
    index = movies_df.iloc[idx, 0]
    return index

 #Getting the index of the movie enetred by the user
def idx_of_movie_in_dataframe(string):
    idx = movies_df[movies_df['original_title'] == string].index.values
    idx = idx.tolist()[0]
    return idx

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

def get_desc(name):
    _id = idx_of_movie_in_dataframe(name)
    desc = movies_df.iloc[_id, 5]
    return desc

def get_genre(name):
    _id = idx_of_movie_in_dataframe(name)
    genre = movies_df.iloc[_id, 3]
    string = ''
    ls = genre.split()

    for i in ls:
        string += i
        string +=  ' '

    return string
    
def recommend_movie(embd):
    similarity = []
    print(embd)
    # print(embeddings_movies)
    # print(embeddings_movies)
    # for i in range(len(embeddings_movies)):
    similarity = cosine_similarity(np.array(embd).reshape(1, -1), embeddings_movies)[0]
    # similarity.append(sine[0])
    movie_indexes = list(enumerate(similarity))
    movie_indexes.sort(key = lambda x: x[1], reverse=True)
    # print(movie_indexes)
    movie_indexes = movie_indexes[1:11]

    movie_names = []
    for i in movie_indexes:
        # print(movies.iloc[i[0], 1])
        movie_names.append([movies_df.iloc[i[0], 4], i[1]])
        # print(movie)
        # print(movies_df.iloc[i[0], 4], i[1])
    return movie_names


#Lowercasing



#Removing Contradictions

import contractions

def remove_contradictions(text):

    return " ".join([contractions.fix(word.text) for word in nlp(text)])



# Removing HTML tags

import re

def remove_html(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)




#Removing URL

import re

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)


def remove_at_the_rate_and_mentions(text):

      return ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)"," ",text).split())


#Remmove punctuation

import string

punc = string.punctuation

def  remove_punc(text):

    return text.translate(str.maketrans('', '', punc))


# Removing stop words


from nltk.corpus import stopwords

stopwords = stopwords.words('english')

def remove_stop_words(text):
    ls = []
    new = []

    ls = nlp(text)

    for word in ls:
        if word.text not in stopwords:

            new.append(word.text)

    return ' '.join(new)


def Lemmetization(text):

    return " ".join([word.lemma_ for word in nlp(text)])



plot = st.text_input('Enter a short description/plot')

max_len = 150
flag=True
cnt=0

if plot != '':
     if st.checkbox('Autocomplete'):
        tok = Tokenizer()
        tok.fit_on_texts(metadata['overview'])
        model = load_model_autocomplete()
        for i in range(10):
                            input_seq = tok.texts_to_sequences([plot])
                            pad_seq = pad_sequences(input_seq, padding='pre', maxlen=max_len-1)
                            predicted = model.predict(pad_seq, batch_size=32)
                            # print(predicted)
                            pos = np.argmax(predicted)
                            # print(pos)
                            print(plot)
                            for word, index in tok.word_index.items():
                                        if pos == index:
                                            plot = plot + ' ' + word
                                            break

                    
     


genre = st.text_input('Want to provide some genres too?') 


with st.expander("Help"):
            
    tags =  []

    st.divider()
     
    if st.checkbox('Proceed'):
        #For plot input

        movie_names_list = list(movies_df.iloc[:, 4])
        name = st.selectbox("Choose a movie", options=movie_names_list, key='2')
        desc = get_desc(name)
        st.code(desc, language='python')
        txt_area_plot = desc
        st.divider()

        
            #Preprocessing
        txt_area_plot = txt_area_plot.lower()
        txt_area_plot = remove_at_the_rate_and_mentions(txt_area_plot)
        txt_area_plot = remove_contradictions(txt_area_plot)
        txt_area_plot = remove_html(txt_area_plot)
        txt_area_plot = remove_punc(txt_area_plot)
        txt_area_plot = remove_stop_words(txt_area_plot)
        txt_area_plot = remove_url(txt_area_plot)
        txt_area_plot = Lemmetization(txt_area_plot)
        
        txt_area_plot = txt_area_plot.split()

        #For genres input
                
        movie_names_list = list(movies_df.iloc[:, 4])
        
        # displaying list of genres
        # genres_ls = movies_df.iloc[:, 3]
        # selected_gen = st.selectbox("Available genres", options=genres_ls)
        # st.divider()
        
        
       
        
        # name = st.selectbox("Choose a movie", options=movie_names_list)
        txt_plot_genre = None
        gen= get_genre(name)
        st.code(gen, language='python')
        txt_plot_genre = gen
        st.divider()

        
        #Preprocessing

        txt_plot_genre = txt_plot_genre.lower()
        txt_plot_genre = remove_at_the_rate_and_mentions(txt_plot_genre)
        txt_plot_genre = remove_contradictions(txt_plot_genre)
        txt_plot_genre = remove_html(txt_plot_genre)
        txt_plot_genre = remove_punc(txt_plot_genre)
        txt_plot_genre = remove_stop_words(txt_plot_genre)
        txt_plot_genre = remove_url(txt_plot_genre)
        txt_plot_genre = Lemmetization(txt_plot_genre)
            
        txt_plot_genre = txt_plot_genre.split()
        
        extracted_actor_names = list(set(extracted_actor_names))
        
        #Preprocessing
        for i in range(len(extracted_actor_names)):
            extracted_actor_names[i] = extracted_actor_names[i] .replace('_', ' ')
            
        # print(extracted_actor_names)
        for i in range(len(extracted_actor_names)):
            extracted_actor_names[i] = extracted_actor_names[i].lower()
            extracted_actor_names[i] = remove_at_the_rate_and_mentions(extracted_actor_names[i])
            extracted_actor_names[i] = remove_contradictions(extracted_actor_names[i])
            extracted_actor_names[i] = remove_html(extracted_actor_names[i])
            extracted_actor_names[i] = remove_punc(extracted_actor_names[i])
            extracted_actor_names[i] = remove_stop_words(extracted_actor_names[i])
            extracted_actor_names[i] = remove_url(extracted_actor_names[i])
            extracted_actor_names[i] = Lemmetization(extracted_actor_names[i])
            # print('--------------------------------------------------')
            # print(i)
        # print(extracted_actor_names)
        if len(txt_area_plot) != 0:
            tags = tags + txt_area_plot
        if  txt_plot_genre is not None:
            tags = tags + txt_plot_genre
        
        
        
        tags = tags  + extracted_actor_names
        # print(tags  + extracted_actor_names)    
        # for i in extracted_actor_names:   
            # print(i, tags)
            # tags += ' '
            # tags += i
         
          
        # print(tags)
        data = []
        corpus=[]
        corpus.append(tags)
        
        
        # print(corpus)   
        # print(corpus)   
        # flatten_corpus = list(itertools.chain(*corpus))

        
         #Dispalying tags
        st.write("Use the following tags in plot section for better results!")
        st.code(tags, language="python")
        st.divider()

