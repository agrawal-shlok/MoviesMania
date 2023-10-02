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
import time
from gensim.models import FastText, Word2Vec
import requests
import plotly.express as px
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
import imdb
import tensorflow as tf     
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.preprocessing.text import text_to_word_sequence


from selenium import webdriver
from selenium.webdriver.edge.options import Options

options = Options()
options.add_argument("--headless=new")


from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from tqdm import tqdm
from scrapy.selector import Selector
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import gc 
gc.collect()

# @st.cache_resource
nlp = spacy.load('en_core_web_sm')

#Dependencies
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords

st.set_page_config(
    page_title = 'Upload a Shorts/Video'
)


st.markdown("<h1 style='text-align: center; color: white;'>Upload Here!</h1>", unsafe_allow_html=True)
st.divider()

filenames = pickle.load(open('filenames.pkl', 'rb'))
embeddings_faces = pickle.load(open('embeddings_faces.pkl', 'rb'))
# word2vec_model_spoiler = pickle.load(open('word2vec_model_spoiler_reviews.pkl', 'rb'))

embeddings_faces_np = np.array(embeddings_faces)

@st.cache_resource
def feature_extractor():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    model_face = load_weights(model)
    return model_face


@st.cache_resource   
def load_weights(_model):
    # Load VGG Face model weights
    _model.load_weights('archive (7)/vgg_face_weights.h5')
    vgg_face=Model(inputs=_model.layers[0].input,outputs=_model.layers[-2].output)
    return vgg_face


model=feature_extractor()

@st.cache_resource
def load_model_spoiler():
    model = tf.keras.saving.load_model("bidirectional_lstm_spoiler_model.keras")
    
    return model

def extract_names(features):
    similarity = []
    for i in tqdm(range(len(embeddings_faces_np))):
        sine = cosine_similarity(np.array(features).reshape(1, -1), np.array(embeddings_faces_np[i].reshape(1, -1)[0]).reshape(1, -1))
        similarity.append(sine[0])
    # print(len(similarity))
        # print(sine)
    face_index = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0]
    return face_index

def extract(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    
    result = model.predict(preprocessed_img).flatten()
    return result
   

def clean():
    detector = MTCNN()

    dir_path = 'extracted_images_mtcnn_pytorch_new/'
    
    new_dir = 'blurred_new/'
    
    if os.path.isdir('blurred_new/'):
        shutil.rmtree('blurred_new/')
        os.mkdir('blurred_new/')
    else:
        os.mkdir('blurred_new/')

    for images in tqdm(os.listdir(dir_path)):
        # print(images
        img_path = dir_path + images

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        text = 'Not Blurry'
               
        threshold = 50

        if fm < threshold:
                text = 'Blurry'
                # print(text)
                old_dir = dir_path + images
                # print(old_dir)
                shutil.move(old_dir,new_dir)

                
    for images in tqdm(os.listdir(dir_path)):

            img_path = os.path.join(dir_path, images)
            # print(img_path)
     
            try:
                img = cv2.imread(img_path)
            
                location = detector.detect_faces(img)
                
            except:
                pass
                
            else:
            
                if len(location) > 0:

                    for face in location:

                        x, y, w, h = face['box']
                        confidence = face['confidence']
                        x2, y2 = x + w, y + h
                        
                        if confidence < 0.99:

                            old_dir = dir_path + images
                            # print(old_dir)
                            shutil.move(old_dir,new_dir)

     
                else:
                    continue

def num_remove(string):
    temp = []
    # ls = string.split('_')
    for i in range(len(string)):
        if string[i].isnumeric()==False :
            temp.append(string[i])
    return ''.join(temp)
    
def wait():
    st.markdown("<h2 style='text-align: center; color: white;'>While you wait take a look at our another feature!</h2>", unsafe_allow_html=True)

    #Loading the final model.pkl file

    file_name = 'final_model_movie_reviews_sentiment_analysis (copy).pkl'
    open_file = open(file_name, 'rb')
    final_model = pickle.load(open_file)
    open_file.close()


    #Loading the word2vec model.pkl file

    file_name = 'word2vec_model (copy).pkl'
    open_file = open(file_name, 'rb')
    word2vec_model = pickle.load(open_file)
    open_file.close()

    def padding():
        tok = Tokenizer()
        tok.fit_on_texts(reviews_and_title['tags2'])
        vocab_size = len(tok.word_index) + 1
        max_len = 200
        encd_reviews = tok.texts_to_sequences(reviews_and_title['tags2'])
        embd_dims=300
        pad_reviews = pad_sequences(maxlen = max_len, padding='pre', sequences=encd_reviews)
        
        return pad_reviews
    
    #Getting thwe movie reviews
    def fetch_reviews(final_name):
        
        number = get_movie_id(final_name)
        driver = webdriver.Edge(options=options)
        url = 'https://www.imdb.com/title/tt{}/reviews?sort=submissionDate&dir=asc&ratingFilter=0'.format(number)

        time.sleep(1)
        driver.get(url)

        # print(driver.title)
        time.sleep(1)


        #Extracting the review count

        sel = Selector(text = driver.page_source)
        review_counts = sel.css('.lister .header span::text').extract_first().replace(',', '').split(' ')[0]

        total = int(int(review_counts) / 25)
        # print(total)
        # #Click on the 'load more' buttion n times

     
        for i in tqdm(range(2)):

                    # css_selector = 'load-more-trigger'
                    # driver.find_element(By.ID, css_selector).click()

                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div/div[2]/div[3]/div[1]/section/div[2]/div[4]/div/button"))).click()
                    #Extracting the reviews and Title

                    reviews = driver.find_elements(By.CSS_SELECTOR, 'div.review-container')

                    review_title_list = []
                    review_list = []

                    for i in tqdm(reviews):
                        try:
                            sel2 = Selector(text = i.get_attribute('innerHTML'))

                            try:
                                    review = sel2.css('.text.show-more__control::text').extract_first()

                            except:
                                        review = np.NaN
                            try:
                                    review_title = sel2.css('a.title::text').extract_first().replace('\n', '')
                            except:
                                    review_title = np.Nan

                            review_list.append(review)
                            review_title_list.append(review_title)

                        except Exception as e:
                            error_url_list.append(url)
                            error_msg_list.append(e)

                    reviews_df = pd.DataFrame({
                                'Review': review_list,
                                'Title': review_title_list
                        })   

   
        reviews_df.to_csv('fetched_reviews.csv')
        return reviews_df




    def preprocess_spoilers(st):
        
        #Lowercasing

        reviews_and_title['{}'.format(st)] = reviews_and_title['{}'.format(st)].str.lower()


        # Removing HTML tags

        import re

        def remove_html(text):
            pattern = re.compile('<.*?>')
            return pattern.sub(r'', text)

        reviews_and_title['{}'.format(st)] = reviews_and_title['{}'.format(st)].apply(remove_html)
        # print(df.head())


        #Remove @

        def remove_at_the_rate(text):

            ls = []
            new = []

            ls = nlp(text)

            for word in ls:
                if word.text != "@":
                    new.append(word.text)

            return ' '.join(new)

        reviews_and_title['{}'.format(st)] = reviews_and_title['{}'.format(st)].apply(remove_at_the_rate)



        #Removing URL

        import re

        def remove_url(text):
            pattern = re.compile(r'https?://\S+|www\.\S+')
            return pattern.sub(r'', text)

        reviews_and_title['{}'.format(st)]= reviews_and_title['{}'.format(st)].apply(remove_url)
       


        #Remmove punctuation

        import string

        punc = string.punctuation

        def  remove_punc(text):

            return text.translate(str.maketrans('', '', punc))

        reviews_and_title['{}'.format(st)]= reviews_and_title['{}'.format(st)].apply(remove_punc)


        # Removing stop words

        stpwords = stopwords.words('english')

        def remove_stop_words(text):
            ls = []
            new = []

            ls = nlp(text)

            for word in ls:
                if word.text not in stpwords:

                    new.append(word.text)

            return ' '.join(new)

        reviews_and_title['{}'.format(st)] = reviews_and_title['{}'.format(st)].apply(remove_stop_words)


        #Removing Contradictions

        import contractions

        def remove_contradictions(text):

            return " ".join([contractions.fix(word.text) for word in nlp(text)])

        reviews_and_title['{}'.format(st)]= reviews_and_title['{}'.format(st)].apply(remove_contradictions)



        def Lemmetization(text):

            return " ".join([word.lemma_ for word in nlp(text)])



        reviews_and_title['{}'.format(st)] = reviews_and_title['{}'.format(st)].apply(Lemmetization)
        
        #Making the corpus
        
        def making_corpus_each(st):
            corpus = []
            corpus.append(st)

            return corpus
        
        reviews_and_title['{}'.format(st)] = reviews_and_title['{}'.format(st)].apply(making_corpus_each)
        
        global story 
        story = []
        
        def making_final_corpus(ls):
 
            story.append(ls)


        
        reviews_and_title['{}'.format(st)] = reviews_and_title['{}'.format(st)].apply(making_final_corpus)
        # print('----------------------------------------------------------------', len(story), story, story[0])
        
        #Converting the story list from 3d to 2d
        from itertools import chain
        global flatten_corpus 
        flatten_corpus = []
        flatten_corpus = list(chain.from_iterable(story))
        # print('----------------------------------------------------------------', len(flatten_corpus), flatten_corpus[0], flatten_corpus)



    def preprocess(st):
        
        #Lowercasing

        reviews_and_title_temp['{}'.format(st)] = reviews_and_title_temp['{}'.format(st)].str.lower()


        # Removing HTML tags

        import re

        def remove_html(text):
            pattern = re.compile('<.*?>')
            return pattern.sub(r'', text)

        reviews_and_title_temp['{}'.format(st)] = reviews_and_title_temp['{}'.format(st)].apply(remove_html)
        # print(df.head())


        #Remove @

        def remove_at_the_rate(text):

            ls = []
            new = []

            ls = nlp(text)

            for word in ls:
                if word.text != "@":
                    new.append(word.text)

            return ' '.join(new)

        reviews_and_title_temp['{}'.format(st)] = reviews_and_title_temp['{}'.format(st)].apply(remove_at_the_rate)



        #Removing URL

        import re

        def remove_url(text):
            pattern = re.compile(r'https?://\S+|www\.\S+')
            return pattern.sub(r'', text)

        reviews_and_title_temp['{}'.format(st)]= reviews_and_title_temp['{}'.format(st)].apply(remove_url)
       


        #Remmove punctuation

        import string

        punc = string.punctuation

        def  remove_punc(text):

            return text.translate(str.maketrans('', '', punc))

        reviews_and_title_temp['{}'.format(st)]= reviews_and_title_temp['{}'.format(st)].apply(remove_punc)


        # Removing stop words

        stpwords = stopwords.words('english')

        def remove_stop_words(text):
            ls = []
            new = []

            ls = nlp(text)

            for word in ls:
                if word.text not in stpwords:

                    new.append(word.text)

            return ' '.join(new)

        reviews_and_title_temp['{}'.format(st)] = reviews_and_title_temp['{}'.format(st)].apply(remove_stop_words)


        #Removing Contradictions

        import contractions

        def remove_contradictions(text):

            return " ".join([contractions.fix(word.text) for word in nlp(text)])

        reviews_and_title_temp['{}'.format(st)]= reviews_and_title_temp['{}'.format(st)].apply(remove_contradictions)



        def Lemmetization(text):

            return " ".join([word.lemma_ for word in nlp(text)])



        reviews_and_title_temp['{}'.format(st)] = reviews_and_title_temp['{}'.format(st)].apply(Lemmetization)
        
        #Making the corpus
        
        def making_corpus_each(st):
            corpus = []
            corpus.append(st)

            return corpus
        
        reviews_and_title_temp['{}'.format(st)] = reviews_and_title_temp['{}'.format(st)].apply(making_corpus_each)
        
        global story 
        story = []
        
        def making_final_corpus(ls):
 
            story.append(ls)


        
        reviews_and_title_temp['{}'.format(st)] = reviews_and_title_temp['{}'.format(st)].apply(making_final_corpus)
        # print('----------------------------------------------------------------', len(story), story, story[0])
        
        #Converting the story list from 3d to 2d
        from itertools import chain
        global flatten_corpus 
        flatten_corpus = []
        flatten_corpus = list(chain.from_iterable(story))
        # print('----------------------------------------------------------------', len(flatten_corpus), flatten_corpus[0], flatten_corpus)


    def preprocess_strings(st):

        #Lowercasing

        st = st.lower()


        # Removing HTML tags

        import re

        def remove_html(text):
            pattern = re.compile('<.*?>')
            return pattern.sub(r'', text)

        st = remove_html(st)
        # print(df.head())


        #Remove @

        def remove_at_the_rate(text):

            ls = []
            new = []

            ls = nlp(text)

            for word in ls:
                if word.text != "@":
                    new.append(word.text)

            return ' '.join(new)

        st = remove_at_the_rate(st)



        #Removing URL

        import re

        def remove_url(text):
            pattern = re.compile(r'https?://\S+|www\.\S+')
            return pattern.sub(r'', text)

        st = remove_url(st)
       


        #Remmove punctuation

        import string

        punc = string.punctuation

        def  remove_punc(text):

            return text.translate(str.maketrans('', '', punc))

        st = remove_punc(st)



        from autocorrect import Speller

        check = Speller()

        def check_spell(text):

            return check(text)

        # train_df['Description_preprocessed'] = train_df['Description_preprocessed'].apply(check_spell)


        # Removing stop words

        stpwords = stopwords.words('english')

        def remove_stop_words(text):
            ls = []
            new = []

            ls = nlp(text)

            for word in ls:
                if word.text not in stpwords:

                    new.append(word.text)

            return ' '.join(new)

        st = remove_stop_words(st)


        #Removing Contradictions

        import contractions

        def remove_contradictions(text):

            return " ".join([contractions.fix(word.text) for word in nlp(text)])

        st = remove_contradictions(st)



        def Lemmetization(text):

            return " ".join([word.lemma_ for word in nlp(text)])



        st = Lemmetization(st)
          
        return st

        
    final_embeddings = []
        
    def embeddings_generation(name):
        
        final_embeddings.clear()
        #Building vocabulary
        vocabulary = set(name.wv.index_to_key)
        
        
        for i in flatten_corpus:
            avg_embeddings = None
            for j in i:

                if j in vocabulary:

                    if avg_embeddings is None:
                        avg_embeddings = name.wv[j]
                    else:
                        avg_embeddings = avg_embeddings + name.wv[j]
            if avg_embeddings is not None:
                avg_embeddings = avg_embeddings / len(avg_embeddings)
                final_embeddings.append(avg_embeddings)

                
    def fetch_corresponding_reviews(pos, neg):
        
        index = []
        review = []
        title = []
        
        for i in pos:
            
                index.append(i[0])
                review.append(reviews_and_title.iloc[i[0]]['Title'])
                title.append(reviews_and_title.iloc[i[0]]['Review'])
            
        pos_dict = ({

                'index':index,
                'Title': review,
                'Review': title
        })
            
            
        index = []
        review = []
        title = []
        
            
        for i in neg:
            
                index.append(i[0])
                review.append(reviews_and_title.iloc[i[0]]['Title'])
                title.append(reviews_and_title.iloc[i[0]]['Review'])
                
        neg_dict = ({

                'index':index,
                'Title': review,
                'Review':title
        })
            
        # print(neg_dict)
        # print(pos, neg)
        global pos_df 
        pos_df = pd.DataFrame(pos_dict) 
        global neg_df 
        neg_df = pd.DataFrame(neg_dict)
        
        
        
    def senti_analysis(df):
        
        preprocess('Review')
        
        embeddings_generation(word2vec_model)
        
        # print(final_embeddings)
        y_pred = final_model.predict(final_embeddings)
        
        # print(y_pred)
        
        predict = list(enumerate(y_pred))
        # print(predict)
        pos = []
        neg = []
        
        for i in predict:
            if i[1] == 1:
                pos.append(i)
            elif i[1] == 0:
                neg.append(i)
        
      
        
        fetch_corresponding_reviews(pos, neg)
        
        
    title = []

    #Fetching the image of the movie
    def fetch_list_of_movies():

        headers = {
           "User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200"
    }
        for i in tqdm(range(1)):
            
            url = "https://www.themoviedb.org/movie?page={}".format(i)
            response = requests.get(url, headers=headers).text

            soup = BeautifulSoup(response, 'lxml')
            
            page_wrapper_div = soup.find_all('div', class_='page_wrapper')

            headings = []
            for item in page_wrapper_div:
                headings = item.find_all('h2')
            
            
            for i in headings:

                title.append(i.text)

        # print(title)
        # print(len(title))
        
        #Saving the data to a pickle file
        
        pickle.dump(title, open('title_list.pkl', 'wb'))


    #Getting the homepage and image url    
    def fetch_image(final_name):
        
        number = get_movie_id(final_name)
        url = 'https://api.themoviedb.org/3/find/tt{}?api_key=7f1ea36a1c2e4ec69b46aaa87b5d24d6&external_source=imdb_id'.format(number)
        headers = {
           "User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200"
    }
       
        response = requests.get(url, headers=headers)
        
        data = response.json()

        image_url = 'https://image.tmdb.org/t/p/w400/' + data['movie_results'][0]['poster_path']
        homepage_url = 'https://www.themoviedb.org/movie/{}'.format(data['movie_results'][0]['id'])
        

        return image_url, homepage_url


    #Getting the synopsis of the movie

    def fetch_synopsis(final_name):
        
        number = get_movie_id(final_name)
        url = 'https://api.themoviedb.org/3/find/tt{}?api_key=7f1ea36a1c2e4ec69b46aaa87b5d24d6&external_source=imdb_id'.format(number)
        headers = {
           "User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200"
    }
       
        response = requests.get(url, headers=headers)
        
        data = response.json()

        overview = data['movie_results'][0]['overview']
        
        return overview


    #to be able to download a dataframe 
    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')



    #Getting the IMDB movie id
    def get_movie_id(final_name):
        
        ia = imdb.Cinemagoer()
        movie = ia.search_movie(final_name)
        string = movie[0]
        movie_id = string.movieID
        return movie_id
        
        
    #Giving our web app a title

    # st.markdown("<h1 style='text-align: center; color: white;'>Movies Review System with Sentiment Analysis</h1>", unsafe_allow_html=True)

    # st.title("")


    #Loading the title_list.pkl file

    file_name = 'title_list (copy).pkl'
    open_file = open(file_name, "rb")
    title_names = pickle.load(open_file)
    open_file.close()


    #Giving movie options to the user
    name = st.selectbox("Choose your movie! ", options=title_names, key='987')
    release_date = st.text_input("Please enter the entered movie's relaese date")

    final_name = name + " " + "(" + release_date + ")"
    print(final_name)





    #Button to fetch reviews

    if st.checkbox('Get reviews!'): 

        #Progress bar

        bar = st.progress(0)
        
        tabs_titles = ['Movie', 'Sentiment Analysis of Reviews!', 'Spoiler Free Reviews']
        
        tab2, tab3, tab4 = st.tabs(tabs_titles)
        
     
        with tab2:
            
            st.title(name)
            col1, col2 = st.columns([2, 2])
            
            with col1:


                    img_url, home_url = fetch_image(final_name)
                    st.image(img_url)
                    st.write("[Explore](%s)" % home_url)


            bar.progress(30)
            time.sleep(1)

            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Synopsis</h3>", unsafe_allow_html=True)
                text = fetch_synopsis(final_name)
                st.markdown("<h6 style='text-align: center; color: white;'>{}</h6>".format(text), unsafe_allow_html=True)

            bar.progress(50)
            time.sleep(1)

            reviews_and_title = fetch_reviews(final_name)
            reviews_and_title_temp = pd.DataFrame()
            reviews_and_title_temp['Review'] = reviews_and_title['Review']
            senti_analysis(reviews_and_title_temp)
            bar.progress(100)
            
                
            
        with tab3:
            
            with st.expander("All Reviews"):
            

                        
                st.dataframe(reviews_and_title, height=250, width=1000, hide_index=True)

                #Option to downlaod the csv file


                csv = convert_df(reviews_and_title)
                
                #THis reset the web page everytime so be careful!
                st.download_button(
                     label="Download data as CSV",
                     data=csv,
                     file_name='reviews_df.csv',
                     mime='text/csv',
                 )
                
                
            with st.expander("Postive Reviews"):
            
                #Data
                
                st.dataframe(pos_df, height=250, width=1000, hide_index=True)
     
            
            with st.expander("Negative Reviews"):
                
                
                #Data
                st.dataframe(neg_df, height=250, width=1000, hide_index=True)
        
        with tab4:
            if st.checkbox('Show'):
                gc.collect()
                spoiler = load_model_spoiler()
                reviews_and_title['tags'] = reviews_and_title['Review'] + reviews_and_title['Title']
                
                #Preprocessing
                reviews_and_title['tags2'] = reviews_and_title['tags']
                preprocess_spoilers('tags')
                # reviews_and_title['tags'] = reviews_and_title['tags'].str.lower()
                # reviews_and_title['tags'] = reviews_and_title['tags'].apply(remove_at_the_rate_and_mentions)
                # reviews_and_title['tags'] = reviews_and_title['tags'].apply(remove_contradictions)
                # reviews_and_title['tags'] = reviews_and_title['tags'].apply(remove_html)
                # reviews_and_title['tags'] = reviews_and_title['tags'].apply(remove_punc)
                # reviews_and_title['tags'] = reviews_and_title['tags'].apply(remove_stop_words)
                # reviews_and_title['tags'] = reviews_and_title['tags'].apply(remove_url)
                # reviews_and_title['tags'] = reviews_and_title['tags'].apply(Lemmetization)
                
                padded_revs = padding()
                
                yhat = spoiler.predict([padded_revs], batch_size=1024)
                y_pred = np.where(yhat > 0.02, 1, 0)
                
                spoiler_true=[]
                spoiler_false=[]
                
                predict = list(enumerate(y_pred))
                for i in predict:
                    if i[1] == 1:
                        spoiler_true.append(i)
                    elif i[1] == 0:
                        spoiler_false.append(i)
                fetch_corresponding_reviews(spoiler_true, spoiler_false)
                
                
                with st.expander("All Reviews"):
            

                     
                    st.dataframe(reviews_and_title, height=250, width=1000, hide_index=True)


                
                with st.expander("Spoiler Free Reviews"):
                
                        #Data
                        spoiler_free_df = pos_df      
                        st.dataframe(spoiler_free_df, height=250, width=1000, hide_index=True)
         
                
                with st.expander("May contain Spoiler "):
                    
                    
                        #Data
                        spoiler_df = neg_df     
                        st.dataframe(spoiler_df, height=250, width=1000, hide_index=True)

                
        # with tab4:
              # pass
            # with st.expander(":bar_chart: Distribution fo Postive and Negative Reviews"):
                 
                # num_pos = pos_df.shape
                # num_neg = neg_df.shape
                # print(num_neg, num_pos)
                # dic = {
                    # 'Criteria': ['Positive', 'Negative'],
                    # 'Number': [num_pos[0], num_neg[0]]
                # }
                
                # chart_data = pd.DataFrame(dic)
                # st.dataframe(chart_data, height=250, width=1000, hide_index=True)
                # fig = px.bar(
                    # x=chart_data.iloc[0],
                    # y=chart_data.iloc[1],
                    
                # )
                # st.plotly_chart(fig)


            # with st.expander("Most frequent words in Postitve reviews"):
            
                # pos_corpus= []
                # neg_corpus = []
         
                # for i in pos_df['Review'].tolist():
                    
                    # string_preprocessed = preprocess_strings(i)
                    # ls = string_preprocessed.split()
                    # print('neg:', string_preprocessed)
                    
                    # for j in ls:
                        
                        # pos_corpus.append(j)
                        
                # for i in neg_df['Review'].tolist():
                    
                    # string_preprocessed = preprocess_strings(i)
                    # ls = string_preprocessed.split()
                    # print('neg:', string_preprocessed)
                    
                    # for j in ls:
                        
                        # neg_corpus.append(j)

                # from collections import Counter
                
                # fig = px.bar(
                    # x=pd.DataFrame(Counter(pos_corpus).most_common(30))[0],
                    # y=pd.DataFrame(Counter(pos_corpus).most_common(30))[1],
                    
                # )
                # st.plotly_chart(fig)
                
                
            
            # with st.expander("Most frequent words in Negative reviews"):
      
                # fig = px.bar(
                    # x=pd.DataFrame(Counter(neg_corpus).most_common(30))[0],
                    # y=pd.DataFrame(Counter(neg_corpus).most_common(30))[1],
                   
                # )
                # st.plotly_chart(fig)



if os.path.isdir('extracted_images_mtcnn_pytorch_new/'):
    shutil.rmtree('extracted_images_mtcnn_pytorch_new/')
    os.mkdir('extracted_images_mtcnn_pytorch_new/')
else:
    os.mkdir('extracted_images_mtcnn_pytorch_new/')
    
if st.checkbox("Already downloaded a video file through the 'Downloads' tab"):

        
        vf = cv2.VideoCapture("videos/video")
        v_len = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))


        mtcnn = MTCNN(margin=20, keep_all=True, post_process=False)

        batch_size = 8
        frames = []

        count=0
        wait()
        for _ in tqdm(range(v_len)):


            success, frame = vf.read()
            if not success:
                continue

            # Add to batch, resizing for speed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            #frame = frame.resize([int(f * 0.5) for f in frame.size])
            frames.append(frame)
            count+=1

            if len(frames) >= batch_size:

                # Batch

                    
                save_paths = [f'extracted_images_mtcnn_pytorch_new/image_{count}.jpg' for i in range(len(frames))]
                mtcnn(frames, save_path=save_paths);

                frames = []
                
                
if st.checkbox("To upload a video file"):
    f = st.file_uploader('Here')
    wait()
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        vf = cv2.VideoCapture(tfile.name)
        v_len = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))


        mtcnn = MTCNN(margin=20, keep_all=True, post_process=False)

        batch_size = 8
        frames = []

        count=0
        for _ in tqdm(range(v_len)):


            success, frame = vf.read()
            if not success:
                continue

            # Add to batch, resizing for speed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            #frame = frame.resize([int(f * 0.5) for f in frame.size])
            frames.append(frame)
            count+=1

            if len(frames) >= batch_size:

                # Batch

                    
                save_paths = [f'extracted_images_mtcnn_pytorch_new/image_{count}.jpg' for i in range(len(frames))]
                mtcnn(frames, save_path=save_paths);

                frames = []
            
clean()

names = []
for file in tqdm(os.listdir('extracted_images_mtcnn_pytorch_new/')):
        embeddings = extract(os.path.join('extracted_images_mtcnn_pytorch_new/', file))
        index = extract_names(embeddings)
        # name = filenames[index[0]]
        names.append(index)
        
actors = sorted(names, reverse=True, key=lambda x: x[1])
print(actors)

actors_names = []

#appending actors names to a list
for i in actors:
    if i[1] >= 0.78:
        string = filenames[i[0]]
        string = string.split('.')
        temp = []
        print(string)
        for i in string[:-1]:
            if i not in ['face','.png', '.jpg']:
                temp.append(i)
        string = ' '.join(temp)
        print(string)
        string = num_remove(string)
        print(type(string))
        # st.write(string)
        temp = []
        string = string.split()
        for i in string:
            if i not in ['face', 'face_', '_face','.png', '.jpg']:
                temp.append(' ')
                temp.append(i)
        string = ''.join(temp)
        actors_names.append(string)
        
print(actors_names)      
        
pickle.dump(actors_names, open('extracted_actor_names.pkl', 'wb'))
st.success("Sucessfully processed information")

