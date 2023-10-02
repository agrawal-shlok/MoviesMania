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

st.set_page_config(
    page_title = 'Recommend through Plot'
)


st.markdown("<h1 style='text-align: center; color: white;'>Be Creative!</h1>", unsafe_allow_html=True)
st.divider()

plot = st.text_input('Enter a short description/plot')
genre = st.text_input('Want to provide some genres too?') #Provide a pre built list here
title = st.text_input('Maybe an appropriate title too?') #Provide a wide sleection of pre defined title here