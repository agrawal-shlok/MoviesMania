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
    
    
    
def load_weights(model):
    # Load VGG Face model weights
    model.load_weights('archive (7)/vgg_face_weights.h5')
    vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
    return vgg_face

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
    model=feature_extractor()
    result = model.predict(preprocessed_img).flatten()
    return result
    
import pickle

filenames = pickle.load(open('filenames.pkl', 'rb'))
embeddings_faces = pickle.load(open('embeddings_faces.pkl', 'rb'))
embeddings_faces_np = np.array(embeddings_faces)

    
st.set_page_config(
    page_title = 'Recommend through Video'
)
name = st.title('Upload a video')
st.divider()

f = st.file_uploader('Here')
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
            if os.path.isdir('extracted_images_mtcnn_pytorch_new/'):
                os.rmdir('extracted_images_mtcnn_pytorch_new/')
            else:
                os.mkdir('extracted_images_mtcnn_pytorch_new/')
                
            save_paths = [f'extracted_images_mtcnn_pytorch_new/image_{count}.jpg' for i in range(len(frames))]
            mtcnn(frames, save_path=save_paths);

            frames = []
            

names = []
for file in tqdm(os.listdir('extracted_images_mtcnn_pytorch_new/')):
        embeddings = extract(os.path.join('extracted_images_mtcnn_pytorch_new/', file))
        index = extract_names(embeddings)
        # name = filenames[index[0]]
        names.append(index)
        
actors = sorted(names, reverse=True, key=lambda x: x[1])

for i in actors:
    if i[1] >= 0.80:
        print(filenames[i[0]])

# print(names)

#     stframe = st.empty()    
    
#     while vf.isOpened():
#         ret, frame = vf.read()
        
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         stframe.image(img)


st.markdown("<h1 style='text-align: center; color: white;'></h1>", unsafe_allow_html=True)
st.divider()
