from pytube import YouTube
import streamlit as st
import os
import shutil

#Making a directory for storing downloaded videos
if os.path.isdir('videos/'):
    shutil.rmtree('videos/')
    os.mkdir('videos/')
else:
    os.mkdir('videos/')


def Download(link):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_by_itag(22)
    # try:
    youtubeObject.download(output_path='videos/', filename='video')
    print("Download is completed successfully")
    # except:
        # print("An error has occurred")
    

st.title("Looking to download YT Shorts?")
st.divider()

link = st.text_input("Enter the YouTube video URL: ")

if st.button("Download!"):
    Download(link)
    st.success("Video downloaded successfully")