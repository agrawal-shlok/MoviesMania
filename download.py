from pytube import YouTube
import streamlit as st

st.set_page_config(
    page_title = 'Download YT Shorts'
)



def Download(link):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_by_itag(22)
    try:
        youtubeObject.download(filename='video.mp4')
    except:
        print("An error has occurred")
    print("Download is completed successfully")

st.title("Looking to download YT Shorts?")
st.divider()


link = st.text_input("Enter the YouTube video URL: ")
Download(link)