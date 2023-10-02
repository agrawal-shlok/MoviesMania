import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from facenet_pytorch import MTCNN
from PIL import Image


v_cap = cv2.VideoCapture('videoplayback.mp4')

v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))


mtcnn = MTCNN(margin=20, keep_all=True, post_process=False)

batch_size = 8
frames = []

count=0
for _ in tqdm(range(v_len)):
    

    success, frame = v_cap.read()
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
        
        
