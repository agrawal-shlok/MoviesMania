import cv2
from tqdm import tqdm
import os
import shutil
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from mtcnn import MTCNN

detector = MTCNN()

dir_path = 'extracted_images_mtcnn_pytorch_new/'

new_dir = 'blurred_new/'

os.mkdir(new_dir)

for images in tqdm(os.listdir(dir_path)):
    # print(images
    img_path = dir_path + images

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    text = 'Not Blurry'
           
    threshold = 60

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
                    
                    if confidence < 0.999:

                        old_dir = dir_path + images
                        # print(old_dir)
                        shutil.move(old_dir,new_dir)

 
            else:
                continue
            
# for images in tqdm(os.listdir(dir_path)):

#         img_path = os.path.join(dir_path, images)
#         # print(img_path)
 
#         try:
#             img = cv2.imread(img_path)
        
#             location = detector.detect_faces(img)
            
#         except:
#             pass
            
#         else:
        
#             if len(location) > 0:

#                 for face in location:

#                     x, y, w, h = face['box']
#                     confidence = face['confidence']
#                     x2, y2 = x + w, y + h
                    
#                     if confidence < 0.97:
                       
#                         try:
#                             old_dir = dir_path + images
#                             # print(old_dir)
#                             shutil.move(old_dir,new_dir)


#                         except:
#                             pass
 
#             else:
#                 continue
