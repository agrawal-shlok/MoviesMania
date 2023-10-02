import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from mtcnn import MTCNN
detector = MTCNN()

actors = os.listdir('clean')


# print(actors)

filenames = []

for actor in tqdm(actors):

    count=0
    for file in os.listdir(os.path.join('clean', actor)):
        img_path = os.path.join('clean', actor, file)
 
        try:
            img = cv2.imread(img_path)
        
            location = detector.detect_faces(img)
            
        except:
            pass
        
        if count >= 100:
            break
            
        else:
        
            if len(location) > 0:

                for face in location:

                    x, y, w, h = face['box']
                    confidence = face['confidence']
                    x2, y2 = x + w, y + h
                    
                    if confidence >=0.9:
                        # cv2.rectangle(img, (x, y), (x2, y2), (255, 255,255), 1)
                        if img is not None:
                            rec = img[y:y2, x:x2]
                        # print(frame_num)
                        try:
                            cv2.imwrite('names_clean/{}'.format(actor) + '_{}'.format(count) + '.jpg', rec)
                            count+=1

                        except:
                            pass
                    
                    # cv2.imshow(rec)
                    # result.write(frame)
                    # cv2.imshow("result",frame)
            else:
                continue
 
 
cv2.destroyAllWindows()
# video.release()
