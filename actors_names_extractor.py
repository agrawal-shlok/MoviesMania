import os
import pickle

path = 'test'
filenames = []

for names in os.listdir(path):
    filenames.append(names)
# print(filenames)
pickle.dump(filenames, open('filenames.pkl', 'wb'))