import numpy as np

from sklearn.model_selection import train_test_split

import os, re, glob
import cv2

dir = './animalface/train/'

categories = os.listdir(dir)
data=[]

for category in categories:
  path = os.path.join(dir, category)
  label = categories.index(category)
  
  for img in os.listdir(path):
    imgpath = os.path.join(path,img)
    animal_img = cv2.imread(imgpath,0)
    try:
      animal_img = cv2.resize(animal_img,(512,512))
      image = np.array(animal_img).flatten()

      data.append([image, label])
    except Exception as e:
      pass
  
features = []
labels = []

for image, label in data:
  features.append(image)
  labels.append(label)
  
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

xy = (x_train, x_test, y_train, y_test)

np.save("./img_data.npy", xy)

