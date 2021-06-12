import numpy as np

import os, re, glob
import cv2

from sklearn.neural_network import MLPClassifier

x_train, x_test, y_train, y_test = np.load('./img_data.npy', allow_pickle=True)

mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), learning_rate_init=0.0001,
                    batch_size=32, solver='adam', verbose=True)

mlp.fit(x_train, y_train)
res = mlp.predict(x_test)

print(len(res))

conf = np.zeros((3,3))

for i in range(len(res)) :
    conf[res[i]][y_test[i]]+=1
print(conf)

correct = 0
for i in range(3):
    correct += conf[i][i]
accuracy =correct/len(res)
print("Accuracy is", accuracy*100)

