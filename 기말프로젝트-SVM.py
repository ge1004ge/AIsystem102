import numpy as np
from sklearn import svm

x_train, x_test, y_train, y_test = np.load('./img_data.npy', allow_pickle=True)

s = svm.SVC(gamma=0.0001, C=10)
s.fit(x_train,y_train)

res = s.predict(x_test)

conf = np.zeros((3,3))
for i in range(len(res)):
  conf[res[i]][y_test[i]]+=1
print(conf)

no_correct = 0
for i in range(3):
  no_correct+=conf[i][i]
accuracy = no_correct/len(res)
print("Accuracy is", accuracy*100)


