import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as dtc

data = pd.read_csv("train.csv").as_matrix()
clf = dtc()

xtrain = data[0:21000, 1:]
train_label = data[0:21000,0]

clf.fit(xtrain, train_label)

xtest = data[21000:,1:]
actual_label = data[21000:,0]

p = clf.predict(xtest)

count = 0
for i in range(0,21000):
    count+=1 if p[i]==actual_label[i] else 0

print("Accuracy : {0}% ".format(round((count/21000)*100,2)))