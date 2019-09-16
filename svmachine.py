
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils import shuffle

dataframe = pd.read_csv('train.csv')
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

X = dataframe.drop(['label'], axis=1)
Y = dataframe['label']


# Dataset has been sliced
X_train, Y_train =  X[0:3600], Y[0:3600]
X_test,Y_test = X[3600:5000],Y[3600:5000] 


grid_data = X_train.values[40].reshape(28,28)
#plt.imshow(grid_data,interpolation=None,cmap="gray")
#plt.title(Y_train.values[40])
#plt.show()

model1 = svm.SVC(kernel="poly",C=2,gamma='auto')
model2 = svm.SVC(kernel="linear",C=2)

print("Fitting this might take some time .....")

model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)

print("Training completed, testing ...")
predictions_poly = model1.predict(X_test)
predictions_linear = model2.predict(X_test)

print("Accuracy Score for polynomial func: ", round(metrics.accuracy_score(Y_test, predictions_poly)*100,2),"%")
print("Accuracy Score for linear func: ", round(metrics.accuracy_score(Y_test, predictions_linear)*100,2),"%")