# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:54:26 2022

@author: Sai pranay
"""
#-----------------------IMPORTING_THE_DATA_SET---------------------------------


import pandas as pd
glass = pd.read_csv("E:\\DATA_SCIENCE_ASS\\KNN\\glass.csv")
print(glass)
list(glass)
glass.dtypes
glass.shape
glass.describe().T
glass.info()
glass.head()
glass.hist()


#-----------------------------SPLITTING_THE_DATA_SET---------------------------

X = glass.iloc[:,:9]
X

y = glass['Type']
y

#-----------                      -Standardization-----------------------------

from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(X)
X_scale
type(X_scale)

x1 = pd.DataFrame(X_scale)
x1

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x1, y, stratify=y ,random_state=42)  # By default test_size=0.25


#----------------------Model_Fitting---------------------------------------

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, p=1) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train, y_train)

#--------------------------------Prediction------------------------------------
y_pred=knn.predict(X_test)

#--------------------- Compute confusion matrix--------------------------------
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

import numpy as np
print(np.mean(y_pred == y_test).round(2))  
print('Accuracy of KNN with K=5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred).round(2)

knn.score(X_test, y_test).round(2)
