# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:25:31 2022

@author: Sai pranay
"""

#-----------------------IMPORTING_THE_DATA_SET---------------------------------


import pandas as pd
zoo = pd.read_csv("E:\\DATA_SCIENCE_ASS\\KNN\\Zoo.csv")
print(zoo)
list(zoo)
zoo.shape
zoo.head()
zoo.info()
zoo.describe().T
zoo.hist()

#--------------Droping---------------------------------------------------------
zoo1 = zoo.drop(["animal name"],axis = 1)
zoo1

#-----------------------------SPLITTING_THE_DATA_SET---------------------------


X = zoo1.iloc[:,0:16]
X

Y = zoo1['type']
Y

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y ,random_state=42)  # By default test_size=0.25


#----------------------Model_Fitting-------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, p=1) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train, y_train)

#--------------------------Prediction------------------------------------------
y_pred=knn.predict(X_test)

# Compute confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

import numpy as np
print(np.mean(y_pred == y_test).round(2))  
print('Accuracy of KNN with K=5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred).round(2)

knn.score(X_test, y_test).round(2)
