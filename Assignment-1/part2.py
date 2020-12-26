# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 00:43:27 2020

@author: zeya umayya
"""

import time
import scipy.io 
import pandas as pd
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import random
import operator


def split_data(x,y,d_size,perc):
    x_train = pd.DataFrame(columns = ['feat1', 'feat2'])
    y_train = pd.DataFrame(columns = ['label'])
    x_test = pd.DataFrame(columns = ['feat1', 'feat2'])
    y_test = pd.DataFrame(columns = ['label'])
    size_train = int(d_size * perc)
    size_test = int(d_size * (1-perc))
    int(size_train)
    
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices,:]
    y = y[indices,:]
    df1 = pd.DataFrame(data=x[0:size_train,0:2], columns = x_train.columns) 
    df2 = pd.DataFrame(data=y[0:size_train,0:1], columns = y_train.columns) 
    x_train = x_train.append(df1,ignore_index = True)
    y_train = y_train.append(df2,ignore_index = True)
 
    df1 = pd.DataFrame(data=x[size_train:,0:2], columns = x_test.columns) 
    df2 = pd.DataFrame(data=y[size_train:,0:1], columns = y_test.columns) 
    x_test = x_test.append(df1,ignore_index = True)
    y_test = y_test.append(df2,ignore_index = True)
    
    
    return[x_train,y_train,x_test,y_test]
    
def accuracy(pred, actual):
    count =0
    for i in range(6000):
        if pred[i]==actual.iloc[i,0]:
            count = count + 1
    acc = count/6000  
    return acc

####Start from here
""" 2(a) """
mat = scipy.io.loadmat('D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-1\\ML(PG)_assignment_1\\dataset_2.mat')
#print(mat)        
x_2 = mat['samples']
y_2 = mat['labels']
x_2.shape
y_2 = y_2.transpose()
y_2.shape

returned = split_data(x_2,y_2,x_2.shape[0],0.7)

x_train = returned[0]
y_train = returned[1]
x_test = returned[2]
y_test = returned[3]

type(y_train)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
#clf.fit(x_train, y_train)
#out = clf.predict(x_test[0:6000])
#out[0]
#y_test.iloc[0]


depth_ = np.arange(2,17, 1) 
a_accuracy = []
for depth in depth_:    
    clf = DecisionTreeClassifier(max_depth = depth,random_state = 0)
    clf.fit(x_train, y_train)    
    y_pred = clf.predict(x_test[0:6000])
    #score = clf.score(x_test, y_test)
    score = accuracy(y_pred, y_test)
    a_accuracy.append(score)
    
index, value = max(enumerate(a_accuracy), key=operator.itemgetter(1))

plt.plot(depth_,a_accuracy)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('Acc with different Depth values')
plt.show()

"""2(b)  train accuracy and validation accuracy"""
acc_train = []
acc_validation = []
for depth in depth_:    
    clf = DecisionTreeClassifier(max_depth = depth,random_state = 0)
    clf.fit(x_train, y_train)    
    y_pred_train = clf.predict(x_train[0:14000])
    y_pred_test = clf.predict(x_test[0:6000])
    score1 = accuracy(y_pred_train, y_train)
    score2 = accuracy(y_pred_test, y_test)
    acc_train.append(score1)
    acc_validation.append(score2)

print('2b Training Accuracy',acc_train)
print('2b Validation Accuracy',acc_validation)


"""2(c)  """
acc_train = []
acc_validation = []
for depth in depth_:    
    clf = DecisionTreeClassifier(max_depth = depth,random_state = 0)
    clf.fit(x_train, y_train)    
    y_pred_train = clf.predict(x_train[0:14000])
    y_pred_test = clf.predict(x_test[0:6000])
    score1 = accuracy_score(y_train, y_pred_train)
    score2 = accuracy_score(y_test, y_pred_test)
    acc_train.append(score1)
    acc_validation.append(score2)

print('2c Training Accuracy',acc_train)
print('2c Validation Accuracy',acc_validation)

















