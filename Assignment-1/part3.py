# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 23:34:35 2020

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
    x_train = pd.DataFrame(columns = x.columns)
    y_train = pd.DataFrame(columns = y.columns)
    x_test = pd.DataFrame(columns = x.columns)
    y_test = pd.DataFrame(columns = y.columns)
    size_train = int(d_size * perc)
    size_test = int(d_size * (1-perc))
    int(size_train)
    
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x.iloc[indices,:]
    y = y.iloc[indices,:]
    df1 = pd.DataFrame(data=x.iloc[0:size_train,0:12], columns = x.columns) 
    df2 = pd.DataFrame(data=y.iloc[0:size_train,0:1], columns = y.columns) 
    x_train = x_train.append(df1,ignore_index = True)
    y_train = y_train.append(df2,ignore_index = True)
 
    df1 = pd.DataFrame(data=x.iloc[size_train+1:,0:12], columns = x.columns) 
    df2 = pd.DataFrame(data=y.iloc[size_train+1:,0:1], columns = y.columns) 
    x_test = x_test.append(df1,ignore_index = True)
    y_test = y_test.append(df2,ignore_index = True)
    
    
    return[x_train,y_train,x_test,y_test]

def split_data_train(x,y,d_size):
  x_train = pd.DataFrame(columns = x.columns)
  y_train = pd.DataFrame(columns = y.columns)
  size_train = int(d_size * 0.5)
  int(size_train)
  indices = np.arange(x.shape[0])
  
  np.random.shuffle(indices)
  x = x.iloc[indices,:]
  y = y.iloc[indices,:]
  df1 = pd.DataFrame(data=x.iloc[0:size_train,0:12], columns = x.columns) 
  df2 = pd.DataFrame(data=y.iloc[0:size_train,0:1], columns = y.columns) 
  x_train = x_train.append(df1,ignore_index = True)
  y_train = y_train.append(df2,ignore_index = True)
  
  return[x_train,y_train]

def ensemble(x_train, y_train, x_test, y_test, size, depth, stump):
    print(size, depth, stump)
    size_data = int(size*0.8)
    clf = []
    for i in range(stump):
      clf_i = DecisionTreeClassifier(max_depth = depth)
      clf.append(clf_i)
      [x_trainc, y_trainc] = split_data_train(x_train,y_train,size_data)
      y_trainc = y_trainc.astype('int')
      clf[i].fit(x_trainc,y_trainc)
      y_pred_test = clf[i].predict(x_test[0:int(size*0.2)])
      print(i,accuracy_score(y_test, y_pred_test))
    return clf

def maj_vot(size, clf, x_data, stump):
    y_pred = [0]*(int(size))
    for i in range(int(size)):
      y_temp = [0,0,0,0,0,0,0,0,0,0,0,0]
      j=0
      for j in range(stump):
        #print(x_data[i:i+1])
        val = clf[j].predict(x_data[i:i+1])
        y_temp[int(val-1)] = y_temp[int(val-1)] +1
      maxx = 0
      #print(y_temp)
      k=0
      for k in range(12):
         if y_temp[k] >  maxx:
           maxx = y_temp[k]
           index = k
      #print(index)
      y_pred[i] = index+1
    return y_pred

"""Main data reading and other stuff starts from here"""

data = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv');

#data.head(100)
data.shape
data.columns
"""dropping index column"""
data = data.drop(['No'],axis=1)

data.isnull().sum()
#data.iloc[:,0:1]

#data['pm2.5'][1:1000]

data.interpolate(limit_direction="both", inplace=True)
data.isnull().sum()  #now zero

data.cbwd[data.cbwd == 'NW'] = 1
data.cbwd[data.cbwd == 'SE'] = 2
data.cbwd[data.cbwd == 'NE'] = 3
data.cbwd[data.cbwd == 'cv'] = 4

x_1 = data.iloc[:,[0,2,3,4,5,6,7,8,9,10,11]]
y_1 = data.iloc[:,1:2]
size = data.shape[0]

returned = split_data(x_1,y_1,size,0.8)

x_train = returned[0]
y_train = returned[1]
x_test = returned[2]
y_test = returned[3]

type(y_train)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)    
y_pred = clf.predict(x_test[0:int(size*0.2)])

score = accuracy_score(y_test,y_pred)
print('Accuracy(for GINI)',score)

clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(x_train, y_train)    
y_pred = clf.predict(x_test[0:int(size*0.2)])

score = accuracy_score(y_test,y_pred)
print('Accuracy(for Entropy)',score)
"""3a complete"""

"""3b"""
depth_b = [2, 4, 8, 10, 15, 30]
acc_train = []
acc_test = []
for depth in depth_b:    
    clf = DecisionTreeClassifier(max_depth = depth,random_state = 0)
    clf.fit(x_train, y_train)    
    y_pred_train = clf.predict(x_train[0:int(size*0.8)])
    y_pred_test = clf.predict(x_test[0:int(size*0.2)])
    score1 = accuracy_score(y_train, y_pred_train)
    score2 = accuracy_score(y_test, y_pred_test)
    acc_train.append(score1)
    acc_test.append(score2)

#plotting graph
plt.plot(acc_train,depth_b)
plt.plot(acc_test,depth_b)
plt.ylabel('Depth')
plt.xlabel('Accuracy')
plt.title('Depth vs Accuracy - 3b')
plt.legend(["Training Accuracy","Testing Accuracy"])
plt.show()

"""3c"""
"""100 Stupms - max_depth = 3"""
#choose 50% of training data randomly for 100 stamps 
stump = 100
depth = 3
print('Generating 100 stumps--')
clf = ensemble(x_train, y_train, x_test, y_test, size, depth, stump)

#accuracy on testing data using majority vote 
y_pred = [0]*(int(size*0.2))
y_pred = maj_vot((size*0.2), clf, x_test, stump)
print('Accuracy score',accuracy_score(y_test,y_pred))
#print(y_pred,y_test)

"""3d"""
"""n Stupms= 10- max_depth = [4, 8, 10, 15, 20, 30]"""

stump = 5
depth_mat = [4, 8, 10, 15, 20, 30]
train_size = x_train.shape[0]
test_size = x_test.shape[0]
for i in depth_mat:
    print('For depth = ',i)
    clf = ensemble(x_train, y_train, x_test, y_test, size, i, stump)
    y_pred = maj_vot(train_size, clf, x_train, stump)
    print('Training Accuracy:: ',accuracy_score(y_train,y_pred))
    y_pred = maj_vot(test_size, clf, x_test, stump)
    print('Training Accuracy:: ',accuracy_score(y_test,y_pred))

"""Testing"""
#stump = 1
#depth_mat = [4, 8, 10, 15, 20, 30]
#train_size = x_train.shape[0]
#test_size = x_test.shape[0]
#for i in depth_mat:
#    print('For depth = ',i)
#    clf = ensemble(x_train, y_train, x_test, y_test, size, i, stump)
#    y_pred = maj_vot(train_size, clf, x_train, stump)
#    print('Training Accuracy:: ',accuracy_score(y_train,y_pred))
#    y_pred = maj_vot(test_size, clf, x_test, stump)
#    print('Training Accuracy:: ',accuracy_score(y_test,y_pred))
    
"""Testing Done"""









