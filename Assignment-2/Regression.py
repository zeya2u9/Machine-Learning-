import joblib
import pandas as pd
import numpy as np
from collections import Iterable
from sklearn import linear_model
import re
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error

class Regression(object):
    """docstring for Regression."""
    def __init__(self, arg):
        super(Regression, self).__init__()
        self.arg = arg
        data = pd.read_table(self.arg,delim_whitespace=True,header=None)
        #file = open('D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\Assignment_2_datasets\\regression_data\\Dataset.data','r')
        mapping = {'M':0, 'F':1, 'I':2}
        data = data.replace({0:mapping})
        self.x = data.iloc[:,0:8]
        self.y = data.iloc[:,8:9]

        #self.fit(x,y)
        
    """You can give any required inputs to the fit()"""
    def fit(self,x,y,path='D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dump\\new.pkl'):
        """Here you can use the fit() from the LinearRegression of sklearn"""
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        # save the model to disk
        lr_model = path
        #joblib.dump(regr, lr_model)
        joblib.dump(regr, open(lr_model, 'wb'))
        
        return lr_model

    def predict(self,X_test,lr_model):
        model = joblib.load(lr_model)
        #result = loaded_model.score(x,y)
        
        col = X_test.shape[1]
        row = X_test.shape[0]
        #print(col,row)
        
        y_val = [0.0]*row
        for i in range(row):
            sum = 0.0
            for j in range(col):
                #print(X_test.iloc[i][j])
                #print(model.coef_[j])
                sum = sum + model.coef_[j] * X_test.iloc[i][j]           
            y_val[i] = model.intercept_ +  sum
            
        return y_val
    def mse(self,y_pred,y_test):
        size = y_test.shape[0]
        sq = 0.0
        for i in range(size):
            diff = y_pred[i] - y_test.iloc[i]
            sq = sq + diff *diff
        mean = sq / size
        return mean
    def get_data(self):
        
        return (self.x,self.y)
    def data_remove(self,x):
        print(x.shape)
        x = x.drop([0],axis=1)
        print(x.shape)
        print(x.iloc[0:1,0:8])
        x.columns = [0,1,2,3,4,5,6]
        #print(X_test.iloc[i:i+1,j:j+1])
        return x


"""Assignment-2 starts here"""
"""1(c)"""
ob1 = Regression('D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\Assignment_2_datasets\\regression_data\\Dataset.data')
X = pd.DataFrame()
Y = pd.DataFrame()
X,Y = ob1.get_data()

plt.scatter(X.iloc[:][0],Y, c='r')
plt.scatter(X.iloc[:][1],Y,c='g')
plt.scatter(X.iloc[:][2],Y,c='b')
plt.scatter(X.iloc[:][3],Y,c='y')
plt.legend(["Feature 1", "Feature 2","Feature 3" ,"Feature 4" ])

plt.scatter(X.iloc[:][4],Y,c='black')
plt.scatter(X.iloc[:][5],Y,c='orange')
plt.scatter(X.iloc[:][6],Y,c='brown')
plt.scatter(X.iloc[:][7],Y,c='dodgerblue')
plt.legend(["Feature 5", "Feature 6","Feature 7" ,"Feature 8" ])
plt.show()

#removing column-0 from dataset and renaming columns
"""optional
X = ob1.data_remove(X)
X.iloc[0:1,0:8]

X.shape
check(X.iloc[fold_index[4][0]:fold_index[4][1]+1,:])

X.iloc[0][1]
Y.iloc[0:10][8]
list(X[:9][1])
abc = list(Y)
Y = Y.astype(int)
Y.dtypes
 optional"""

#1(c)
#saving 5 fold's indexes
f1 = int(X.shape[0]/5)-1
fold_index = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3] , [f1*4+4,f1*5+6]]
f_i = fold_index
#checking predict with fold1-fold4 as training data and fold5 as testing data
X_train = X.iloc[0:fold_index[3][1]+1, :]
Y_train = Y.iloc[0:fold_index[3][1]+1][8]
M = ob1.fit(X_train,Y_train)
y_pred = pd.DataFrame()
y_pred = ob1.predict(X.iloc[fold_index[4][0]:fold_index[4][1]+1,:], M)
#y_pred = ob1.predict(X_train, M)
y_pred[0:1][0:1]

Y_test = Y.iloc[fold_index[4][0]:fold_index[4][1]+1, 0:1]


Y_test.iloc[2][8]

count = 0
for i in range(Y_test.shape[0]):
    a = y_pred[i:i+1][0:1]
    if((a[0] - int(a[0])) >= 0.5):
        a[0] = a[0] +1.0
    b = (Y_test.iloc[i][8])
    print('(',math.floor(a[0]),',',b,')')
    if math.floor(a[0])==b:
        count = count + 1
acc = count/Y_test.shape[0]

"""Calculating MSE for 5 folds"""
f1 = int(X.shape[0]/5)-1
fold_index = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3] , [f1*4+4,f1*5+6]]
f_i = fold_index
m_c = ['D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_c1.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_c2.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_c3.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_c4.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_c5.pkl']
t_mse = [0.0]*5
v_mse = [0.0]*5
t = [0.0]*5
v = [0.0]*5
X_train = X.iloc[f_i[1][0]:f_i[4][1]+1, :] #1
Y_train = Y.iloc[f_i[1][0]:f_i[4][1]+1][8]
M = ob1.fit(X_train,Y_train,m_c[0])
y_pred = ob1.predict(X_train, m_c[0])
t_mse[0] = ob1.mse(y_pred,Y_train)
t[0] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[0][0]:fold_index[0][1]+1, 0:1]
X_test = X.iloc[fold_index[0][0]:fold_index[0][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[0] = ob1.mse(y_pred,Y_test)
v[0] = mean_squared_error(Y_test, y_pred)

#2
X_train = X.iloc[f_i[0][0]:f_i[0][1]+1, :] #2
X_train = X_train.append(X.iloc[f_i[2][0]:f_i[4][1]+1, :], ignore_index=True)
Y_train = Y.iloc[f_i[0][0]:f_i[0][1]+1][8]
Y_train = Y_train.append(Y.iloc[f_i[2][0]:f_i[4][1]+1][8], ignore_index=True)
M = ob1.fit(X_train,Y_train,m_c[1])
y_pred = ob1.predict(X_train, M)
t_mse[1] = ob1.mse(y_pred,Y_train)
t[1] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[1][0]:fold_index[1][1]+1][8]
X_test = X.iloc[fold_index[1][0]:fold_index[1][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[1] = ob1.mse(y_pred,Y_test)
v[1] = mean_squared_error(Y_test, y_pred)

#3
X_train = X.iloc[f_i[0][0]:f_i[1][1]+1, :] 
X_train = X_train.append(X.iloc[f_i[3][0]:f_i[4][1]+1, :], ignore_index=True)
Y_train = Y.iloc[f_i[0][0]:f_i[1][1]+1][8]
Y_train = Y_train.append(Y.iloc[f_i[3][0]:f_i[4][1]+1][8], ignore_index=True)
M = ob1.fit(X_train,Y_train,m_c[2])
y_pred = ob1.predict(X_train, m_c[2])
t_mse[2] = ob1.mse(y_pred,Y_train)
t[2] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[2][0]:fold_index[2][1]+1][8]
X_test = X.iloc[fold_index[2][0]:fold_index[2][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[2] = ob1.mse(y_pred,Y_test)
v[2] = mean_squared_error(Y_test, y_pred)
#4
X_train = X.iloc[f_i[0][0]:f_i[2][1]+1, :] 
X_train = X_train.append(X.iloc[f_i[4][0]:f_i[4][1]+1, :], ignore_index=True)
Y_train = Y.iloc[f_i[0][0]:f_i[2][1]+1][8]
Y_train = Y_train.append(Y.iloc[f_i[4][0]:f_i[4][1]+1][8], ignore_index=True)
M = ob1.fit(X_train,Y_train,m_c[3])
y_pred = ob1.predict(X_train, M)
t_mse[3] = ob1.mse(y_pred,Y_train)
t[3] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[3][0]:fold_index[3][1]+1][8]
X_test = X.iloc[fold_index[3][0]:fold_index[3][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[3] = ob1.mse(y_pred,Y_test)
v[3] = mean_squared_error(Y_test, y_pred)

#5
X_train = X.iloc[f_i[0][0]:f_i[3][1]+1, :] #1
Y_train = Y.iloc[f_i[0][0]:f_i[3][1]+1][8]
M = ob1.fit(X_train,Y_train,m_c[4])
y_pred = ob1.predict(X_train, M)
t_mse[4] = ob1.mse(y_pred,Y_train)
t[4] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[4][0]:fold_index[4][1]+1, 0:1]
X_test = X.iloc[fold_index[4][0]:fold_index[4][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[4] = ob1.mse(y_pred,Y_test)
v[4] = mean_squared_error(Y_test, y_pred)


t_mean = (t_mse[0]+t_mse[1]+t_mse[2]+t_mse[3]+t_mse[4])/5
v_mean = (v_mse[0]+v_mse[1]+v_mse[2]+v_mse[3]+v_mse[4])/5
"""1(c) done"""

"""1(b)"""
"""Calculating MSE for 5 folds using sklearn.predict"""
m_b = ['D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_b1.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_b2.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_b3.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_b4.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_1\\m_b5.pkl']
t_mse = [0.0]*5
v_mse = [0.0]*5
t = [0.0]*5
v = [0.0]*5
X_train = X.iloc[f_i[1][0]:f_i[4][1]+1, :] #1
Y_train = Y.iloc[f_i[1][0]:f_i[4][1]+1][8]
M = ob1.fit(X_train,Y_train,m_b[0])
model = joblib.load(M)
y_pred = model.predict(X_train)
t_mse[0] = ob1.mse(y_pred,Y_train)
t[0] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[0][0]:fold_index[0][1]+1, 0:1]
X_test = X.iloc[fold_index[0][0]:fold_index[0][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[0] = ob1.mse(y_pred,Y_test)
v[0] = mean_squared_error(Y_test, y_pred)

#2
X_train = X.iloc[f_i[0][0]:f_i[0][1]+1, :] #2
X_train = X_train.append(X.iloc[f_i[2][0]:f_i[4][1]+1, :], ignore_index=True)
Y_train = Y.iloc[f_i[0][0]:f_i[0][1]+1][8]
Y_train = Y_train.append(Y.iloc[f_i[2][0]:f_i[4][1]+1][8], ignore_index=True)
M = ob1.fit(X_train,Y_train,m_b[1])
model = joblib.load(M)
y_pred = model.predict(X_train)
t_mse[1] = ob1.mse(y_pred,Y_train)
t[1] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[1][0]:fold_index[1][1]+1][8]
X_test = X.iloc[fold_index[1][0]:fold_index[1][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[1] = ob1.mse(y_pred,Y_test)
v[1] = mean_squared_error(Y_test, y_pred)

#3
X_train = X.iloc[f_i[0][0]:f_i[1][1]+1, :] 
X_train = X_train.append(X.iloc[f_i[3][0]:f_i[4][1]+1, :], ignore_index=True)
Y_train = Y.iloc[f_i[0][0]:f_i[1][1]+1][8]
Y_train = Y_train.append(Y.iloc[f_i[3][0]:f_i[4][1]+1][8], ignore_index=True)
M = ob1.fit(X_train,Y_train,m_b[2])
model = joblib.load(M)
y_pred = model.predict(X_train)
t_mse[2] = ob1.mse(y_pred,Y_train)
t[2] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[2][0]:fold_index[2][1]+1][8]
X_test = X.iloc[fold_index[2][0]:fold_index[2][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[2] = ob1.mse(y_pred,Y_test)
v[2] = mean_squared_error(Y_test, y_pred)
#4
X_train = X.iloc[f_i[0][0]:f_i[2][1]+1, :] 
X_train = X_train.append(X.iloc[f_i[4][0]:f_i[4][1]+1, :], ignore_index=True)
Y_train = Y.iloc[f_i[0][0]:f_i[2][1]+1][8]
Y_train = Y_train.append(Y.iloc[f_i[4][0]:f_i[4][1]+1][8], ignore_index=True)
M = ob1.fit(X_train,Y_train,m_b[3])
model = joblib.load(M)
y_pred = model.predict(X_train)
t_mse[3] = ob1.mse(y_pred,Y_train)
t[3] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[3][0]:fold_index[3][1]+1][8]
X_test = X.iloc[fold_index[3][0]:fold_index[3][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[3] = ob1.mse(y_pred,Y_test)
v[3] = mean_squared_error(Y_test, y_pred)

#5
X_train = X.iloc[f_i[0][0]:f_i[3][1]+1, :] #1
Y_train = Y.iloc[f_i[0][0]:f_i[3][1]+1][8]
M = ob1.fit(X_train,Y_train,m_b[4])
model = joblib.load(M)
y_pred = model.predict(X_train)
t_mse[4] = ob1.mse(y_pred,Y_train)
t[4] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[4][0]:fold_index[4][1]+1, 0:1]
X_test = X.iloc[fold_index[4][0]:fold_index[4][1]+1, :]
y_pred = ob1.predict(X_test, M)
v_mse[4] = ob1.mse(y_pred,Y_test)
v[4] = mean_squared_error(Y_test, y_pred)


t_mean = (t_mse[0]+t_mse[1]+t_mse[2]+t_mse[3]+t_mse[4])/5
v_mean = (v_mse[0]+v_mse[1]+v_mse[2]+v_mse[3]+v_mse[4])/5
"""1(b) done"""

"""1(d)"""
ob1 = Regression('D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\Assignment_2_datasets\\regression_data\\Dataset.data')
X = pd.DataFrame()
Y = pd.DataFrame()
X,Y = ob1.get_data()

f1 = int(X.shape[0]/5)-1
fold_index = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3] , [f1*4+4,f1*5+6]]
f_i = fold_index


t = [0.0]*5
v = [0.0]*5
X_train = X.iloc[f_i[1][0]:f_i[4][1]+1, :] #1
Y_train = Y.iloc[f_i[1][0]:f_i[4][1]+1][8]
M = ob1.fit(X_train,Y_train)
model = joblib.load(M)
y_pred = model.predict(X_train)
t[0] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[0][0]:fold_index[0][1]+1, 0:1]
X_test = X.iloc[fold_index[0][0]:fold_index[0][1]+1, :]
y_pred = model.predict(X_test)
v[0] = mean_squared_error(Y_test, y_pred)

#2
X_train = X.iloc[f_i[0][0]:f_i[0][1]+1, :] #2
X_train = X_train.append(X.iloc[f_i[2][0]:f_i[4][1]+1, :], ignore_index=True)
Y_train = Y.iloc[f_i[0][0]:f_i[0][1]+1][8]
Y_train = Y_train.append(Y.iloc[f_i[2][0]:f_i[4][1]+1][8], ignore_index=True)
M = ob1.fit(X_train,Y_train)
model = joblib.load(M)
y_pred = model.predict(X_train)
t[1] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[1][0]:fold_index[1][1]+1][8]
X_test = X.iloc[fold_index[1][0]:fold_index[1][1]+1, :]
y_pred = model.predict(X_test)
v[1] = mean_squared_error(Y_test, y_pred)

#3
X_train = X.iloc[f_i[0][0]:f_i[1][1]+1, :] 
X_train = X_train.append(X.iloc[f_i[3][0]:f_i[4][1]+1, :], ignore_index=True)
Y_train = Y.iloc[f_i[0][0]:f_i[1][1]+1][8]
Y_train = Y_train.append(Y.iloc[f_i[3][0]:f_i[4][1]+1][8], ignore_index=True)
M = ob1.fit(X_train,Y_train)
model = joblib.load(M)
y_pred = model.predict(X_train)
t[2] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[2][0]:fold_index[2][1]+1][8]
X_test = X.iloc[fold_index[2][0]:fold_index[2][1]+1, :]
y_pred = model.predict(X_test)
v[2] = mean_squared_error(Y_test, y_pred)
#4
X_train = X.iloc[f_i[0][0]:f_i[2][1]+1, :] 
X_train = X_train.append(X.iloc[f_i[4][0]:f_i[4][1]+1, :], ignore_index=True)
Y_train = Y.iloc[f_i[0][0]:f_i[2][1]+1][8]
Y_train = Y_train.append(Y.iloc[f_i[4][0]:f_i[4][1]+1][8], ignore_index=True)
M = ob1.fit(X_train,Y_train)
model = joblib.load(M)
y_pred = model.predict(X_train)
t[3] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[3][0]:fold_index[3][1]+1][8]
X_test = X.iloc[fold_index[3][0]:fold_index[3][1]+1, :]
y_pred = model.predict(X_test)
v[3] = mean_squared_error(Y_test, y_pred)

#5
X_train = X.iloc[f_i[0][0]:f_i[3][1]+1, :] #1
Y_train = Y.iloc[f_i[0][0]:f_i[3][1]+1][8]
M = ob1.fit(X_train,Y_train)
model = joblib.load(M)
y_pred = model.predict(X_train)
t[4] = mean_squared_error(Y_train, y_pred)
Y_test = Y.iloc[fold_index[4][0]:fold_index[4][1]+1, 0:1]
X_test = X.iloc[fold_index[4][0]:fold_index[4][1]+1, :]
y_pred = model.predict(X_test)
v[4] = mean_squared_error(Y_test, y_pred)


t_mean = (t[0]+t[1]+t[2]+t[3]+t[4])/5
v_mean = (v[0]+v[1]+v[2]+v[3]+v[4])/5
"""1(d) done"""











