import joblib
import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import random
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score

class LogRegression(object):
    """docstring for LogRegression."""
    def __init__(self, arg,l2_reg = 0, lambdaa=1,alpha=0.5):
        super(LogRegression, self).__init__()
        self.arg = arg
        self.t_accs = [0.0] * 5000
        mat = scipy.io.loadmat(self.arg)
        x_2 = mat['samples']
        y_2 = mat['labels']
        x_2.shape
        y_2 = y_2.transpose()
        y_2.shape
        
        self.lambdaa = lambdaa
        self.alpha = alpha
        self.l2_reg = l2_reg
        self.x = x_2
        self.y = y_2
        self.loss = [0.0] * 5000
        self.vloss = [0.0] * 5000
        self.tscore = [0.0] * 5000
        self.vscore = [0.0] * 5000
    """You can give any required inputs to the fit()"""
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def fit(self,X_train,Y_train,X_test,Y_test,path='D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dump\\new.pkl'):
        #regr = LogisticRegression() #to save weights
        b0 = 0.1
        b1 = 0.5
        b2 = 0.5
        #self.alpha = 0.001
        size = X_train.shape[0]
        for j in range(400):
            y_outt = [0.0]*size
            y_out=0.0
            y_d = 0.0
            y_f = 0.0
            y_f1 = 0.0
            for i in range(size):
                out = b0 + b1 * X_train[i:i+1,0:1]+ b2 * X_train[i:i+1, 1:2]
                y_out = self.sigmoid(out)
                y_outt[i] = y_out
                y_d = y_d + (Y_train[i:i+1,0:1] - y_out)
                y_f = y_f + (Y_train[i:i+1,0:1] - y_out)*X_train[i:i+1,0:1]
                y_f1 = y_f1 + (Y_train[i:i+1,0:1] - y_out)*X_train[i:i+1,1:2]
            y_d = y_d/size#print(y_d)
            y_f =y_f/size#print(y_f)
            y_f1 = y_f1/size#print(y_f1)  
            b1_add = 0.0
            b2_add = 0.0
            if(self.l2_reg):
                b1_add = (self.lambdaa * b1*self.alpha)
                b2_add = (self.lambdaa * b2*self.alpha)
                val = y_d + self.lambdaa * ((b1*b1)+(b2*b2))
            else:
                val = y_d
            if val<0:
                val = val*(-1)
            
            b0 = b0 + self.alpha * y_d  * 1.0
            b1 = b1 + self.alpha * y_f + b1_add
            b2 = b2 + self.alpha * y_f1 + b2_add
            #regr.intercept_[0] = b0;regr.coef_[0][0] = b1;regr.coef_[0][1] = b2      
            joblib.dump([b0,b1,b2], open(path, 'wb'))
            #print(b0,b1,b2)    
            self.loss[j] = self.cross_loss(Y_train, np.array(y_outt),b1,b2)
            self.tscore[j] = self.score(np.array(y_outt),Y_train)
            y_pred = self.predict(X_test,path)
            self.vloss[j] = self.cross_loss(Y_test, np.array(y_pred),b1,b2)
            self.vscore[j] = self.score(np.array(y_pred),Y_test)
            #print(self.loss[j])
            #if(self.loss[j]<0.4):
            #    break
        #save the model to disk
        joblib.dump([b0,b1,b2], open(path, 'wb'))
        return path
    def cross_loss(self,y_true,y_pred,b1,b2):
        size = y_true.shape[0]
        loss = 0.0
        for i in range(size):
            h = y_pred[i]
            y = y_true[i]
            loss = loss+(-y * np.log(h) - (1 - y) * np.log(1 - h))
        if self.l2_reg==1:
            loss = loss+ self.lambdaa*(b1*b1 + b2*b2)*0.5
        loss= loss/size
        #print('loss = ',loss)
        return loss
    def get_loss_score(self):
        return self.tscore,self.loss,self.vscore,self.vloss
    def predict(self,X_test,model_path='D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dump\\new.pkl'):
        model = joblib.load(model_path)
        b0= model[0]
        b1=model[1]
        b2=model[2]
        y_predict = [0.0]*X_test.shape[0]
        for i in range(X_test.shape[0]):
            out = b0 + b1 * X_test[i:i+1,0:1]+ b2 * X_test[i:i+1, 1:2]
            y_out = self.sigmoid(out)
            y_predict[i] = y_out
        # load the model from disk
        #loaded_model = joblib.load(model)
        #result = loaded_model.score(X_test, Y_test)
        return y_predict
    def mse(self,y_pred,y_test):
        size = y_pred.shape[0]
        sq = 0.0
        for i in range(size):
            diff = y_pred[i:i+1] - y_test[i:i+1]
            sq = sq + diff *diff
        mean = sq / size
        return mean
    def score(self,y_pred,y_test):
        size = y_pred.shape[0]
        count = 0
        for i in range(size):
            if y_pred[i:i+1] > 0.5: 
                label = 1
            else:
                label = 0
            if y_test[i:i+1] == label:
                count = count+1
        score = count/y_pred.shape[0]

        return score
    def get_data(self):       
        return [self.x,self.y]
    def get_taccs(self):
        return self.t_accs
    def transform_data(self,y,cla):
        for i in range(y.shape[0]):
            if y[i]==cla:
                y[i]=1
            else:
                y[i]=0    
        return y
    def shuffle(self,x,y):
        
        for i in range(y.shape[0]):
            if y[i]==3:
                y[i]=0
            else:
                y[i]=1
        c = list(zip(x,y))
        random.shuffle(c)
        a, b = zip(*c)
        
        return np.array(a),np.array(b)
    def get_fold_data(self,X,Y,f_i,i):
        if i==0:
            X_train = X[f_i[1][0]:f_i[4][1]+1, :] #1
            Y_train = Y[f_i[1][0]:f_i[4][1]+1]  
        elif i==1:
            X_train = X[f_i[0][0]:f_i[0][1]+1, :] #2
            X_train = np.append(X_train, X[f_i[2][0]:f_i[4][1]+1, :], axis = 0)
            Y_train = Y[f_i[0][0]:f_i[0][1]+1]
            Y_train = np.append(Y_train, Y[f_i[2][0]:f_i[4][1]+1], axis = 0)   
        elif i==2:
            X_train = X[f_i[0][0]:f_i[1][1]+1, :] #3
            X_train = np.append(X_train, X[f_i[3][0]:f_i[4][1]+1, :], axis = 0)
            Y_train = Y[f_i[0][0]:f_i[1][1]+1]
            Y_train = np.append(Y_train, Y[f_i[3][0]:f_i[4][1]+1], axis = 0)
        elif i==3:
            X_train = X[f_i[0][0]:f_i[2][1]+1, :] #4
            X_train = np.append(X_train, X[f_i[4][0]:f_i[4][1]+1, :], axis = 0)
            Y_train = Y[f_i[0][0]:f_i[2][1]+1]
            Y_train = np.append(Y_train, Y[f_i[4][0]:f_i[4][1]+1], axis = 0)
        else:
            X_train = X[f_i[0][0]:f_i[3][1]+1, :] #5
            Y_train = Y[f_i[0][0]:f_i[3][1]+1]
        X_test = X[f_i[i][0]:f_i[i][1]+1, :]
        Y_test = Y[f_i[i][0]:f_i[i][1]+1]
        
        return X_train,Y_train,X_test,Y_test    
    def classwise_acc(self,prd,tr):
        accs = [0.0]*4
        clas = [0]*4
        class_count = np.unique(tr,return_counts=True)#acces as class_count[1][0]
            
        for i in range(tr.shape[0]):
            if tr[i]==prd[i]:
                clas[tr[i][0]]=clas[tr[i][0]]+1
            
        for i in range(4):
            accs[i] = clas[i]/class_count[1][i]
        return accs
#just checking
def plot_curve(p,r,i):
    print('Fold-',i)
    xx = [i for i in range(400)]
    plt.plot(xx,p[0:400])
    plt.plot(xx,r[0:400])
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Validation" ])
    plt.title("Accuracy")
def plot_loss(q,s,i):
    print('Fold-',i)
    xx = [i for i in range(400)]
    b = [q[i][0][0] for i in range(400)]
    d = [s[i][0][0] for i in range(400)]
    plt.plot(xx,b[0:400])
    plt.plot(xx,d[0:400])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation" ])
    plt.title("Loss")


"""2(a)"""
ob1 = LogRegression('D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dataset_1.mat')
[X,Y] = ob1.get_data()
ob1.sigmoid(0)

np.unique(Y)
sne_df = pd.DataFrame(columns = ['one','two','y'])

x = X[: , [0]]
y = X[: , [1]]
z=Y

sne_df['one'] = X[:,0]
sne_df['two'] = X[:,1]
sne_df['y'] = z[:,]


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="one", y="two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=sne_df,
    legend="full",
    alpha=0.3
)
"""2(a) done"""

"""2(c)"""
#saving 5 fold's indexes
file = 'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dataset_1.mat'
ob1 = LogRegression(file,0,0,0.5)
[X,Y] = ob1.get_data()
f1 = int(X.shape[0]/5)-1
f_i = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3] , [f1*4+4,f1*5+4]]

B_vals = [[0.0]*3 for _ in range(5)]
t_mse = [0.0]*5
t_score = [0.0]*5

m_c = ['D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_c1.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_c2.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_c3.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_c4.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_c5.pkl']
#separting data for training  1
X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,0)#1
M = ob1.fit(X_train, Y_train,X_test,Y_test,m_c[0])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[0] = ob1.mse(y_pred,Y_train)
t_score[0] = ob1.score(y_pred,Y_train)
a,b,c,d=ob1.get_loss_score()
plot_curve(a,c,0)
plot_loss(b,d,0)


X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,1)#2
M = ob1.fit(X_train, Y_train,X_test,Y_test,m_c[1])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[1] = ob1.mse(y_pred,Y_train)
t_score[1] = ob1.score(y_pred,Y_train)
a,b,c,d=ob1.get_loss_score()
plot_curve(a,c,1)
plot_loss(b,d,1)

X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,2)#3
M = ob1.fit(X_train, Y_train,X_test,Y_test,m_c[2])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[2] = ob1.mse(y_pred,Y_train)
t_score[2] = ob1.score(y_pred,Y_train)
a,b,c,d=ob1.get_loss_score()
plot_curve(a,c,2)
plot_loss(b,d,2)

X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,3)#4
M = ob1.fit(X_train,Y_train,X_test,Y_test,m_c[3])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[3] = ob1.mse(y_pred,Y_train)
t_score[3] = ob1.score(y_pred,Y_train)
a,b,c,d=ob1.get_loss_score()
plot_curve(a,c,3)
plot_loss(b,d,3)

X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,4)#5
M = ob1.fit(X_train, Y_train,X_test,Y_test,m_c[4])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[4] = ob1.mse(y_pred,Y_train)
t_score[4] = ob1.score(y_pred,Y_train)
a,b,c,d=ob1.get_loss_score()
plot_curve(a,c,4)
plot_loss(b,d,4)

mean = (t_mse[0]+t_mse[1]+t_mse[2]+t_mse[3]+t_mse[4])/5

#now using predict    
score = [0.0]*5
v_mse = [0.0]*5
y_predd = [0.0] * (f_i[0][1]+1)
for j in range(5):
    count = 0
    X_test = X[f_i[j][0]:f_i[j][1]+1, :]
    Y_test = Y[f_i[j][0]:f_i[j][1]+1, :]
    
    y_predd = ob1.predict(X_test,m_c[j]) 
    y_predd = np.array(y_predd)
    score[j] = ob1.score(y_predd,Y_test)
    v_mse[j] = ob1.mse(y_predd,Y_test)

mean = (v_mse[0]+v_mse[1]+v_mse[2]+v_mse[3]+v_mse[4])/5
"""2(c) done"""

"""2(d) L2 regularization"""
file = 'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dataset_1.mat'
ob1 = LogRegression(file,1,0.001,0.5)
[X,Y] = ob1.get_data()

m_d = ['D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_d1.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_d2.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_d3.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_d4.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_2\\m_d5.pkl']

t_mse = [0.0]*5
t_score = [0.0]*5
B_vals = [[0.0]*3 for _ in range(5)]

f1 = int(X.shape[0]/5)-1
f_i = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3] , [f1*4+4,f1*5+4]]
X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,0)#1
M = ob1.fit(X_train, Y_train,X_test,Y_test,m_d[0])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[0] = ob1.mse(y_pred,Y_train)
t_score[0] = ob1.score(y_pred,Y_train)
a,b,c,d = ob1.get_loss_score()
plot_curve(a,c,0)
plot_loss(b,d,0)


X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,1)#2
M = ob1.fit(X_train, Y_train,X_test,Y_test,m_d[1])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[1] = ob1.mse(y_pred,Y_train)
t_score[1] = ob1.score(y_pred,Y_train)
a,b,c,d=ob1.get_loss_score()
plot_curve(a,c,1)
plot_loss(b,d,1)

X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,2)#3
M = ob1.fit(X_train, Y_train,X_test,Y_test,m_d[2])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[2] = ob1.mse(y_pred,Y_train)
t_score[2] = ob1.score(y_pred,Y_train)
a,b,c,d=ob1.get_loss_score()
plot_curve(a,c,2)
plot_loss(b,d,2)

X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,3)#4
M = ob1.fit(X_train,Y_train,X_test,Y_test,m_d[3])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[3] = ob1.mse(y_pred,Y_train)
t_score[3] = ob1.score(y_pred,Y_train)
a,b,c,d=ob1.get_loss_score()
plot_curve(a,c,3)
plot_loss(b,d,3)

X_train,Y_train,X_test,Y_test = ob1.get_fold_data(X,Y,f_i,4)#5
M = ob1.fit(X_train, Y_train,X_test,Y_test,m_d[4])
y_pred = ob1.predict(X_train,M) 
y_pred = np.array(y_pred)
t_mse[4] = ob1.mse(y_pred,Y_train)
t_score[4] = ob1.score(y_pred,Y_train)
a,b,c,d=ob1.get_loss_score()
plot_curve(a,c,4)
plot_loss(b,d,4)

mean = (t_mse[0]+t_mse[1]+t_mse[2]+t_mse[3]+t_mse[4])/5

#now using predict    
score = [0.0]*5
v_mse = [0.0]*5
y_predd = [0.0] * (f_i[0][1]+1)
for j in range(5):
    count = 0
    X_test = X[f_i[j][0]:f_i[j][1]+1, :]
    Y_test = Y[f_i[j][0]:f_i[j][1]+1, :]
    
    y_predd = ob1.predict(X_test,m_d[j]) 
    y_predd = np.array(y_predd)
    score[j] = ob1.score(y_predd,Y_test)
    v_mse[j] = ob1.mse(y_predd,Y_test)

mean = (v_mse[0]+v_mse[1]+v_mse[2]+v_mse[3]+v_mse[4])/5
"""2{d} end"""

"""2(e) Use sklearn.logisticregression"""
ob3 = LogRegression('D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dataset_1.mat')
[X,Y] = ob3.get_data()
f1 = int(X.shape[0]/5)-1
f_i = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3] , [f1*4+4,f1*5+4]]

X_train = X[0:f_i[3][1]+1, :]#fold-1
Y_train = Y[0:f_i[3][1]+1, :]
clf = LogisticRegression(random_state=0).fit(X_train,Y_train)
clf.predict(X_train)
clf.predict_proba(X_train)
clf.score(X_train,Y_train)

X_train = X[f_i[0][0]:f_i[0][1]+1, :] #2
X_train = np.append(X_train, X[f_i[2][0]:f_i[4][1]+1, :], axis = 0)
Y_train = Y[f_i[0][0]:f_i[0][1]+1, :]
Y_train = np.append(Y_train, Y[f_i[2][0]:f_i[4][1]+1, :], axis = 0)
clf1 = LogisticRegression(random_state=0).fit(X_train,Y_train)
clf1.predict(X_train)
clf1.predict_proba(X_train)
clf1.score(X_train,Y_train)

X_train = X[f_i[0][0]:f_i[1][1]+1, :] #3
X_train = np.append(X_train, X[f_i[3][0]:f_i[4][1]+1, :], axis = 0)
Y_train = Y[f_i[0][0]:f_i[1][1]+1, :]
Y_train = np.append(Y_train, Y[f_i[3][0]:f_i[4][1]+1, :], axis = 0)
clf2 = LogisticRegression(random_state=0).fit(X_train,Y_train)
clf2.predict(X_train)
clf2.predict_proba(X_train)
clf2.score(X_train,Y_train)

X_train = X[f_i[0][0]:f_i[2][1]+1, :] #4
X_train = np.append(X_train, X[f_i[4][0]:f_i[4][1]+1, :], axis = 0)
Y_train = Y[f_i[0][0]:f_i[2][1]+1, :]
Y_train = np.append(Y_train, Y[f_i[4][0]:f_i[4][1]+1, :], axis = 0)
clf3 = LogisticRegression(random_state=0).fit(X_train,Y_train)
clf3.predict(X_train)
clf3.predict_proba(X_train)
clf3.score(X_train,Y_train)

X_train = X[f_i[0][0]:f_i[3][1]+1, :] #5
Y_train = Y[f_i[0][0]:f_i[3][1]+1, :]
clf4 = LogisticRegression(random_state=0).fit(X_train,Y_train)
clf4.predict(X_train)
clf4.predict_proba(X_train)
clf4.score(X_train,Y_train)

clf_all = [clf,clf1,clf2,clf3,clf4]
score = [0.0]*5
v_mse = [0.0]*5
for j in range(5):
    count = 0
    X_test = X[f_i[j][0]:f_i[j][1]+1, :]
    Y_test = Y[f_i[j][0]:f_i[j][1]+1, :]
    score[j] = clf_all[j].score(X_test,Y_test)
    #y_predd = clf_all[j].predict_proba(X_test) 
    #v_mse[j] = ob3.mse(y_predd,Y_test)
"""2(e) end"""

"""3(a)"""
ob4 = LogRegression('D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dataset_2.mat')
X,Y = ob4.get_data()
np.unique(Y)
sne_df = pd.DataFrame(columns = ['one','two','y'])

x = X[: , [0]]
y = X[: , [1]]
z=Y

sne_df['one'] = X[:,0]
sne_df['two'] = X[:,1]
sne_df['y'] = z[:,]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="one", y="two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=sne_df,
    legend="full",
    alpha=0.3
)
"""3(a) done"""
#saving 5 fold's indexes
f1 = int(X.shape[0]/5)-1
fold_index = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3] , [f1*4+4,f1*5+4]]

#seeing distribution of data with classes
plt.xlabel("Feature1,2 values")
plt.ylabel("Classes")
plt.scatter(X[:,0:1],Y, c='r')
plt.scatter(X[:,1:2],Y,c='g')
plt.legend(["Feature 1", "Feature 2" ])

"""3(b)"""
file = 'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dataset_2.mat'
obj = LogRegression(file,1,0.001,0.5)
X,Y_orig = obj.get_data()
classes = np.unique(Y_orig,return_counts=True)
classes[1][0]
#saving 5 fold's indexes
f1 = int(X.shape[0]/5)-1
f_i = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3], [f1*4+4,f1*5+4]]

data_0 = np.array([[0.0,0.0]]*classes[1][0])
data_1 = np.array([[0.0,0.0]]*classes[1][1])
data_2 = np.array([[0.0,0.0]]*classes[1][2])
data_3 = np.array([[0.0,0.0]]*classes[1][3])
data_0.shape[0]
j=0;k=0;l=0;m=0
for i in range(X.shape[0]):
    if Y_orig[i] == 0:
        data_0[j][0] = X[i][0]; data_0[j][1] = X[i][1]; j=j+1
    if Y_orig[i] == 1:
        data_1[k][0] = X[i][0]; data_1[k][1] = X[i][1]; k=k+1
    if Y_orig[i] == 2:
        data_2[l][0] = X[i][0]; data_2[l][1] = X[i][1]; l=l+1
    if Y_orig[i] == 3:
        data_3[m][0] = X[i][0]; data_3[m][1] = X[i][1]; m=m+1
        
np.random.seed(42)
y_0 = np.array([[0]]*2500)
y_1 = np.array([[1]]*2500)
y_2 = np.array([[2]]*2500)
y_3 = np.array([[3]]*2500)
M1=np.append(data_0,data_3,axis=0) #5000
M2=np.append(data_1,data_3,axis=0) #5000
M3=np.append(data_2,data_3,axis=0) #5000
M1y=np.append(y_0,y_3,axis=0)
M2y=np.append(y_1,y_3,axis=0)
M3y=np.append(y_2,y_3,axis=0)

M1,M1y = obj.shuffle(M1,M1y) 
M2,M2y = obj.shuffle(M2,M2y)
M3,M3y = obj.shuffle(M3,M3y)
#data prep done

#only saving fold-1 weights
m_b = [['D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_b1.pkl',
        'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_b2.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_b3.pkl'],
       ['D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_b4.pkl',
        'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_b5.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_b6.pkl'],
       ['D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_b7.pkl',
        'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_b8.pkl',
        'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_b9.pkl']]

ref = 3 #making 3 as reference class
f1 = int(M1.shape[0]/5)-1
f_i = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3], [f1*4+4,f1*5+4]]
M1y[0:10]
v_mse = [0.0]*5
v_score = [0.0]*5
class_acc = [[0.0]*4 for _ in range(5)]
for i in range(5):   #for 5 folds
    print('Fold-',i, 'starting')
    B_vals = [[0.0]*3 for _ in range(3)]
    for j in range(3):
        if j==0:
            X_train,Y_train,X_test,Y_test = obj.get_fold_data(M1,M1y,f_i,i) 
        if j==1:
            X_train,Y_train,X_test,Y_test = obj.get_fold_data(M2,M2y,f_i,i) 
        if j==2:
            X_train,Y_train,X_test,Y_test = obj.get_fold_data(M3,M3y,f_i,i) 
        M = obj.fit(X_train, Y_train,X_test,Y_test,m_b[i][j])
        model = joblib.load(m_b[i][j])
        B_vals[j][0] = model[0];B_vals[j][1] = model[1];B_vals[j][2] = model[2]
    #calculate p(A|x) ....
    X,Y = obj.get_data()
    a,b,X_test,Y_test = obj.get_fold_data(X,Y,f_i,i) 
    pred = [0.0]*X_test.shape[0]
    for k in range(X_test.shape[0]):
        out_0 = math.exp(B_vals[0][0] + B_vals[0][1] * X_test[k:k+1,0:1]+ B_vals[0][2] * X_test[k:k+1, 1:2])
        out_1 = math.exp(B_vals[1][0] + B_vals[1][1] * X_test[k:k+1,0:1]+ B_vals[1][2] * X_test[k:k+1, 1:2])
        out_2 = math.exp(B_vals[2][0] + B_vals[2][1] * X_test[k:k+1,0:1]+ B_vals[2][2] * X_test[k:k+1, 1:2])        
        prob_3 = 1/(1+out_0+out_1+out_2) 
        prob_2 = out_2 * prob_3
        prob_1 = out_1 * prob_3
        prob_0 = out_0 * prob_3
        max = prob_0
        label = 0
        if prob_1>max:
            max = prob_1
            label = 1
        if prob_2>max:
            max = prob_2
            label = 2
        if prob_3>max:
            max = prob_3
            label = 3
        pred[k] = label
    v_score[i] = accuracy_score(Y_test,pred)
    ##introduce classwise accuracy
    class_acc[i]=obj.classwise_acc(pred,Y_test)
    print(class_acc[i])
    print('Fold-',i, 'done')
print(v_score)
"""3(b) done"""


""" 3(c) one vs rest"""
file = 'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dataset_2.mat'
obj = LogRegression(file,1,0.001,0.5)
X,Y_orig = obj.get_data()
classes = np.unique(Y_orig)
#saving 5 fold's indexes
f1 = int(X.shape[0]/5)-1
f_i = [[0,f1], [f1+1,f1*2+1], [f1*2+2,f1*3+2], [f1*3+3,f1*4+3], [f1*4+4,f1*5+4]]

m_c = ['D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_c1.pkl',
        'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_c2.pkl',
        'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_c3.pkl',
       'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\models_3\\m_c4.pkl']

v_mse = [0.0]*5
v_score = [0.0]*5
class_acc = [[0.0]*4 for _ in range(5)]
for i in range(5):    
    print('fold-',i)
    obj = LogRegression(file,1,0.001,0.5)
    X,Y = obj.get_data()

    B_vals = [[0.0]*3 for _ in range(4)]
    pred = [[0.0]*X_test.shape[0] for _ in range(4)]
    for j in classes:
        obj = LogRegression(file,1,0.001,0.5)
        X,Y = obj.get_data()
        X_train,Y_train,X_test,Y_test = obj.get_fold_data(X,Y,f_i,i)
        Y_train = obj.transform_data(Y_train,j)
        Y_test = obj.transform_data(Y_test,j)
        M = obj.fit(X_train, Y_train,X_test,Y_test,m_c[j])
        model = joblib.load(M)
        B_vals[j][0] = model[0];B_vals[j][1] = model[1];B_vals[j][2] = model[2]
        pred[j] = obj.predict(X_test,m_c[j])
        pred[j] = np.array(pred[j])
    
    obj = LogRegression(file,1,0.001,0.5)
    X,Y = obj.get_data()
    a,b,X_t,Y_t = obj.get_fold_data(X,Y,f_i,i)
    final_pred = [0]*X_t.shape[0]
    for k in range(X_t.shape[0]):
        max = 0.0
        if pred[0][k]>max:
            max = pred[0][k]
            label = 0
        if pred[1][k]>max:
            max = pred[1][k]
            label = 1
        if pred[2][k]>max:
            max = pred[2][k]
            label = 2
        if pred[3][k]>max:
            max = pred[3][k]
            label = 3
        final_pred[k] = label
    v_score[i] = accuracy_score(Y_t,final_pred) 
    class_acc[i]=obj.classwise_acc(final_pred,Y_t)
    print(class_acc[i])
    print('Fold-',i, 'done')
"""3(d)"""
file = 'D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-2\\dataset_2.mat'
obj = LogRegression(file,1,0.01,0.01)
X,Y_orig = obj.get_data()

kf = KFold(n_splits=5, random_state=0, shuffle=True)
regr = LogisticRegression()
ovo = OneVsOneClassifier(regr)
cross_val_score(ovo, X,Y_orig,scoring='accuracy', cv=kf, n_jobs=-1)

ovr = OneVsRestClassifier(regr)
cross_val_score(ovr, X,Y_orig,scoring='accuracy', cv=kf, n_jobs=-1)

"""3(d) done"""






