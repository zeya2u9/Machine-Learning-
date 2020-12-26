# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:58:55 2020

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
from mpl_toolkits.mplot3d import Axes3D
##

matt = scipy.io.loadmat('dataset_1.mat')
#print(matt['samples'])
#print(matt)
x_1 = matt['samples']
y_1 = matt['labels']
#print(y_1)
np.unique(y_1)
#print(x_1[4000,:,:])
y_1 = y_1.transpose()

#f = x_1[4001,:,:]  # retrieve a grayscale image
#plt.imshow(f, cmap=plt.cm.gray)
#1a
for i in range(10):
    k=0
    for j in range(10):
        while(y_1[k]!=i):
            k=k+1
        f = x_1[k,:,:]
        k=k+1
        plt.imshow(f, cmap=plt.cm.gray)
        plt.show()

#1b
mat = scipy.io.loadmat('D:\\Research 1.0\\2 Machine Learning\\Assignments\\A-1\\ML(PG)_assignment_1\\dataset_2.mat')
#print(mat)        
x_2 = mat['samples']
y_2 = mat['labels']
#print(x1)
x = x_2[: , [0]]
y = x_2[: , [1]]
z=y_2.transpose()
#print(z)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#taking average
xx = (x+y)/2
yy = z

plt.scatter(xx,yy, c='g' )
plt.show()

##another one
"""Final Figure"""
sne_df = pd.DataFrame(columns = ['one','two','y'])

sne_df['one'] = x_2[:,0]
sne_df['two'] = x_2[:,1]
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

#1c
x_1.shape
N=50000

[X, labels] = [x_1[0:N,:,:],y_1]
nsamples, nx, ny = X.shape
data_set = X.reshape((nsamples,(nx*ny))) 

data_set.shape

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_set)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

sne_df = pd.DataFrame(columns = ['tsne-2d-one','tsne-2d-two','y'])

sne_df['tsne-2d-one'] = tsne_results[:,0]
sne_df['tsne-2d-two'] = tsne_results[:,1]
sne_df['y'] = y_1[0:N,]


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=sne_df,
    legend="full",
    alpha=0.3
)


#1d
N=50000

[X, labels] = [x_1[0:N,:,:],y_1]
nsamples, nx, ny = X.shape
data_set = X.reshape((nsamples,(nx*ny))) 

time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_set)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

sne_df = pd.DataFrame(columns = ['tsne-3d-one','tsne-3d-two','tsne-3d-three','y'])

sne_df['tsne-3d-one'] = tsne_results[:,0]
sne_df['tsne-3d-two'] = tsne_results[:,1]
sne_df['tsne-3d-three'] = tsne_results[:,2]
sne_df['y'] = y_1[0:N,]

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=sne_df.loc[:]["tsne-3d-one"], 
    ys=sne_df.loc[:]["tsne-3d-two"], 
    zs=sne_df.loc[:]["tsne-3d-three"], 
    c=sne_df.loc[:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('tsne-3d-one')
ax.set_ylabel('tsne-3d-two')
ax.set_zlabel('tsne-3d-three')
plt.show()










